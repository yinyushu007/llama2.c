"""Summarise the results of bench.sh.

Reads results/system.txt, results/flags.csv and (if present) results/threads.csv,
then writes:
  results/threads.png        thread-scaling chart
  results/README_section.md  README section with the measured numbers filled in
"""
import csv
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

OUT = Path("results")
REPO_URL = "https://github.com/yinyushu007/llama2.c.git"


def load_system():
    info = {}
    for line in (OUT / "system.txt").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key] = value.strip()
    return info


def summarise(path, key):
    """Return {key: (mean, sd, n)} in file order, plus {key: flags} when present."""
    groups, flags = OrderedDict(), {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            groups.setdefault(row[key], []).append(float(row["tok_s"]))
            if "flags" in row:
                flags[row[key]] = row["flags"]
    stats = OrderedDict()
    for k, values in groups.items():
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[k] = (statistics.mean(values), sd, len(values))
    return stats, flags


def fmt(v):
    """Whole numbers for large values, one decimal for small ones."""
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"


def plot_threads(threads, phys, logi, no_omp_mean, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator, FixedFormatter
    except ImportError:
        print("matplotlib not installed: run  pip3 install matplotlib  then  python3 analyse_bench.py")
        return False

    xs = sorted(threads, key=int)
    x = [int(t) for t in xs]
    y = [threads[t][0] for t in xs]
    err = [threads[t][1] for t in xs]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(x, y, yerr=err, marker="o", capsize=3, linewidth=2, color="#1f6feb",
                label="-Ofast + OpenMP (mean ± sd)")
    if no_omp_mean is not None:
        ax.axhline(no_omp_mean, linestyle=":", color="#6e7781",
                   label=f"no OpenMP: {no_omp_mean:.0f} tok/s")
    ax.axvline(phys, linestyle="--", color="#cf222e", linewidth=1,
               label=f"physical cores = {phys}")
    if logi != phys:
        ax.axvline(logi, linestyle="--", color="#bf8700", linewidth=1,
                   label=f"logical cores = {logi}")

    best = max(xs, key=lambda t: threads[t][0])
    last = best == xs[-1]
    ax.annotate(f"peak {threads[best][0]:.0f} tok/s",
                xy=(int(best), threads[best][0]), xytext=(-10 if last else 0, 12),
                textcoords="offset points", ha="right" if last else "center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in x]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("OMP_NUM_THREADS")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("llama2.c CPU inference: throughput vs OpenMP threads")
    ax.set_ylim(bottom=0, top=max(v + e for v, e in zip(y, err)) * 1.18)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    if not (OUT / "system.txt").exists():
        sys.exit("results/system.txt not found: run bash bench.sh first.")
    info = load_system()
    flags, flag_text = summarise(OUT / "flags.csv", "build")
    base = flags["O3"][0]
    fast = info["fast_build"]
    runs, steps, model = info["runs"], info["steps"], info["model"]

    lines = [
        "# llama2.c — CPU Inference Benchmarking",
        "",
        "Fork of [karpathy/llama2.c](https://github.com/karpathy/llama2.c), a ~700-line C inference "
        "engine for LLaMA 2, used to measure how compiler flags and OpenMP threading affect "
        "inference throughput on CPU. All numbers come from [`bench.sh`](bench.sh) and "
        "[`analyse_bench.py`](analyse_bench.py).",
        "",
        f"**Test setup:** {info['cpu']} ({info['physical_cores']} physical / "
        f"{info['logical_cores']} logical cores) · {info['os']} · {info['compiler']} · "
        f"model `{model}` · {steps} tokens per run, greedy decoding (`-t 0.0`) · "
        f"mean ± sd of {runs} runs after one warm-up run.",
        "",
        "## Compiler flags",
        "",
        "| Build | Flags | Throughput (tok/s) | Speed-up vs -O3 |",
        "|---|---|---|---|",
    ]
    for name, (mean, sd, _) in flags.items():
        lines.append(f"| {name} | `{flag_text[name]}` | {fmt(mean)} ± {fmt(sd)} | {mean / base:.2f}× |")
    fast_mean = flags[fast][0]
    lines += [
        "",
        f"`{flag_text[fast]}` reaches **{fast_mean:.0f} tok/s, {fast_mean / base:.1f}× the -O3 baseline**.",
        "",
        "- `-march=native` / `-mcpu=native` lets the compiler target the host CPU's full instruction "
        "set, so the loops in `matmul()` (where most inference time is spent) can use wider vector "
        "instructions.",
        "- `-Ofast` adds `-ffast-math`, which lets the compiler reorder floating-point operations such "
        "as dot-product reductions, enabling further vectorisation at the cost of strict IEEE 754 "
        "semantics.",
        "",
    ]

    # thread scaling
    threads_csv = OUT / "threads.csv"
    if info.get("omp") == "yes" and threads_csv.exists():
        threads, _ = summarise(threads_csv, "threads")
        phys, logi = int(info["physical_cores"]), int(info["logical_cores"])
        drew = plot_threads(threads, phys, logi, fast_mean, OUT / "threads.png")
        order = sorted(threads, key=int)
        best = max(order, key=lambda t: threads[t][0])
        one = threads[order[0]][0]
        lines += ["## OpenMP thread scaling", ""]
        if drew:
            lines += ["![Throughput vs OpenMP threads](results/threads.png)", ""]
        lines += ["| Threads | Throughput (tok/s) | vs 1 thread |", "|---|---|---|"]
        for t in order:
            mean, sd, _ = threads[t]
            lines.append(f"| {t} | {fmt(mean)} ± {fmt(sd)} | {mean / one:.2f}× |")
        lines += [
            "",
            f"Throughput peaks at **{best} threads ({threads[best][0]:.0f} tok/s, "
            f"{threads[best][0] / one:.1f}× one thread)**. Without OpenMP the same flags give "
            f"{fast_mean:.0f} tok/s.",
            "",
            "**Interpretation:** [Write 1–2 sentences in your own words: why does throughput stop "
            "improving (or fall) beyond the peak? Think about model size vs threading overhead, "
            "and physical vs logical cores.]",
            "",
        ]

    # accuracy
    if info["accuracy"] == "identical":
        acc = (f"With greedy decoding, the -O3 and `{flag_text[fast]}` builds generated **identical "
               "text** for the test prompt, so on this run `-ffast-math` did not change the output. "
               "It does not guarantee bit-identical arithmetic, so results may still differ for other "
               "prompts or models.")
    else:
        acc = (f"With greedy decoding, the -O3 and `{flag_text[fast]}` builds generated **different "
               "text** for the same prompt: the relaxed floating-point rules of `-ffast-math` changed "
               "which token was most likely at some step. Both outputs are in `results/outputs/`.")
    lines += ["## Accuracy check", "", acc, ""]

    lines += [
        "## Reproduce",
        "",
        "```bash",
        f"git clone {REPO_URL}",
        "cd llama2.c",
        "bash bench.sh                         # downloads stories15M.bin if missing",
        "MODEL=stories42M.bin bash bench.sh    # optional: a bigger model",
        "```",
        "",
        "---",
        "",
        "*Everything below is the original llama2.c README.*",
        "",
    ]
    (OUT / "README_section.md").write_text("\n".join(lines))

    print()
    print("Done. Files written:")
    print("  results/README_section.md  -> paste over the top section of README.md")
    if (OUT / "threads.png").exists():
        print("  results/threads.png        -> upload to the results/ folder of your repo")
    print()
    print(f"For your CV: -O3 baseline {base:.0f} tok/s -> {fast_mean:.0f} tok/s with "
          f"{flag_text[fast]} ({fast_mean / base:.1f}x), {model}, {info['cpu']}")


if __name__ == "__main__":
    main()
