# llama2.c — CPU Inference Benchmarking

Fork of [karpathy/llama2.c](https://github.com/karpathy/llama2.c), a ~700-line C inference engine for LLaMA 2, used to measure how compiler flags and OpenMP threading affect inference throughput on CPU. All numbers come from [`bench.sh`](bench.sh) and [`analyse_bench.py`](analyse_bench.py).

**Test setup:** Apple M4 (10 physical / 10 logical cores) · macOS 26.3.1 · Apple clang version 17.0.0 (clang-1700.6.4.2) · model `stories15M.bin` · 256 tokens per run, greedy decoding (`-t 0.0`) · mean ± sd of 5 runs after one warm-up run.

## Compiler flags

| Build | Flags | Throughput (tok/s) | Speed-up vs -O3 |
|---|---|---|---|
| O3 | `-O3` | 98.6 ± 1.7 | 1.00× |
| O3-native | `-O3 -march=native` | 100 ± 0.1 | 1.02× |
| Ofast | `-Ofast` | 378 ± 5.0 | 3.83× |
| Ofast-native | `-Ofast -march=native` | 381 ± 12.1 | 3.87× |

`-Ofast -march=native` reaches **381 tok/s, 3.9× the -O3 baseline**, and almost all of that gain comes from `-Ofast` alone.

- **`-Ofast` (3.8×):** it turns on `-ffast-math`. The inner loop of `matmul()` is a dot product (`val += w[i * n + j] * x[j]`). Strict IEEE 754 rules do not allow these floating-point additions to be reordered, so under `-O3` the compiler cannot split the sum across vector lanes. `-ffast-math` allows the reordering, so the loop can be vectorised, at the cost of results that are no longer guaranteed to be bit-identical.
- **`-march=native` (≤ 2%):** on the M4 it makes almost no difference, because the default arm64 target for macOS already targets Apple Silicon and enables the NEON vector instructions.

## OpenMP thread scaling

![Throughput vs OpenMP threads](results/threads.png)

| Threads | Throughput (tok/s) | vs 1 thread |
|---|---|---|
| 1 | 378 ± 15.0 | 1.00× |
| 2 | 644 ± 2.5 | 1.70× |
| 4 | 562 ± 4.8 | 1.49× |
| 8 | 411 ± 6.3 | 1.09× |
| 10 | 374 ± 5.3 | 0.99× |

Throughput peaks at **2 threads (644 tok/s, 1.7× one thread)**. Without OpenMP the same flags give 381 tok/s.

**Interpretation:** `stories15M` is a small model (dim = 288), so each of the 43 matrix multiplications per token gives every thread only microseconds of work, and the cost of starting and synchronising the OpenMP threads for each one soon outweighs the gain. This is likely why throughput peaks at just 2 threads. Beyond 4 threads, some threads also have to run on the M4's slower efficiency cores (4 performance + 6 efficiency cores); since the work is split evenly, every matmul then waits for the slowest thread. If this explanation is right, a larger model (`MODEL=stories110M.bin bash bench.sh`) should move the peak to more threads.

## Accuracy check

With greedy decoding, the -O3 and `-Ofast -march=native` builds generated **identical text** for the test prompt, so on this run `-ffast-math` did not change the output. It does not guarantee bit-identical arithmetic, so results may still differ for other prompts or models.

## Reproduce

```bash
git clone https://github.com/yinyushu007/llama2.c.git
cd llama2.c
bash bench.sh                         # downloads stories15M.bin if missing
MODEL=stories42M.bin bash bench.sh    # optional: a bigger model
```

---

*Everything below is the original llama2.c README.*
