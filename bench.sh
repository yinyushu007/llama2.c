#!/usr/bin/env bash
# bench.sh — benchmark llama2.c inference on this machine.
#
# Measures:
#   1. throughput under different compiler flags (-O3, -O3 + native, -Ofast, -Ofast + native)
#   2. whether -Ofast changes the generated text (greedy decoding)
#   3. OpenMP thread scaling (1, 2, 4, ... threads)
# Then runs analyse_bench.py to draw the chart and write results/README_section.md.
#
# Usage, from the root of the llama2.c repo:
#   bash bench.sh
# Options (environment variables):
#   MODEL=stories42M.bin bash bench.sh   # use a bigger checkpoint
#   RUNS=10 bash bench.sh                # timed runs per configuration (default 5)
#   SKIP_OMP=1 bash bench.sh             # skip the thread-scaling part

set -eu

MODEL="${MODEL:-stories15M.bin}"
RUNS="${RUNS:-5}"
STEPS="${STEPS:-256}"
PROMPT="${PROMPT:-Once upon a time}"
OUT="results"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f run.c ]; then
  echo "Run this script from the root of the llama2.c repository (run.c not found)." >&2
  exit 1
fi
mkdir -p "$OUT/bin" "$OUT/outputs"
printf 'bin/\n' > "$OUT/.gitignore"   # compiled binaries stay out of the repo

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ---------- compiler ----------
if [ -z "${CC:-}" ]; then
  if command -v gcc >/dev/null 2>&1; then CC=gcc; else CC=cc; fi
fi

compiles_with() {  # succeeds if a trivial C program builds with the given flags
  printf 'int main(void){return 0;}\n' > "$TMP/t.c"
  "$CC" "$@" "$TMP/t.c" -o "$TMP/t" >/dev/null 2>&1
}

NATIVE=""
for f in -march=native -mcpu=native; do
  if compiles_with "$f"; then NATIVE="$f"; break; fi
done
if [ -z "$NATIVE" ]; then
  echo "Note: compiler accepts neither -march=native nor -mcpu=native; native builds skipped."
fi

# ---------- system info ----------
if [ "$(uname -s)" = "Darwin" ]; then
  CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  OS="macOS $(sw_vers -productVersion 2>/dev/null || echo)"
  PHYS="$(sysctl -n hw.physicalcpu 2>/dev/null || echo 1)"
  LOGI="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 1)"
else
  CPU="$(LC_ALL=C lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1)"
  if [ -z "$CPU" ]; then CPU="$(uname -m)"; fi
  OS="$( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || uname -sr)"
  LOGI="$(nproc 2>/dev/null || echo 1)"
  PHYS="$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l | tr -d ' ')"
  if [ -z "$PHYS" ] || [ "$PHYS" = "0" ]; then PHYS="$LOGI"; fi
fi
COMPILER="$("$CC" --version 2>/dev/null | head -n 1)"

# ---------- model ----------
if [ ! -f "$MODEL" ]; then
  echo "Downloading $MODEL ..."
  curl -fL -o "$MODEL" "https://huggingface.co/karpathy/tinyllamas/resolve/main/$MODEL"
fi

run_once() {  # $1 = binary, $2 = file for the generated text; prints tok/s
  "$1" "$MODEL" -t 0.0 -n "$STEPS" -i "$PROMPT" > "$2" 2> "$TMP/err"
  sed -n 's/^achieved tok\/s: *//p' "$TMP/err" | tail -n 1
}

timed_runs() {  # $1 = binary, $2 = label for CSV, $3 = CSV file, $4 = output text file
  run_once "$1" "$4" >/dev/null   # warm-up run, not recorded
  i=1
  while [ "$i" -le "$RUNS" ]; do
    tps="$(run_once "$1" "$4")"
    if [ -z "$tps" ]; then
      echo "Could not read tok/s from the output of $1:" >&2
      cat "$TMP/err" >&2
      exit 1
    fi
    echo "$2,$i,$tps" >> "$3"
    printf '    run %d: %s tok/s\n' "$i" "$tps"
    i=$((i + 1))
  done
}

# ---------- 1. compiler flags ----------
BUILDS="O3|-O3"
if [ -n "$NATIVE" ]; then BUILDS="$BUILDS
O3-native|-O3 $NATIVE"; fi
BUILDS="$BUILDS
Ofast|-Ofast"
if [ -n "$NATIVE" ]; then BUILDS="$BUILDS
Ofast-native|-Ofast $NATIVE"; fi

echo "build,flags,run,tok_s" > "$OUT/flags.csv"
echo "== Compiler flags ($RUNS runs each, $STEPS tokens, model $MODEL) =="
while IFS='|' read -r name flags; do
  if [ -z "$name" ]; then continue; fi
  echo "  $name: $CC $flags"
  # shellcheck disable=SC2086
  "$CC" $flags -o "$OUT/bin/run_$name" run.c -lm
  timed_runs "$OUT/bin/run_$name" "$name,$flags" "$OUT/flags.csv" "$OUT/outputs/$name.txt"
done <<EOF
$BUILDS
EOF

# ---------- 2. accuracy ----------
if [ -n "$NATIVE" ]; then FAST="Ofast-native"; else FAST="Ofast"; fi
if cmp -s "$OUT/outputs/O3.txt" "$OUT/outputs/$FAST.txt"; then ACC="identical"; else ACC="different"; fi
echo "== Accuracy: greedy output of O3 vs $FAST is $ACC =="

# ---------- 3. OpenMP thread scaling ----------
OMP="no"
if [ "${SKIP_OMP:-0}" != "1" ]; then
  OMP_BIN="$OUT/bin/run_omp"
  # shellcheck disable=SC2086
  if "$CC" -Ofast $NATIVE -fopenmp -o "$OMP_BIN" run.c -lm 2>/dev/null; then
    OMP="yes"
  elif [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    LIBOMP="$(brew --prefix libomp 2>/dev/null || true)"
    if [ -n "$LIBOMP" ] && [ -d "$LIBOMP/lib" ]; then
      # shellcheck disable=SC2086
      if "$CC" -Ofast $NATIVE -Xpreprocessor -fopenmp -I"$LIBOMP/include" -o "$OMP_BIN" run.c \
           -L"$LIBOMP/lib" -lomp -lm 2>/dev/null; then
        OMP="yes"
      fi
    fi
  fi

  if [ "$OMP" = "yes" ]; then
    THREADS="$( { t=1; while [ "$t" -le "$LOGI" ]; do echo "$t"; t=$((t * 2)); done; echo "$PHYS"; echo "$LOGI"; } | sort -n | uniq )"
    echo "threads,run,tok_s" > "$OUT/threads.csv"
    echo "== OpenMP thread scaling (physical cores: $PHYS, logical: $LOGI) =="
    for t in $THREADS; do
      echo "  OMP_NUM_THREADS=$t"
      export OMP_NUM_THREADS="$t"
      timed_runs "$OMP_BIN" "$t" "$OUT/threads.csv" "$TMP/omp_out.txt"
    done
    unset OMP_NUM_THREADS
  else
    echo "OpenMP build failed, thread scaling skipped."
    if [ "$(uname -s)" = "Darwin" ]; then
      echo "On macOS install OpenMP with:  brew install libomp   then run this script again."
    fi
  fi
fi

# ---------- system.txt + analysis ----------
cat > "$OUT/system.txt" <<EOF
cpu=$CPU
os=$OS
compiler=$COMPILER
physical_cores=$PHYS
logical_cores=$LOGI
model=$MODEL
runs=$RUNS
steps=$STEPS
prompt=$PROMPT
native_flag=$NATIVE
fast_build=$FAST
accuracy=$ACC
omp=$OMP
EOF

if command -v python3 >/dev/null 2>&1; then
  python3 "$HERE/analyse_bench.py"
else
  echo "python3 not found: install it, then run  python3 analyse_bench.py"
fi
