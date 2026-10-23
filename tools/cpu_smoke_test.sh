#!/usr/bin/env bash
# Fresh-environment test of the repo on any CPU machine (no GPU needed), about 10 minutes plus downloads.
# It stands in for a RunPod pod: a new venv with torch 2.8 (CPU wheel), a copy of this repo, setup.sh --no-gemma,
# then every exercise except 3 (Gemma). Exercises 4-5 run as CPU copies, with "cuda" changed to "cpu" in them and in ROME.
#   bash tools/cpu_smoke_test.sh                 # reuses $HF_HOME (or the default cache) for models
#   bash tools/cpu_smoke_test.sh --fresh-cache   # downloads everything (about 11 GB) into a temporary cache
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
T=$(mktemp -d -t rlr-smoke-XXXX)
echo "work dir: $T"
mkdir -p "$T/repo" && tar -C "$REPO" --exclude=./third_party --exclude=./work -cf - . | tar -C "$T/repo" -xf - && mkdir -p "$T/repo/work"
if [[ "${1:-}" == "--fresh-cache" ]]; then export HF_HOME="$T/hf"; fi
"$PY" -m venv "$T/venv" && source "$T/venv/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu || { echo "FAIL torch install"; exit 1; }
cd "$T/repo"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
declare -a RESULTS=()
step() {  # step NAME EXPECTED_REGEX TIMEOUT COMMAND...
  local name=$1 want=$2 limit=$3; shift 3
  echo "== $name"; local out; out=$(timeout "$limit" "$@" 2>&1); local rc=$?
  echo "$out" | grep -v "it/s\]" | tail -6
  if [[ $rc -eq 0 && "$out" =~ $want ]]; then RESULTS+=("PASS  $name"); else RESULTS+=("FAIL  $name (exit $rc)"); fi
}
# setup.sh ends with check_setup.py, which exits non-zero here only because there is no GPU: accept exactly that
echo "== setup.sh --no-gemma"; out=$(timeout 2400 bash setup.sh --no-gemma 2>&1); echo "$out" | grep -E "FAIL|problem|Ready" | tail -4
if [[ "$out" =~ "1 problem(s), affecting exercise(s) 3, 4, 5" && "$out" =~ "FAIL CUDA GPU" ]] || [[ "$out" =~ "Ready:" ]]; then RESULTS+=("PASS  setup.sh --no-gemma (no GPU here, as expected)"); else RESULTS+=("FAIL  setup.sh --no-gemma"); fi
step "exercise 1" "raw probe 0\.95.*SAE probe 0\.92" 900 python exercises/ex1_beat_the_sae_probe.py
step "exercise 2" "0\.78.*0\.494.*0\.525.*0\.499" 300 python exercises/ex2_find_the_leak.py
mkdir -p "$T/cpu"
for f in ex4_trace_a_fact ex5_edit_a_fact; do
  sed -e 's/device="cuda"/device="cpu"/g' -e 's/\.cuda()/.cpu()/g' -e 's/\.to("cuda")/.to("cpu")/g' "exercises/$f.py" > "$T/cpu/$f.py"
done
grep -rl '"cuda"\|\.cuda()' third_party/rome --include=*.py | xargs sed -i -e 's/"cuda"/"cpu"/g' -e 's/\.cuda()/.cpu()/g'
step "exercise 4 (CPU copy)" "peak at (le|.le) layer" 1800 python "$T/cpu/ex4_trace_a_fact.py"
step "exercise 5 (CPU copy)" "Louvre is located in the city of \(' Paris'.*-> \(' Rome'" 1800 python "$T/cpu/ex5_edit_a_fact.py"
echo; printf '%s\n' "${RESULTS[@]}"
echo "Exercise 3 (Gemma-2-9B) is not covered here; run it on a pod."
[[ ! " ${RESULTS[*]} " =~ FAIL ]]
