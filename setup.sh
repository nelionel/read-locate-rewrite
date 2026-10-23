#!/usr/bin/env bash
# One-time setup for a workshop pod. Safe to rerun: finished steps are skipped.
#   export HF_TOKEN=hf_...   # your Hugging Face token
#   bash setup.sh            # or: bash setup.sh --no-gemma
set -euo pipefail
cd "$(dirname "$0")"
GEMMA=1
if [[ "${1:-}" == "--no-gemma" ]]; then GEMMA=0; fi

# Models go to the persistent volume, which survives a pod stop
if [[ -z "${HF_HOME:-}" && -d /workspace ]]; then export HF_HOME=/workspace/hf; fi
if [[ -n "${HF_HOME:-}" ]] && ! grep -qs "export HF_HOME=" ~/.bashrc; then echo "export HF_HOME=$HF_HOME" >> ~/.bashrc; fi

echo "== 1/4 Python packages"
python -m pip install -q -r requirements.txt

echo "== 2/4 The authors' ROME code"
if [[ ! -d third_party/rome/.git ]]; then git clone -q https://github.com/kmeng01/rome third_party/rome; fi
git -C third_party/rome checkout -q 0874014cd9837e4365f3e6f3c71400ef11509e04

echo "== 3/4 ROME's datasets and layer-17 key statistics (rome.baulab.info)"
fetch() {
  if [[ ! -s "$2" ]]; then
    mkdir -p "$(dirname "$2")"
    curl -fsSL --retry 5 --retry-delay 3 -o "$2.part" "$1" && mv "$2.part" "$2"
  fi
}
R=https://rome.baulab.info/data
S=gpt2-xl/wikipedia_stats/transformer.h.17.mlp.c_proj_float32_mom2_100000.npz
fetch "$R/dsets/known_1000.json" third_party/rome/data/known_1000.json
fetch "$R/dsets/counterfact.json" third_party/rome/data/counterfact.json
fetch "$R/stats/$S" "third_party/rome/data/stats/$S"

# keep the token for later shells (stored under $HF_HOME by Hugging Face, not in this repo)
if [[ -n "${HF_TOKEN:-}" ]]; then python -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)" >/dev/null; fi

echo "== 4/4 Models, to ${HF_HOME:-the default Hugging Face cache}"
GEMMA=$GEMMA python - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("gpt2-xl", allow_patterns=["*.json", "*.txt", "model.safetensors"])
for f in ("layer_20/width_16k/average_l0_408/params.npz", "layer_20/width_131k/average_l0_114/params.npz"):
    hf_hub_download("google/gemma-scope-9b-pt-res", f)
if os.environ["GEMMA"] == "1":
    snapshot_download("google/gemma-2-9b", allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
print("models ready")
PY

echo "== Done. Run 'source ~/.bashrc' (or open a new shell) so that HF_HOME=${HF_HOME:-} holds in every shell."
if [[ $GEMMA == 1 ]]; then python check_setup.py; else python check_setup.py --no-gemma; fi
