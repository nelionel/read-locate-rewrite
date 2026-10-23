"""Download the replication results from Hugging Face (dataset Ionel2023/read-locate-rewrite-replication).

    python tools/get_replication_data.py results    # ledgers, results, figures, labels: about 2.7 GB, into /workspace/replication
    python tools/get_replication_data.py all        # everything, about 10 GB, into /workspace/replication
    python tools/get_replication_data.py notebook   # Gemma-2-9B layer-20 activations, laid out for the notebook's Checkpoint B
"""
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = "Ionel2023/read-locate-rewrite-replication"
part = sys.argv[1] if len(sys.argv) > 1 else "results"
if part in ("results", "all"):
    dest = Path(os.environ.get("REPLICATION_DIR", "/workspace/replication"))
    keep = None if part == "all" else ["*.md", "sae/results/*", "sae/labels/*", "rome/results/*", "rome/figures/*"]
    snapshot_download(REPO, repo_type="dataset", local_dir=dest, allow_patterns=keep)
    print("Replication data in", dest, "- start with CLAIMS_TABLE.md")
elif part == "notebook":
    root = Path(os.environ.get("SAE_ARTIFACTS", "/workspace/sae_artifacts"))
    snapshot_download(REPO, repo_type="dataset", local_dir=root / "_hf", allow_patterns=["sae/activations/gemma-2-9b/*"])
    acts = root / "data" / "model_activations_gemma-2-9b"; acts.mkdir(parents=True, exist_ok=True)
    for f in (root / "_hf" / "sae" / "activations" / "gemma-2-9b").glob("*.pt"):
        if not (acts / f.name).exists():
            (acts / f.name).symlink_to(f)
    print(len(list(acts.glob("*.pt"))), "activation files in", acts)
    if not (root / "data" / "cleaned_data").exists():
        print("Checkpoint B also needs the authors' task files (cleaned_data/, a 575 MB zip), which this dataset does not mirror.")
        print("Download cleaned_data.zip once per team from the authors' data release (linked in github.com/JoshEngels/SAE-Probes),")
        print("then unzip it so that", root / "data" / "cleaned_data" / "92_glue_sst2.csv", "exists. Checkpoint A and Part 2 need none of this.")
else:
    sys.exit(__doc__)
