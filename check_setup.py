"""Check that this pod is ready for the workshop. Run from the repo root:

    python check_setup.py              # everything, including a short GPU test
    python check_setup.py --no-gemma   # skip Gemma-2-9B (Exercise 3 and P3)
    python check_setup.py --quick      # skip the GPU test
"""
import importlib.metadata as md
import os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
GEMMA, QUICK = "--no-gemma" not in sys.argv, "--quick" in sys.argv
ROME_COMMIT = "0874014cd9837e4365f3e6f3c71400ef11509e04"
STATS = "third_party/rome/data/stats/gpt2-xl/wikipedia_stats/transformer.h.17.mlp.c_proj_float32_mom2_100000.npz"
results = []


def check(name, ok, fix="", needed_by="all"):
    results.append((name, bool(ok), fix, needed_by))
    print(("  ok   " if ok else "  FAIL ") + name + ("" if ok else f"\n         fix: {fix}"), flush=True)
    return bool(ok)


print("Python and GPU")
check(f"Python {sys.version.split()[0]}", sys.version_info >= (3, 10), "use Python 3.10 or newer")
try:
    import torch
    check(f"torch {torch.__version__}", True)
    cuda = torch.cuda.is_available()
    if cuda:
        gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        check(f"GPU {torch.cuda.get_device_name(0)}, {gb:.0f} GB", True)
        if GEMMA:
            check("GPU memory for Gemma-2-9B (about 20 GB)" + ("" if gb >= 40 else "; should fit, untested below 40 GB"), gb >= 23,
                  "Exercise 3 needs about 20 GB of GPU memory; use a 48 GB pod for it", "3")
    else:
        check("CUDA GPU", False, "start a GPU pod; exercises 1-2 still run on CPU", "3-5")
except ImportError:
    check("torch", False, "use the RunPod PyTorch template, or pip install torch")
    cuda = False

print("Packages")
for line in open("requirements.txt"):
    m = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*(==\s*([^\s#]+))?", line)
    if not m or line.lstrip().startswith("#"):
        continue
    pkg, want = m.group(1), m.group(3)
    try:
        have = md.version(pkg)
        check(f"{pkg} {have}" + (f" (pinned {want})" if want and have != want else ""), not want or have == want,
              f"pip install {pkg}=={want}")
    except md.PackageNotFoundError:
        check(pkg, False, "pip install -r requirements.txt")

print("Files")
npz = "data/ex1_92_glue_sst2_layer20.npz"
if check(npz, os.path.exists(npz), "git pull; the file is part of the repo", "1"):
    import numpy as np
    d = np.load(npz)
    check(f"  X {d['X'].shape}, {len(d['train_idx'])} train / {len(d['test_idx'])} test rows",
          d["X"].shape == (5000, 3584), "the file is damaged; download it again", "1")
head = subprocess.run(["git", "-C", "third_party/rome", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
check("third_party/rome at the pinned commit", head == ROME_COMMIT, "bash setup.sh", "5")
for f, need in [("third_party/rome/data/known_1000.json", "4"), ("third_party/rome/data/counterfact.json", "5"), (STATS, "5")]:
    check(f, os.path.exists(f) and os.path.getsize(f) > 0, "bash setup.sh (it downloads from rome.baulab.info)", need)

if not os.environ.get("HF_HOME") and os.path.isdir("/workspace/hf"):
    check("HF_HOME is set in this shell", False,
          "run: source ~/.bashrc (otherwise models download again into the container disk)")
print("Models (" + os.environ.get("HF_HOME", "default Hugging Face cache") + ")")
from huggingface_hub import get_token, try_to_load_from_cache
for repo, f, need in [("gpt2-xl", "model.safetensors", "4-5"),
                      ("google/gemma-scope-9b-pt-res", "layer_20/width_16k/average_l0_408/params.npz", "1"),
                      ("google/gemma-scope-9b-pt-res", "layer_20/width_131k/average_l0_114/params.npz", "3")]:
    p = try_to_load_from_cache(repo, f)
    check(f"{repo}/{f}", isinstance(p, str) and os.path.exists(p), "bash setup.sh", need)
if GEMMA:
    check("Hugging Face token (needed for Gemma)", bool(get_token()), "export HF_TOKEN=hf_... or run huggingface-cli login", "3")
    p = try_to_load_from_cache("google/gemma-2-9b", "model.safetensors.index.json")
    check("google/gemma-2-9b", isinstance(p, str), "request access on huggingface.co, then bash setup.sh", "3")

print("Smoke tests")
t = time.time()
r = subprocess.run([sys.executable, "exercises/ex2_find_the_leak.py"], capture_output=True, text=True, timeout=300)
check(f"Exercise 2 runs on CPU ({time.time() - t:.0f} s)", r.returncode == 0 and "the leak" in r.stdout,
      r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "see the error above", "2")
if cuda and not QUICK:
    t = time.time()
    code = ("import torch; from transformer_lens import HookedTransformer; torch.set_grad_enabled(False);"
            "m = HookedTransformer.from_pretrained('gpt2-xl', device='cuda');"
            "p = m('The Eiffel Tower is located in the city of', prepend_bos=False)[0, -1].softmax(-1);"
            "print(repr(m.to_string(p.argmax())), round(p.max().item(), 2))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    check(f"GPT-2 XL on the GPU says {out} ({time.time() - t:.0f} s)", r.returncode == 0 and "Paris" in out,
          r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "see the error above", "4-5")

bad = [r for r in results if not r[1]]
print()
if not bad:
    print("Ready: every exercise and proposal can run on this pod.")
else:
    ex = set()
    for r in bad:
        a, _, b = r[3].partition("-")
        ex |= set(range(1, 6)) if a == "all" else set(range(int(a), int(b or a) + 1))
    print(f"{len(bad)} problem(s), affecting exercise(s) {', '.join(map(str, sorted(ex)))}. Fix them and run this again.")
    sys.exit(1)
