# Read · Locate · Rewrite

The hands-on half of a two-hour interpretability workshop. On your own GPU pod you will:
- probe Gemma‑2‑9B and its sparse autoencoder features;
- trace where GPT‑2 XL recalls a fact;
- edit that fact with ROME.

Claude Code is your lab partner: it runs the exercises and implements the extensions you choose.

The workshop page is the guide. Open [`site/index.html`](site/index.html) in a browser, or the link your instructor sends, and work through its five chapters. This repo holds everything you run.

The two papers:
- **ROME.** Meng, Bau, Andonian & Belinkov, *Locating and Editing Factual Associations in GPT*, NeurIPS 2022.
- **SAE probing.** Kantamneni, Engels, Rajamanoharan, Tegmark & Nanda, *Are Sparse Autoencoders Useful? A Case Study in Sparse Probing*, ICML 2025.

## 1 · Before the session (about 30 minutes)

### Accounts

| Account | What for | Note |
|---|---|---|
| [RunPod](https://www.runpod.io) | your GPU pod | Add a few dollars of credit; the session uses about 3 GPU-hours. |
| [Hugging Face](https://huggingface.co/settings/tokens) | model downloads | Create a *read* token. For Exercise 3, request access to [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) **a day ahead**: requests are approved by hand. |
| [Claude](https://claude.ai) | Claude Code | A Pro, Max, Team or Enterprise plan, or an Anthropic API key. The free plan does not include Claude Code. |

### Start a pod

1. **Add your SSH key before you deploy.** RunPod copies keys into a pod only when it starts. If you have no key, run `ssh-keygen -t ed25519` on your laptop. Paste the contents of `~/.ssh/id_ed25519.pub` into RunPod under **Settings → SSH Public Keys**.
2. **Deploy a pod.**
   - **GPU:** one 48 GB card, such as an A40, RTX A6000 or L40S. A 24 GB card runs Exercises 1, 2, 4 and 5. Exercise 3 needs about 20 GB, so it should fit too, but that is untested.
   - **Template:** *RunPod PyTorch*, version 2.8 or newer.
   - **Volume disk:** 100 GB. It is mounted at `/workspace` and keeps the models while the pod is stopped.
3. **Connect from your laptop.** Use the command under **Connect → SSH over exposed TCP**:
   ```bash
   ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
   ```
   You can also add the pod as an SSH connection in the Claude desktop app and open a Code session there.

### Install

On the pod:
```bash
cd /workspace
git clone https://github.com/nelionel/read-locate-rewrite.git
cd read-locate-rewrite
export HF_HOME=/workspace/hf   # models live on the persistent volume
export HF_TOKEN=hf_...         # your Hugging Face read token
bash setup.sh                  # add --no-gemma until your Gemma access is approved
source ~/.bashrc               # later shells keep HF_HOME
```
`setup.sh` does the following, and is safe to rerun:
- installs the Python packages, pinned to the versions the page was run with;
- clones the authors' ROME code into `third_party/rome`;
- downloads GPT‑2 XL (6 GB), two Gemma Scope SAEs (4.3 GB) and Gemma‑2‑9B (37 GB) into `/workspace/hf`;
- downloads ROME's datasets and layer-17 key statistics into `third_party/rome/data`;
- stores your Hugging Face token under `/workspace/hf`, so later shells can load Gemma;
- ends with `python check_setup.py`.

### Start Claude Code

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude
```
`claude` prints a login link; open it on your laptop. Inside this folder Claude reads [`CLAUDE.md`](CLAUDE.md), so it already knows the workshop, the models and the rules. Run long sessions inside `tmux` so a dropped connection does not stop them.

## 2 · During the session

Read the page one chapter at a time. Each chapter lists its deliverables in the page's left rail. When you reach an exercise, run it yourself or ask Claude:

```bash
python exercises/ex4_trace_a_fact.py
```
```
/exercise 4
```

Claude asks for your prediction first, runs the exercise, and compares the output with your prediction.

Each exercise ends with a **proposal**: an extension worth an experiment. Ask Claude to implement one:
```
/proposal P4
```
Claude writes the code in `work/P4/`, runs it within the proposal's time budget, and reports the result next to your prediction.

| Exercise | Chapter | Runs on | Its proposal |
|---|---|---|---|
| 1 · Beat the SAE probe | 2 Read | CPU | **P1** How many SAE features close the gap? |
| 2 · Find the leak | 2 Read | CPU | **P2** How much validation data stops you picking noise? |
| 3 · Test a feature's label | 2 Read | GPU, Gemma‑2‑9B | **P3** Write a label that survives testing |
| 4 · Trace a fact | 3 Locate | GPU | **P4** Corrupt by swapping the subject |
| 5 · Edit a fact and probe its edges | 4 Rewrite | GPU | **P5** Does a fact's trace predict where editing it works? · **P6** Why did one edit break a run of 1,000? |

[`proposals/README.md`](proposals/README.md) describes all six.

## What is where

```
site/index.html      the workshop page
AGENTS.md            how any coding agent should work in this repo (CLAUDE.md imports it)
manifest.json        machine-readable index of the exercises and proposals
exercises/           the five exercises, the same code as on the page
proposals/           P1–P6: the extensions, with time budgets and reference results
work/                your runs; Claude writes here
data/                Exercise 1's activations and the edit order used in P6
setup.sh             one-time setup; check_setup.py checks it
CLAUDE.md            what Claude Code reads in this folder
third_party/rome     the authors' ROME code (created by setup.sh)
tools/               cpu_smoke_test.sh (tests the repo without a GPU); how the data file was made
```

## Troubleshooting

- **A `401` or `GatedRepoError` for google/gemma-2-9b** means your access request is still pending or `HF_TOKEN` is not set. Run `bash setup.sh --no-gemma` now; Exercise 3 can wait.
- **`CUDA out of memory`:** keep one model on the GPU at a time and restart Python between exercises. Gemma‑2‑9B needs about 20 GB and GPT‑2 XL about 7 GB.
- **ROME prints "Computing locally…":** its key statistics did not download. Rerun `bash setup.sh`, which retries.
- **Disk full:** the volume needs about 60 GB. Gemma‑2‑9B alone is 37 GB.

## Credits

The exercises call the authors' code and data:
- [kmeng01/rome](https://github.com/kmeng01/rome) (MIT), with CounterFact and its statistics from rome.baulab.info;
- [Gemma Scope](https://huggingface.co/google/gemma-scope-9b-pt-res) SAEs (CC BY 4.0);
- [Gemma 2](https://ai.google.dev/gemma/terms) under its terms of use.

The Exercise 1 file holds Gemma‑2‑9B activations for the SST‑2 task from the authors' SAE-probing data release.
