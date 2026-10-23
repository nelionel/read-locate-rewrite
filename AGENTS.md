# AGENTS.md

This repo is the hands-on half of **Read · Locate · Rewrite**, a two-hour interpretability workshop on two papers:
- **ROME:** Meng et al. 2022, causal tracing and rank-one editing in GPT‑2 XL.
- **SAE probing:** Kantamneni et al. 2025, sparse-autoencoder probes against plain baselines on Gemma‑2‑9B.

Participants read the workshop page (`site/index.html`, chapters 1–5) and run the exercises and proposals on their own RunPod GPU pod, with a coding agent as their lab partner.

`manifest.json` lists every exercise and proposal: file, device, run time, inputs, where the answer is, and how it was verified. Read it first.

## How the workshop runs

Every team does two things, in this order:
1. **The notebook.** Teams work through `notebook/Mech_Interp_Probing_SAEs_Causal_Tracing.ipynb`: Part 1 on SAE probing, then Part 2 on causal tracing.
   - Help the way `/notebook` does: say what each section tests, and ask for a prediction before cells run.
   - For an "Exercise" cell, let the participant write the code first. Keep `USE_SOLUTIONS = False`, and never paste a reference solution unless they ask.
   - Checkpoint A runs as soon as setup has finished. Checkpoint B needs `python tools/get_replication_data.py notebook` plus the authors' task files (see README.md, "Replication results on Hugging Face"). If `RAW_ARTIFACTS_READY` is False, say so and move on to Part 2.
2. **One or two exercises, of the team's choice**, from `exercises/`. Each has a proposal P1–P6 to extend it. Run both with the lab loop below.

**Replication results:** the public Hugging Face dataset `Ionel2023/read-locate-rewrite-replication` holds every claim's test and result, the result files, the code, and Gemma‑2‑9B layer-20 activations, encodings and labels for all 113 tasks.
- Fetch it with `python tools/get_replication_data.py results|all|notebook`.
- Use it to answer "what did the replication find for claim X?" (start from `CLAIMS_TABLE.md`), or to rerun probing experiments on any task without the GPU.

## Which job are you doing?

- **Lab partner:** a participant asks you to run an exercise or implement a proposal. Follow *The lab loop* below.
- **Maintainer:** you are asked to change the repo itself. Read *Maintaining the repo* first; several files are generated and must not be edited by hand.

## Repo map

| Path | What it is | Edit by hand? |
|---|---|---|
| `README.md` | Participant onboarding: RunPod → clone → `setup.sh` → Claude Code, then the session's two steps | yes |
| `notebook/…ipynb` | The workshop notebook: every team works through it first. It uses `/content` on Colab and the notebook's folder on a pod | yes |
| `AGENTS.md` | This file: instructions for any coding agent | yes |
| `CLAUDE.md` | Imports this file, plus Claude Code specifics | yes |
| `manifest.json` | Machine-readable index of exercises and proposals | yes; keep in sync |
| `exercises/ex1…ex5_*.py` | The five exercises, byte-identical to the code on the page | **no**: generated |
| `proposals/P1…P6-*.md` | One extension per file. Each has the same sections: Question, Predict, Do, Reference, Pitfalls | yes |
| `proposals/exercises.md` | Each exercise's Predict question and reference answer | yes |
| `site/index.html` | The workshop page, standalone | **no**: generated |
| `data/` | Exercise 1 activations (33 MB) and the CounterFact edit order (see `data/README.md`) | no |
| `setup.sh`, `check_setup.py`, `requirements.txt` | Pod setup, its check, and pinned packages | yes |
| `tools/build_ex1_file.py` | How the Exercise 1 file was made; needs the authors' SAE-Probes repo | **no**: generated |
| `tools/cpu_smoke_test.sh` | Fresh-environment test on any CPU machine | yes |
| `tools/get_replication_data.py` | Downloads the Hugging Face replication dataset (`results`, `all`, or `notebook`) | yes |
| `.claude/commands/` | The `/notebook`, `/exercise` and `/proposal` commands for Claude Code | yes |
| `work/` | Participants' runs, one folder per ID; git-ignored | runtime only |
| `third_party/rome/` | The authors' ROME code at commit 0874014, cloned by `setup.sh`; git-ignored | no |

## The lab loop

Every exercise and proposal run follows the same loop. IDs are `E1`–`E5` for exercises and `P1`–`P6` for proposals, and each run gets a folder `work/<ID>/`.

1. **Prediction first.** Ask for a one-line prediction unless one was given, and wait for it. Save it to `work/<ID>/prediction.md`. Never show a reference result before the participant predicts.
2. **Plan in three bullets:** what changes from the starting code, what you measure, and how long it takes.
3. **Code in `work/<ID>/`.** Copy from `exercises/`; never edit `exercises/`.
4. **Run with `timeout`,** printing progress. A proposal's stated budget is pre-approved: stay inside it by shrinking n, not by running long, and ask before going beyond it. Ask before anything else that takes over 10 minutes or downloads over 1 GB, such as `setup.sh`.
5. **Report:**
   - the prediction;
   - the result, with its n and spread;
   - the reference;
   - one sentence on what it means.

   Save `work/<ID>/summary.md` and a PNG figure. If the result disagrees with the reference, say so plainly, and look for the cause before explaining it.

**Exercise N:** run `timeout 900 python exercises/exN_*.py` from the repo root, unchanged, and save its output to `work/EN/output.txt`. The Predict question and the answer are in `proposals/exercises.md`. Read answers there, not in `site/index.html`: the page fills in its numbers with JavaScript, so the raw HTML shows "Answer pending.".
**Proposal PN:** follow `proposals/PN-*.md`.

## The pod

- Run from the repo root with the system `python`. There is one GPU; models are in `$HF_HOME` (`/workspace/hf`).
- Keep one model on the GPU at a time: `del model; torch.cuda.empty_cache()`.
- **GPT‑2 XL:** TransformerLens `"gpt2-xl"` for tracing and Hugging Face `"gpt2-xl"` for ROME. fp32, about 7 GB, 48 layers, d_model 1,600, MLP width 6,400.
- **Gemma‑2‑9B:** `"google/gemma-2-9b"`, loaded with TransformerLens and `dtype="bfloat16"`, about 20 GB. It is gated and needs `HF_TOKEN`. Read `blocks.20.hook_resid_post` and skip `<bos>`.
- **SAEs (Gemma Scope, `google/gemma-scope-9b-pt-res`):**
  - `layer_20/width_16k/average_l0_408` for Exercise 1;
  - `layer_20/width_131k/average_l0_114` for Exercise 3, feature 122774.

  They are JumpReLU: `pre = x @ W_enc + b_enc`, `f = where(pre > threshold, pre, 0)`.
- **ROME:** `third_party/rome`.
  - It reads `globals.yml` and `hparams/` relative to its folder, so first do `os.chdir("third_party/rome")`, as `exercises/ex5_edit_a_fact.py` does.
  - It edits layer 17, counting from 0.
  - Set `tok.pad_token = tok.eos_token`. ROME pads batches, and without a pad token it crashes.
  - Seed with `torch.manual_seed(0)`. ROME samples its random prefixes.
  - `apply_rome_to_model` edits in place. Read anything you need before the edit, pass `return_orig_weights=True`, and restore between independent edits:
    ```python
    from util import nethook   # ROME's helper, importable after the chdir
    model, orig = apply_rome_to_model(model, tok, request, hparams, return_orig_weights=True)
    ...  # measure the edited model
    with torch.no_grad():
        for name, w in orig.items():
            nethook.get_parameter(model, name)[...] = w
    ```
  - After `os.chdir("third_party/rome")`, paths into this repo need `../../`, for example `../../data/counterfact_order.json`.
- **Data:**
  - `third_party/rome/data/known_1000.json`: 1,209 facts for tracing, each with `subject`, `prompt` (for example "Vinson Massif is located in the continent of"), `attribute` (the answer) and `known_id`.
  - `third_party/rome/data/counterfact.json`: 21,919 edit requests. Each has:
    - `case_id`;
    - `requested_rewrite`, with `prompt` (containing `{}` for the subject), `subject`, `target_new.str` and `target_true.str`;
    - `paraphrase_prompts` (2);
    - `neighborhood_prompts` (10).
  - `data/counterfact_order.json`: the page's edit order.

## Conventions the page uses

- **Tracing:**
  - No BOS token.
  - The noise standard deviation is 0.1346. That is 3σ, where σ is the standard deviation of GPT‑2 XL's embeddings of subject tokens.
  - Average over 10 noise samples.
  - Restore MLP and attention outputs in 10-layer windows, l−5 … l+4.
  - Indirect effect = P(answer | corrupted, cell restored) − P(answer | corrupted).
- **Probing:**
  - Score with AUC on the authors' split (`train_idx`, `test_idx`). Validation is the last 100 training rows.
  - An SAE probe keeps the top-k features by class-mean gap, then fits L1 logistic regression (`liblinear`).
- **CounterFact:** per prompt, compare the mean NLL of the new object with that of the true object.
  - Efficacy and paraphrase count prompts where the new object wins; neighbourhood counts prompts where the true object wins.
  - S is the harmonic mean of the three.
  - The rewrite score is (P_new after − P_new before) / (1 − P_new before).

## Maintaining the repo

- **Generated files.** These are written by `build_repo.py` and `build_site.py`, which live in the instructor's workspace (`/opt/interp-workshop/site/` on the build machine), not in this repo:
  - `exercises/*.py` and `tools/build_ex1_file.py`, copied from the code blocks of the page's source (`template_v3.html`);
  - `site/index.html`, built from the same source and the replication results.

  To change an exercise, change the page source and regenerate. Editing the copy here makes the page and the repo disagree. Without the build workspace, change `exercises/` and say in your pull request that the page needs regenerating.

  Then update by hand:
  - the exercise's entry in `proposals/exercises.md`;
  - its `verified` field in `manifest.json`;
  - its expected-output pattern in `tools/cpu_smoke_test.sh`;
  - the status below.
- **Keep three places in step** when you add or rename an exercise or proposal: `manifest.json`, `proposals/README.md` and the page.
- **Verify after any change:**
  - `bash tools/cpu_smoke_test.sh`, on any machine, about 10 minutes, no GPU;
  - `python check_setup.py`, on a pod.
- **Rules:** every run finishes within an hour, under a hard `timeout`. Never write a token into a file in this repo. Tokens come from `HF_TOKEN`, or from Hugging Face's own login cache under `$HF_HOME`, which `setup.sh` fills in.

## Status (2026-10-23)

- **Tested:** a fresh Python 3.11 environment with torch 2.8 on CPU, standing in for the RunPod image.
  - `setup.sh --no-gemma` ran end to end.
  - Exercises 1 and 2 print exactly the page's numbers.
  - The CPU copies of Exercises 3–5, with `cuda` changed to `cpu`, run and agree with the page:
    - Exercise 3 within 0.2 activation;
    - Exercise 4 with the same peak token, layer 14 inside the page's plateau of 11–20, since the noise draw differs between CPU and GPU;
    - Exercise 5 to two decimals.
  - Exercise 5 first crashed on a missing pad token; the page and the file are fixed.
- **References:** P1, P2 and P6 were computed on CPU for this repo. P3–P5 come from the page's replication runs.
- **On a GPU (2026-10-23, an H200 RunPod pod, fresh clone, torch 2.8 cu128):** `setup.sh` with Gemma finished, and `check_setup.py` reported "Ready", including its GPT‑2 XL GPU test (peak 5.7 GB). The exercise runs were stopped before Exercise 1, because another job claimed the GPU.
- **Not yet tested:**
  - Exercises 1–5 run unchanged on a GPU (they were checked on CPU);
  - the Gemma download with a participant's own token;
  - the Claude Code login on a pod;
  - a full `/proposal` run by an agent.
- **Notebook:** checked only before its path fix. The original ran end to end on CPU on 2026-10-23, and Checkpoint A also ran in a separate dry run. The version with pod paths has not been re-run.
- **Open:** there is no LICENSE file yet.
