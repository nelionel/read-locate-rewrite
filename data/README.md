# Data

## `ex1_92_glue_sst2_layer20.npz` (33 MB)
Exercise 1 and proposals P1–P2. SST‑2 sentiment, task 92 of the SAE-probing paper.

| Key | Contents |
|---|---|
| `X` | float16 `[5000, 3584]`: Gemma‑2‑9B's residual stream after layer 20 (`blocks.20.hook_resid_post`), at the last token of each prompt |
| `y` | `[5000]` labels, 0 or 1, balanced |
| `train_idx` | the authors' 1,024 training rows (seed 42); the last 100 are the validation rows |
| `test_idx` | the authors' 3,975 test rows |
| `prompts` | the 5,000 prompts |

The activations were extracted with TransformerLens from fp32 weights, truncating prompts on the left at 1,024 tokens. The authors released activations for 1,000 CoLA prompts only; on those, this extraction matches theirs to a median relative error of 0.04%. `tools/build_ex1_file.py` shows how the file was made.

## `counterfact_order.json`
The order in which the page applied CounterFact edits: a seed-0 shuffle of the dataset, first 1,100 case IDs. The page's 1,000-edit run used the first 1,000. The next 100 were the never-edited facts it tracked. Proposals P5 and P6 use this order.
