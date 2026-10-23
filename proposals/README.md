# Proposals

Each exercise on the page ends with a proposal: an extension that asks a new question. Pick one, write down your prediction, and ask Claude to implement it with `/proposal P1`.

| ID | Chapter | Question | Starts from | Runs on | Budget |
|---|---|---|---|---|---|
| [P1](P1-sweep-features.md) | 2 Read | How many SAE features close the gap to the raw probe? | Exercise 1 | CPU | 5 min |
| [P2](P2-validation-size.md) | 2 Read | How much validation data stops you picking noise? | Exercise 2 | CPU | 10 min |
| [P3](P3-better-label.md) | 2 Read | Can you write a feature label that survives testing? | Exercise 3 | GPU (Gemma) | 15 min |
| [P4](P4-swap-the-subject.md) | 3 Locate | Does swapping the subject locate a fact where noise does? | Exercise 4 | GPU | 15 min |
| [P5](P5-trace-vs-edit-layer.md) | 4 Rewrite | Does a fact's trace predict where editing it works? | Exercises 4–5 | GPU | 25 min |
| [P6](P6-the-edit-that-broke.md) | 4 Rewrite | Why did one edit break a run of 1,000? | Exercise 5 | GPU | 10 min |

Every proposal file has the same parts:
- **Question**
- **Predict:** what to write down before running
- **Do:** the design
- **Reference:** what the page's run found. Claude shows it only after your run. Some references also appear on the page; if you have already read one, predict how your smaller run will differ from it.
- **Pitfalls**

`exercises.md` holds the predictions and answers for the five exercises themselves.
