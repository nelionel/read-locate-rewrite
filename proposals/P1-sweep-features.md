# P1 · How many SAE features close the gap?

Chapter 2 (Read) · starts from `exercises/ex1_beat_the_sae_probe.py` · CPU · budget 5 minutes

## Question
On SST‑2 (task 92), the top‑16 SAE probe trails logistic regression on the raw activation by about 0.03 AUC. Does keeping more SAE features close the gap? Where does a probe with no training at all land: the difference of the two class means?

## Predict
1. At what k, from 1 to 512, does the SAE probe match the raw probe's 0.952, if ever?
2. Is the curve monotone in k?
3. Where does a difference-of-means probe land (score = x · (mean of class 1 − mean of class 0))?

## Do
1. Encode the activations with the 16k SAE, as in Exercise 1. Rank the features once by the absolute class-mean gap on the training rows.
2. For k = 1, 2, 4, …, 512, fit the same L1 logistic regression (`liblinear`, C = 1) on the top k features. Record the test AUC.
3. Add two reference lines: the raw probe, and the difference-of-means probe on the raw activations.
4. Plot test AUC against k on a log axis.
5. If time remains: for each k, choose C from {0.01, 0.1, 1, 10} by 5-fold cross-validation inside the training rows. Does tuning change the answer?

## Reference (show after the run)

The untuned SAE probe never catches up:

| k | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| Test AUC | 0.600 | 0.734 | 0.826 | 0.898 | 0.920 | **0.926** | 0.924 | 0.923 | 0.907 | 0.904 |

- The raw probe reaches 0.952.
- All 16,384 features with L2 logistic regression reach 0.929.
- The difference of means reaches 0.903, with no fitting at all.

Beyond 32 features the fixed L1 strength lets the probe overfit. The paper tunes it, which is one reason its curves keep rising with k on average. On task 92, the gap to the raw activation stays even when every feature is kept.

## Pitfalls
- Rank features on the training rows only, and use the test rows only for the final score.
- SAE features are non-negative and differ widely in scale, so expect the L1 strength to matter.
