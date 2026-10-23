# P2 · How much validation data stops you picking noise?

Chapter 2 (Read) · starts from `exercises/ex2_find_the_leak.py` · CPU · budget 10 minutes

## Question
With 100 validation examples, the best of 16,384 pure-noise features shows a validation AUC of about 0.73. On new data the same feature scores 0.50.
- **Part A:** how many validation examples does it take before the best of K noise candidates stops fooling you, that is, comes within 0.02 of its true AUC?
- **Part B:** on task 92, does choosing between the raw probe and the top‑16 SAE probe by cross-validation, rather than by the last 100 rows, change the choice?

## Predict
1. For K = 10, 1,000 and 100,000 candidates: about how many validation examples are needed?
2. How does that number grow with K?
3. On task 92, does 5-fold cross-validation inside the 924 fitting rows pick the SAE probe?

## Do
**Part A.**
1. Simulate pure-noise features on balanced labels. For each K and each validation size n, record the best validation AUC among K features, as a median over at least 50 draws.
2. Large K can need too much memory. Then use the null distribution of one feature's AUC, approximately Normal(0.5, (n+1) / (12 n₁ n₀)), and draw the maximum of K normals.
3. Check that approximation first against a direct simulation at K = 16,384 and n = 100, which should give about 0.73.
4. Find the smallest n at which the median best-of-K is at most 0.52.

**Part B.**
1. On `data/ex1_92_glue_sst2_layer20.npz`, compare two choosers between raw logistic regression and the top‑16 SAE probe:
   - the last 100 training rows;
   - 5-fold stratified cross-validation inside the first 924 training rows.

   Rank features inside each fold's training part.
2. Report both choices and their AUCs.

## Reference (show after the run)

**Part A** (normal approximation, median over draws):

| K | Validation examples needed |
|---|---|
| 10 | about 2,000 |
| 1,000 | about 8,800 |
| 100,000 | about 15,700 |

At n = 100 the best of K scores 0.58, 0.68 and 0.75. The cost grows roughly with log K: every extra factor of 10 in candidates buys the winner about the same extra luck.

**Part B:** on task 92 the raw probe wins both ways.

| Chooser | Raw probe | Top‑16 SAE probe |
|---|---|---|
| Last 100 rows | 0.979 | 0.936 |
| 5-fold cross-validation | 0.955 ± 0.009 | 0.924 ± 0.003 |

Here the leak changes nothing: ranking on 924 or 1,024 rows gives the same validation AUC. The leak matters on tasks where the SAE probe and the baseline are close, which is where it decides the winner.

## Pitfalls
- AUC under the null is not normal at very small n; check the approximation before relying on it.
- Use balanced labels, or put the class sizes into the variance formula.
