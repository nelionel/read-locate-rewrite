# The five exercises: predictions and answers

Claude uses this file for `/exercise N`. Show an answer only after the participant has predicted.

## Exercise 1 · Beat the SAE probe
`exercises/ex1_beat_the_sae_probe.py` · CPU · about 1 minute

**Predict:** on SST‑2 sentiment (task 92), which wins on test AUC, logistic regression on all 3,584 dimensions or an L1 probe on the 16 SAE features whose class means differ most? By how much? Where would a text-only probe land?

**Answer:** the code prints about 0.952 for the raw probe and 0.920 for the SAE probe. With the paper's hyperparameter search they reach 0.958 and about 0.92. A text-only TF‑IDF probe reaches 0.783. On this task the model clearly helps, but keeping only 16 SAE features costs about 0.03–0.04 AUC.

## Exercise 2 · Find the leak
`exercises/ex2_find_the_leak.py` · CPU · seconds

**Predict:** make 16,384 pure-noise features and keep the 16 with the largest class-mean gap. Fit on 924 rows, then score on the last 100 (validation) and on 2,000 fresh rows (test). Predict the validation and test AUC twice: ranking on all 1,024 rows (the leak), and ranking on the 924 fitting rows only.

**Answer:**
- With the leak: 0.780 on validation and 0.494 on test.
- Without the leak: 0.525 and 0.499.
- Over 10 seeds, validation gives 0.737 ± 0.036 with the leak against 0.520 ± 0.029 without, and test gives 0.50 both times.

The leak manufactures a validation score of about 0.74 from nothing.

## Exercise 3 · Test a feature's label
`exercises/ex3_test_a_feature_label.py` · GPU, Gemma‑2‑9B in bfloat16 · about 3 minutes, most of it loading

**Predict:** feature 122774 is labelled "living rooms". Which of these make it fire: "sitting room", "lounge", "sofa", "Le salon est grand", "salon de coiffure" (a hair salon)?

**Answer (bfloat16):**

| Input | Activation |
|---|---|
| living room | 52.4 |
| sitting room | 42.4 |
| Le salon est grand | 23.5 |
| family room | 18.4 |
| salon de coiffure | 12.0 |
| parlour | 10.9 |
| lounge, sofa, bedroom, room, Salon | 0.0 |

The feature tracks a home's sitting room, but not every word for it ("lounge" is silent). It also fires on French *salon* once more French follows. In "salon de coiffure" the peak is at "de", before the phrase has said it means a hair salon. The page's figure shows 55.3 for "Living room": the capital L, not the dtype, makes the difference.

## Exercise 4 · Trace a fact
`exercises/ex4_trace_a_fact.py` · GPU · about 1 minute · saves `ex4_mlp_trace.png`

**Predict:** for "The Space Needle is in downtown" → "Seattle", which token shows the strongest MLP effect ("Space", "Need", "le" or the last token), and around which layers?

**Answer:**
- **MLP:** strongest at "le", the last subject token, on a broad plateau from layer 11 to 20. Anywhere on it, restoring the MLP window brings P(Seattle) back to about 0.72–0.90, against 0.98 clean and 0.022 corrupted. The exact peak layer depends on the noise draw.
- **Attention:** peaks at the last token ("downtown"), at layer 32.

Both match the paper's Figure 1.

## Exercise 5 · Edit a fact and probe its edges
`exercises/ex5_edit_a_fact.py` · GPU · about 1 minute

**Predict:** after teaching GPT‑2 XL that the Eiffel Tower is in Rome, what is the next word for each prompt?
- (a) "The Eiffel Tower is in the city of"
- (b) "The Louvre is located in the city of"
- (c) "The Eiffel Tower is located in Rome. The Louvre is located in the city of"

**Answer (before → after):**
- (a) "Paris" → "Rome": the edit generalises.
- (b) "Paris" → "Paris", with P(Paris) moving only from 0.82 to 0.76: the edit is specific.
- (c) "Paris" → "Rome", with P(Rome) going from 0.0000 to 1.000.

In (c), the unedited model ignores the false sentence, and the edited model applies it to the Louvre. This is the leak that the standard neighbourhood score never sees.
