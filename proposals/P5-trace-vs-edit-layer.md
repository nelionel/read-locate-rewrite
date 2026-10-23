# P5 · Does a fact's trace predict where editing it works?

Chapter 4 (Rewrite) · starts from `exercises/ex4_trace_a_fact.py` and `exercises/ex5_edit_a_fact.py` · GPU · budget 25 minutes

## Question
If tracing showed where a fact is stored, a fact's tracing effect at a layer should predict how well an edit at that layer works. Hase et al. (2023) found that it barely does, in GPT‑J. Does that hold for a handful of facts in GPT‑2 XL?

## Predict
1. Over 20 facts and 6 layers, how much of the variation in edit success does the tracing effect explain: most, some, or almost none? The page answers this for 300 facts; predict whether 20 are enough to see the same result.
2. How often is the layer with the strongest trace among the best layers to edit? With six layers, chance is 1 in 6, or a little more when layers tie.

## Do
1. Choose 20 CounterFact records whose true answer GPT‑2 XL already predicts. Use the order in `data/counterfact_order.json` and keep the first 20 the model gets right.
2. **Trace.** For each fact, measure the MLP window effect at the last subject token, at layers 5, 11, 17, 23, 29 and 35. Use 3σ noise and 10 samples, as in Exercise 4.
3. **Edit.** For each fact and each of the six layers:
   - edit with ROME at that layer only (`hparams.layers = [L]`);
   - measure the rewrite score on the rewrite prompt: (P_new after − P_new before) / (1 − P_new before);
   - restore the weights.

   Each new layer downloads its key statistics the first time, about 160 MB.
4. **Fit.** Regress edit success on the layer alone (one-hot), on the tracing effect alone, and on both. Report each R², and how often the strongest-trace layer is the best edit layer.

## Reference (show after the run)

**The page's run** (GPT‑2 XL, 300 facts, the same six layers):
- **Variance explained:** the layer edited explains 30.2% of the variance in edit success, and the tracing effect explains 1.3%.
- **Layer effects:** editing works almost everywhere. The rewrite score is 0.98 at layer 5 and 0.99 at layer 17, falling to 0.69 at layer 35.
- **Choosing the layer:** a randomly chosen layer is among a fact's best layers 23% of the time. The layer where its trace is strongest is among them 19% of the time, no better than chance.

**Hase et al.** (GPT‑J, 652 facts, every layer): the layer explains 94.7%, and the tracing effect alone 1.6%. Added to the layer, it raises the variance explained by 0.1 points. The layer explains more there because they edited at every layer, including late ones where editing works much less well.

With 20 facts your R² values will be noisy. The conclusion to check is that the trace adds almost nothing once the layer is known.

## Pitfalls
- Restore the original weights after every edit; edits accumulate otherwise.
- Use the same facts for tracing and editing, and the rewrite prompt for both.
