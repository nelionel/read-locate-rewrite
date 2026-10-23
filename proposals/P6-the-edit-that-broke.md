# P6 · Why did one edit break a run of 1,000?

Chapter 4 (Rewrite) · starts from `exercises/ex5_edit_a_fact.py` · GPU · budget 10 minutes (the long version takes 45)

## Question
The page's run applied 1,000 CounterFact edits to GPT‑2 XL, one after another. At edit 491 ("Toronto is a twin city of" → Damascus) one update came out about 400 times the usual size. The share of earlier edits that still held then fell from 99% to 60%. Is that blow-up a property of this one fact, or of the 490 edits before it? And what makes an edit blow up?

## Predict
1. Applied alone to a fresh GPT‑2 XL, will this edit's update be normal-sized or enormous?
2. Which edit requests would you expect to blow up? Name a property of the prompt or the subject that you can check.

## Do
1. **Alone.** Load CounterFact case 3629 from `third_party/rome/data/counterfact.json`. Apply it alone to a fresh model with the released code, as in Exercise 5, and measure ‖ΔW‖, the norm of the change to `transformer.h.17.mlp.c_proj.weight`.
   - For comparison, edit the first 5 cases of `data/counterfact_order.json`, restoring the weights after each one.
2. **Find the cause.**
   - ROME builds u = C⁻¹k̄ from the *prefix-averaged* key k̄. It then makes the edit exact at the key k of the *plain* prompt, and so divides by kᵀu.
   - For each case, compute kᵀu, k̄ᵀu and the cosine between k and k̄. Use `rome.compute_u.compute_u` and `rome.compute_v.get_module_input_output_at_word`; no v* optimisation is needed.
   - **Optional, the consistent key.** `compute_v` returns the released update's right vector r, which is exact at k. GPT‑2 multiplies row vectors by W (`kW + b`), so kW + b + (kᵀu)·r = v*. So recover v* = out(k) + r·(kᵀu). Then make the update exact at k̄ instead: r̄ = (v* − (k̄W + b)) / (k̄ᵀu). Compare the norms of u rᵀ and u r̄ᵀ.
3. **Test your hypothesis** from Predict 2 on 50 or more cases chosen to have and to lack the property. Computing u and the keys takes about a second per case.
4. **Long version (optional, about 45 minutes):** apply the first 500 edits of `data/counterfact_order.json` in a row, without restoring the weights. Every 50 edits, record the share of edits so far that still hold, and the largest update norm.

## Reference (show after the run)

These are CPU runs of the authors' code (seed 0).

**Alone on a fresh model:**

| Case | Released update ‖ΔW‖ | Consistent-key update ‖ΔW‖ | kᵀu | cos(k, k̄) |
|---|---|---|---|---|
| 3629: Toronto is a twin city of → Damascus | **4,697** | 38 | **0.04** | **−0.16** |
| The first 5 ordinary cases | 7.6–9.1 | 7.7–9.1 | 9.5–13.0 | 0.82–0.98 |

The blow-up is a property of the fact. ROME divides by kᵀu, and here kᵀu is almost zero.

**Which edits blow up:** those whose subject is a single token at the very start of the prompt.

| Edits | n | Median kᵀu (range) | Median cos(k, k̄) |
|---|---|---|---|
| One-token subject at the start ("London is a twin city of…") | 30 | **0.05** (0.02–0.07) | −0.14 |
| Two-token subject at the start | 20 | 8.6 (6.9–11.3) | 0.82 |
| The first 300 edits of the page's order | 300 | 9.9 (1.6–18.5) | 0.93 |

GPT‑2 treats the first position of a sequence unlike any other. The key there is nearly orthogonal to the same token's key after a prefix (cosine about −0.15), so it misses the direction ROME optimised for. CounterFact has 80 such requests out of 21,919, and edit 491 was the first of them in the page's order.

The consistent key of r-ROME (Gupta et al., 2024) makes the edit exact at k̄ instead, which removes the division by a near-zero number. With that update the page's 1,000-edit run never blew up: the largest update was 34.9, and 96% of the edits still held at the end.

## Pitfalls
- Restore the weights between independent edits. `apply_rome_to_model` edits in place.
- ROME samples its prefixes, so seed it (`torch.manual_seed(0)`) to compare runs.
- Count positions without a BOS token, as ROME does: "Toronto" is position 0.
