# P4 · Corrupt by swapping the subject

Chapter 3 (Locate) · starts from `exercises/ex4_trace_a_fact.py` · GPU · budget 15 minutes

## Question
Gaussian noise on the subject's embeddings is an unnatural input: no real prompt looks like that. The alternative swaps in a real subject with the same number of tokens and no noise (symmetric token replacement; Zhang & Nanda, 2024). Does tracing then locate the fact in the same place?

## Predict
1. With a swapped subject, will the Space Needle's own MLP map look like its noise map? Or will the peak move earlier, or disappear?
2. The page's table reports the 200-fact answer. Will 20 facts be enough to see it?

## Do
1. **One fact.** Take "The Space Needle is in downtown" → "Seattle". Pick three replacement subjects of exactly three tokens that have nothing to do with Seattle; check the count with `model.to_tokens(..., prepend_bos=False)`.
   - The corrupted run is the prompt with the swapped subject. Check that P(Seattle) is low there.
   - Restore clean MLP windows at each token and layer, as in Exercise 4.
   - Average the map over the three swaps, and compare it with the Gaussian map from Exercise 4.
2. **Twenty facts.** Use the first 20 facts in `third_party/rome/data/known_1000.json` that GPT‑2 XL gets right.
   - Restrict the trace to the MLP window at the last subject token, layers 0–47; this is fast.
   - Run it twice: with 3σ noise, and with one same-length swapped subject per fact, drawn from other facts' subjects.
   - Plot the two average curves of indirect effect against layer.

## Reference (show after the run)

The page's run, on the first 200 known facts:

| Corruption | P(answer) when corrupted | Total effect (points) | MLP peak at the last subject token (points) | Peak layer (window centre) |
|---|---|---|---|---|
| Gaussian, 3σ | 3.3% | 24.8 | 6.7 | 16 |
| Subject swapped | 1.7% | 26.3 | 3.5 | 5 (95% interval 5–17) |

The swap destroys the fact as thoroughly as the noise does, yet the MLP effect at the last subject token halves. Its best estimate is window centre 5, but the 95% interval (5–17) includes the noise's 16, so the location is uncertain; the hidden-state and attention sites do not move. How you corrupt changes how much the MLPs appear to matter. Single-fact maps are noisy, so expect the 20-fact average to be clearer than the one-fact map.

## Pitfalls
- The replacement must have exactly the same number of tokens, or the positions no longer line up.
- If the swapped prompt still predicts the answer, the corruption has not removed the fact; choose another replacement.
- Compare effects only between runs with the same corruption.
