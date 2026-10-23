# P3 · Write a label that survives testing

Chapter 2 (Read) · starts from `exercises/ex3_test_a_feature_label.py` · GPU with Gemma‑2‑9B · budget 15 minutes

## Question
Feature 122774 of a 131k-feature Gemma Scope SAE is labelled "living rooms". A label is a hypothesis about which inputs make a feature fire. Write the most precise one-sentence label you can, then try to break it.

## Predict
1. Your label, in one sentence.
2. Three inputs you expect to fire strongly (above 20), and three you expect to stay silent (0).

## Do
1. Load Gemma‑2‑9B once. Write `act(text)`, the feature's maximum activation over the tokens of `text`, skipping `<bos>`.
2. Run the participant's six inputs first and report them against the predictions.
3. Then run a systematic set, about 40 inputs:
   - synonyms: lounge, den, front room, drawing room, family room, sitting room, TV room;
   - other rooms: kitchen, bedroom, bathroom, dining room;
   - furniture: sofa, couch, armchair, coffee table;
   - the phrase in sentences: "We watched TV in the living room.", "living-room furniture";
   - the parts alone: "living", "room";
   - other languages, in sentences: French, Spanish and German.
4. Sort by activation. Ask the participant to revise the label, and test one input that the revised label predicts.

## Reference (show after the run)

From Exercise 3, in bfloat16:

| Input | Activation |
|---|---|
| living room | 52.4 |
| sitting room | 42.4 |
| Le salon est grand | 23.5 |
| family room | 18.4 |
| salon de coiffure | 12.0 |
| parlour | 10.9 |
| lounge, sofa, bedroom, room, Salon | 0.0 |

No one-word label fits all of these. A good label names the words and the contexts: some phrases for a home's main sitting room, and French *salon* followed by more French. In "salon de coiffure" the peak is at "de", before the phrase has said it means a hair salon. Of the page's 14 translations, the feature fires on all but French "Salon", which is silent on its own.

## Pitfalls
- Report the maximum over tokens, and skip `<bos>`, whose activations are huge.
- bfloat16 and float32 differ by a few units: "living room" scores 52.4 in bf16 and 55.3 in fp32.
