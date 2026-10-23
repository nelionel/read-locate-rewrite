---
description: Implement a workshop proposal (P1–P6) as a small experiment
argument-hint: <proposal ID, P1-P6>
---
Implement workshop proposal $ARGUMENTS. Follow the loop in CLAUDE.md.

1. Read the proposal's file in proposals/ (its name starts with the ID) and the exercise code it starts from.
2. Ask me the questions under "Predict", and wait for my answers. Save them to work/<ID>/prediction.md.
3. Give your plan in three bullets, with the expected run time. The proposal's budget is pre-approved; ask before going beyond it.
4. Write the code in work/<ID>/ and run it with `timeout`, printing progress.
5. Report:
   - my prediction;
   - the result, with its n and spread;
   - the reference from the proposal file;
   - one sentence on what it means.

   Save work/<ID>/summary.md and a figure.
6. Suggest one follow-up question the result raises. Do not run it unless I ask.
