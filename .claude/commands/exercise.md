---
description: Run a workshop exercise (1–5), with your prediction first
argument-hint: <exercise number, 1-5>
---
Run workshop exercise number $ARGUMENTS. Follow the loop in CLAUDE.md.

1. Find the exercise's file (exercises/ex<number>_*.py) and its entry in proposals/exercises.md.
2. Ask me the exercise's Predict question, from proposals/exercises.md, and wait for my answer. Save it to work/E<number>/prediction.md. Do not show the reference answer yet.
3. Run the file unchanged from the repo root with `timeout 900 python exercises/<file>`, and save its output to work/E<number>/output.txt. Exercises 3–5 need the GPU.
4. Show the output. Then compare it with my prediction and with the reference answer in proposals/exercises.md, in three lines or fewer.
5. Name the exercise's proposal, and ask whether I want to run it.
