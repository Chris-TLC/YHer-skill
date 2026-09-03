# YHer Two-Minute Demo Recording Script

Target length: 2:00
Screen: 1280×800 browser, recording `http://127.0.0.1:8700` on this machine
Positioning: founder engineering demo — not a real student case, no claims of score gains or long-term mastery

## Before recording

1. Run `curl -fsS http://127.0.0.1:8700/health` and confirm the branch is the intended demo branch, the target SHA is correct, and the open-node count matches expectations.
2. Use a dedicated ID such as `demo_recording_20260713`; do not reuse real names, phone numbers, or school information.
3. Choose "氧化还原反应" (redox) and 30 minutes.
4. Do one dry run and memorize the first question you will deliberately answer wrong. Editing may cut out repeated answer-waiting time, but may not fake pages or rewrite results.
5. Confirm the screen shows no terminal, no `.env`, no model names, no tokens, no costs, or other credentials.
6. Post-fix content review passes within the authoritative-projection scope. Narration must describe explanations as "DeepSeek-assisted, with the verified standard solution as the authoritative projection" — never "the AI can now freely generate deep explanations for every question."

If the recording uses material from `demo/synthetic_scenarios/`, the screen must show `SYNTHETIC_DEMO`. The default service on 8700 is the normal founder QA service; do not describe a session as server-marked synthetic merely because the user ID contains the word "synthetic".

## 0:00–0:10 — Positioning

Screen: the YHer home page; title, Shanghai high-school chemistry, knowledge node, and time tier all visible at once.

Narration:

> This is YHer's local pre-alpha demo. It runs one narrow loop: Shanghai high-school chemistry diagnosis, evidence-bound explanations, independent verification, and profile updates.

On-screen action: do not scroll; do not show marketing pages or the old README.

## 0:10–0:24 — Create the time box

Screen: enter the dedicated demo ID, choose grade, review mode, redox, 30 minutes, and click start.

Narration:

> The student first gives a time budget. Thirty minutes promises quick localization, not finding every cause in one go. Only nodes meeting the deterministic five-family gate are open at the moment.

On-screen action: let the phase title, remaining time, and first question appear in full.

## 0:24–0:44 — Server-side scoring and progression

Screen: answer the first question deliberately wrong, submit; pause on the feedback, then advance. The second question only needs to be shown, not completed in this segment.

Narration:

> Answers are never delivered in advance; scoring happens server-side. Every response updates four states — mastered, prerequisite gap, concept confusion, and insufficient evidence. The next question is chosen by information gain, descending to prerequisite knowledge when needed.

Small caption: `Questions, options, and sources are visible; answers, rubrics, and item/family IDs are not sent down`.

Editing: answer time on subsequent diagnostic questions may be cut; other students' or other sessions' reports may not be spliced in.

## 0:44–1:10 — Learning checkpoint

Screen: cut to the learning checkpoint in the same session. Pause on the diagnosis summary, then slowly scroll through data/conditions, the causal chain, and exam steps.

Narration:

> After diagnosis there is always an explicit learning checkpoint. Public chemistry steps come only from standard solutions that passed answer verification; DeepSeek handles limited selection and organization and cannot write freely generated chemistry facts into the student-facing side.

Continued narration:

> This boundary is a deliberate contraction after content audit. What passed this round is the full-starting-point, difficulty, and real-mistake scaffolding under standard-solution constraints — not free chemistry-fact generation.

On-screen action: do not show providers, model names, or internal usage.

## 1:10–1:28 — Video recommendation and watch evidence

Screen: show one recommended title, its reason, and its completion criterion; click "Watch" and return; click "Watched" or the corresponding watch-record button; confirm to continue.

Narration:

> Recommendations come only from signed tracks with catalog evidence. Of 43 entities, 30 are enabled and 13 stay neutral; only 8 segments of organic-chemistry resources have real time anchors — no timestamps are fabricated for other links.

On-screen action: external videos need only a brief title or landing page; do not play long copyrighted excerpts.

## 1:28–1:45 — Independent held-out

Screen: enter the independent verification question and submit one response; if a second question is needed, use fast cuts to keep the question transition and the final result.

Narration:

> The final questions were frozen when the session began, disjoint from diagnosis and practice at both the item and family level. Passing counts as verified for this session; failing only produces reinforcement suggestions — no fixed improvement numbers are generated.

On-screen action: do not say "nobody has ever seen this question"; say only "a family unseen in this session".

## 1:45–2:00 — Report and boundaries

Screen: the report page's outcome at the top, then the four-state beliefs, session delta, evidence count, FSRS/7-day hint; end on the reinforcement suggestion or review date.

Narration:

> The report computes only what this session can prove has changed, and keeps the failure reason and the next review time. The 12 automation and 12 independent computer-use journeys prove the engineering loop closes — not real score gains. Next steps remain teacher spot-checks and real, compliant small-sample validation.

Final-frame caption:

```text
YHer Chemistry Demo
pre-alpha · localhost · evidence-bound
```

## Forbidden language

- "It is live" or "students can use it at scale right now"
- "The AI fully understands the student"
- "Mastery improved X% after watching the video"
- "All 28 nodes are currently open"
- "All 43 video tracks are signed and enabled"
- "12 QA journeys prove effective score improvement"
- "Free dynamic deep explanation has fully passed"

## Alternate ending

If the held-out fails during the real recording, keep the failure on screen; do not re-record until everything passes. Use this narration instead:

> Independent verification did not fully pass, so the system did not write "mastered". Instead it offers reinforcement from a different question family and a next verification plan. That is what honest loop closure looks like.
