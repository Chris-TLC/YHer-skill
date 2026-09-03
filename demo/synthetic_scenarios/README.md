# SYNTHETIC_DEMO scenarios

The 24 scenarios in this directory exist only for reproducible engineering demos. They are not real student records and cannot be used as evidence of learning gains, retention, or user scale. Every scenario and episode is explicitly marked `"synthetic": true`.

## Isolation rules

- The replayer makes no network calls and uses no paid LLM.
- Default output goes to `/tmp/yher_synthetic_demo_replays/`.
- The replayer refuses to write into `data/local_store/`, `data/study_logs/`, or anywhere else under the repository's `data/`.
- Synthetic user IDs use the `SYNTHETIC_DEMO_` prefix.
- The product currently opens 27 nodes; the plan matrix keeps the 28th ("化学反应速率" / reaction rate) reserved and replays it as expected-closed, rather than bypassing the five-family opening gate.

## Verification and replay

```bash
python3 -m demo.synthetic_scenarios.validate
python3 -m demo.synthetic_scenarios.replay \
  --output /tmp/yher_synthetic_demo_replays/manual_run
```

The first command validates 24 scenarios, 32 episodes, the budget/outcome matrix, node coverage, and current open status. The second runs a fully offline deterministic replay with the real R5 catalog and the real `SessionService`.
