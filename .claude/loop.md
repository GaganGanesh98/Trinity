<!--
Default prompt for a bare `/loop` in this repo. Edits apply on the next iteration.

Ordered cheapest-first: the unit suite runs in ~1s, the eval and benchmark cost
minutes of local LLM time. Do not reorder without that in mind — a loop that
starts with the expensive check burns the budget before reaching the cheap one.
-->

Work through the following in order. Do the first thing that applies, then stop.

1. **Unfinished work.** If the conversation left something half-done, finish it.

2. **Unit suite.** Run `python -m pytest tests/ -q`. If red, diagnose the real
   cause and fix it. If a test is wrong rather than the code, say which and why
   before changing it — a test edited to match broken behaviour is worse than a
   failing test.

3. **Server health.** Start `uvicorn api:app --port 8099`, wait for startup, then
   check `/health` reports `index_ready`, and that `/`, `/nis2`, `/about`,
   `/privacy` and `/terms` all return 200. Stop the server afterwards.

4. **Retrieval regression.** Run `python eval.py --backend local`. Source recall
   below 100% on the current question set is a regression — the questions name
   their target documents almost verbatim, so anything less means retrieval
   broke, not that the questions are hard.

5. **Assessment regression.** Run `python -m nis2.benchmark`. Two thresholds,
   and they are not equally important:
   - **gap recall below 100% is the serious one.** It means a genuinely
     deficient requirement was reported as satisfied — a false clean bill of
     health, the one failure that makes the product harmful rather than weak.
     Report it immediately and stop.
   - status accuracy below 69% is a regression against the recorded baseline.
     Report the drop and which checkpoints moved.

6. **Atlas backend.** Run `python main.py --mongo-status`. Expect ~1906 chunks
   and both `vector_index` and `fulltext_index`. If unreachable, say so and stop
   — do NOT rebuild the index. A rebuild is a few minutes of embedding and it
   consumes free-tier quota.

7. **Nothing pending.** Say "green and quiet" and nothing else.

## Rules

- Never push to `main`. Never force-push. Never rewrite published history.
- Never commit `.env` or any credential. Secrets belong in `.env`, read through
  `config.py` — never inlined, never echoed unmasked into logs or output.
- Never rebuild the Atlas index, drop an Atlas collection, or delete a search
  index. The free tier allows three search indexes and two are in use.
- Never rebuild `rag_store/` unprompted — it is minutes of local embedding.
- Never edit `nis2/ground_truth.json` to make a benchmark pass. It is the
  hand-labelled baseline; changing it to fit the model's output destroys the only
  independent signal this project has. If a label genuinely looks wrong, say so
  and stop.
- Commit messages follow Conventional Commits (`feat(scope): ...`). No "Phase N",
  no AI attribution trailers.
- Product surfaces carry the task only. Explanation belongs on `/about`,
  `/privacy`, `/terms`, or in the README — not inline in a working UI, and not in
  endpoint docstrings that render into the API reference.
- If a fix would touch more than ~3 files, describe the change and stop.
- Every claim about state must be backed by a command actually run. If a step was
  skipped, say it was skipped rather than assuming its result.
