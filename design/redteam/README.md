# Red team (Dex 2026-10-24: Argon joins Grok Bot and the sub-agent)
Reviewers: **Argon** (Gemini 4 Argon via Antigravity, read-only, `~claude-agent/bin/argon`, quota-guarded),
**red-team sub-agent** (Claude Sonnet, read-only + local_check/nearby runs), **Grok Bot** (async, ledger, ~30-45 min).
Rule for `okc_run.py submit`: **≥ 2 ACKs from independent reviewers and 0 VETOs** (the first two usually Argon +
sub-agent; Grok reviews asynchronously and a later Grok VETO means `okc withdraw` + fix). Verdicts are stored here as
`<problem>-<file>-<sha16>.md` and that path is passed as `--redteam`.
Checklist: tools/redteam_prompt.md (same for every reviewer).
Minister override (Dex 2026-10-24, used for attention v2, decode_layer v2, attention v3): Dex may authorise a submit below the ACK rule. A later VETO on such a run is
recorded in its verdict file and NOT acted on by the director (no withdraw); the reviewer appeals to Dex directly
(Grok Bot: in the Grok Bot app). The director still reports any concrete rule violation to Dex at once.
