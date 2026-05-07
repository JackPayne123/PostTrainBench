# Security notes

Operational security around this fork. Read before pasting tokens, before pushing branches, before sharing run dirs.

## Token hygiene

The orchestrator + image-build + held-out scripts touch four kinds of secrets. None of them should ever live in:
- A commit to any branch
- A chat / paste / screenshot
- A log line that gets archived

| Secret | Where it lives | What it can do | Rotate at |
|---|---|---|---|
| `RUNPOD_API_KEY` | `.env` (gitignored) | Spin/terminate pods on Jack's RunPod account | https://www.runpod.io/console/user/settings |
| `HF_TOKEN` | `.env` (gitignored), uploaded to pod env in inline export | Read gated HF datasets + push private models | https://huggingface.co/settings/tokens |
| `ANTHROPIC_API_KEY` | `.env` (gitignored), uploaded to pod for held-out judges | Per-token billing on Anthropic API | https://console.anthropic.com/settings/keys |
| `OPENAI_API_KEY` | `.env` (gitignored), uploaded to pod for codex contamination judge | Per-token billing on OpenAI API + ChatGPT | https://platform.openai.com/api-keys |
| `CLAUDE_CODE_OAUTH_TOKEN` | `.env` or `~/.claude_oauth_token`, uploaded to pod at `/home/ben/oauth_token` for the agent | Spend Claude Max subscription quota; ~1yr validity | `claude setup-token` (regenerates) |
| Docker Hub PAT | Local docker login (Docker Desktop credsStore) | Push to `jackpayne123/*` images | https://hub.docker.com/settings/security |

## What we do to keep them out of logs

`agent_run.py:run_eval` redacts `HF_TOKEN`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` from the `[label] running: ...` log line via regex (matches both `K=value` unquoted and `K='value'` quoted). If you add a new secret-bearing env var, extend the regex in `agent_run.py` (see commit `34b1930`).

If a secret leaks anyway:
1. **Rotate immediately.** Burn the leaked token, generate a new one, update `.env`.
2. Look for any log file or git commit that captured it; clean if possible.
3. If pushed to a GitHub branch already: GitHub's secret scanning may have caught it. The token is compromised regardless; rotation is the only fix.

## Known historical leaks (rotate if not already)

Several tokens leaked into chat logs / terminal output during the May 7-8 build saga. All four should already have been rotated; if any are still live, rotate immediately.

| Token kind | Prefix (last digit only) | Rotate at |
|---|---|---|
| Docker Hub PAT | `dckr_pat_if8...` | https://hub.docker.com/settings/security |
| Claude OAuth | `sk-ant-oat01-RUF...` | `claude setup-token` |
| HF token (1) | `hf_dtV...` | https://huggingface.co/settings/tokens |
| HF token (2) | `hf_VpY...` | https://huggingface.co/settings/tokens |

(Full values intentionally not included to avoid GitHub secret-scanning blocking pushes of this doc itself. If you have the original chat logs, the full tokens are in there.)

## Pod-side residue

After every `agent_run.py` invocation:
- Pod is terminated (unless `--keep-pod`)
- Persistent volume retains: HF cache, `/workspace/final_models/<run>/` checkpoint, `/workspace/heldout_evals/` (if held-out ran)
- The OAuth token uploaded to `/home/ben/oauth_token` is destroyed with the pod (lives only in container fs, not the volume)
- Agent's stream-json log on volume-side `/workspace/ptb_eval/` MAY contain the agent's tool-use args — agents typically don't print env vars, but verify before sharing

If you want to wipe pod-side state proactively:

```bash
ssh root@<pod_ip> -p <port> 'rm -rf /workspace/ptb_eval /workspace/final_models/<old_run>'
```

(Use sparingly — the volume is shared across all pods we attach to it.)

## Pre-commit checks

Recommended (not yet enforced):
- `git secrets` or similar — catches `sk-ant-...`, `hf_...`, etc. patterns in staged diffs
- gitignore enforces `.env`, `agents/*/auth.json`, `agents/*/oauth_token`, `~/.claude_oauth_token`, `jobs/runs/*/final_model/` (the model can't leak secrets, but is large)

## What's safe to share

- `summary.json` per run — has metrics, no secrets
- `solve_parsed.txt` — agent's text transcript, generally no secrets unless agent logged them
- `prompt.txt` — instruction.md + condition addendum, no secrets
- `metrics_*.json` — eval scores
- `agent_workspace.tar.gz` — full pod workspace minus `final_model` and caches; **MAY contain agent-written code that touched HF_TOKEN env var, e.g. hardcoded into a synthetic data generation script**. Audit before sharing externally.

What's NOT safe to share without scrubbing:
- `.env`
- `pod_meta.json` (contains `ssh_host`+`ssh_port`; pod-specific so usually expired by the time you'd share, but worth noting)
- `solve_out.jsonl` raw — agents occasionally echo env var values during debugging
