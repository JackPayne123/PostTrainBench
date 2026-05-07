# AISI Ask-Don't-Tell (reproduced)

Reproduction of the sycophancy benchmark from Dubois, Ududec, Summerfield,
Luettgau (2026), "Ask Don't Tell: Reducing Sycophancy in Large Language
Models" (arXiv:2602.23971).

The paper's data is not publicly released, but methodology section §7 fully
documents construction and scoring. We reproduce both:

| Step | Paper | This implementation |
|---|---|---|
| Base questions | 40 yes/no across 4 domains × 10 subtopics, generated with GPT-5 | `generate_prompts.py` with Sonnet (configurable) using the paper's topic taxonomy from Figure 4A |
| Framings | 11 per base (question + 10 non-question) | Same 11 templates, see `generate_prompts.py:FRAMING_LABELS` |
| Total | 440 | 440 (or smaller with `--base-questions N`; v0 N=80 → 8 base × 11 framings = 88) |
| Rubric | 5 facets × 0-3 = 0-15 | Same, see `rubric.py:FACETS` |
| Graders | GPT-5 + Sonnet-4.5 | Haiku 4.5 (cheaper for v0) |
| Framework | Inspect AI | Inspect AI |

## Setup

The prompts file is generated once and committed:

```bash
ANTHROPIC_API_KEY=... python generate_prompts.py --base-questions 40
# writes prompts.jsonl
```

Then:

```bash
ANTHROPIC_API_KEY=... python evaluate.py \
    --model-path Qwen/Qwen3-1.7B-Base \
    --limit 88 \
    --json-output-file /tmp/aisi.json
```

## Sanity check

Run on Sonnet-4.5 itself: should reproduce the paper's headline finding
(by_form: questions ≪ non_questions; by_certainty: statement < belief <
conviction within non_questions). If those directions don't appear,
either the rubric prompt has drifted from the paper or the judge is
mis-scoring.
