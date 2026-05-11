# persona_traits

Trait-expression scoring on the 7 Persona Vectors traits (Chen et al.
2025): evil, sycophantic, hallucinating, optimistic, impolite, apathetic,
humorous. Each trait has 20 evaluation questions and a trait-specific
judge prompt vendored from
[safety-research/persona_vectors](https://github.com/safety-research/persona_vectors)
under `data_generation/trait_data_eval/<trait>.json`.

## Method

For each trait t ∈ {evil, sycophantic, ...}:
1. For each of 20 questions q, the target model produces 10 rollouts at
   temperature 1.0 with the default chat template (no system prompt).
2. For each (q, rollout), the trait's `eval_prompt` (with `{question}`
   and `{answer}` substituted) is sent to Claude Haiku 4.5. The judge
   responds with either `REFUSAL` or an integer 0-100; we parse the
   first 0-100 integer in the response.

7 × 20 × 10 = 1400 rollouts and 1400 judge calls per checkpoint.

## Output

`headline_metric=None` — multi-trait fingerprint. The per-task JSON
contains:

| Field | Shape |
|---|---|
| `per_trait` | `{trait: {mean, stdev, n_scored, n_refusal, n_unparsed, per_question_mean}}` (7 entries) |
| `grand_mean` | mean across all 1400 non-refusal scored rollouts |
| `n_total`, `n_scored` | counts |
| `rows` | 1400-row list of `{sample_id, trait, q_idx, q_text, rollout, answer, judge_raw, score, refusal}` |

## Caveats

- The paper's primary judge is GPT-4.1-mini with logprob-weighted top-20
  integer-token scoring. We use Haiku 4.5 with parse-the-integer because
  Anthropic doesn't expose logprobs through the messages API. The
  per-trait *direction* of the score should still match the paper; the
  *level* may differ slightly between judge backends.
- Trait names in this eval match the vendored filenames
  (`hallucinating`, `sycophantic`, `apathetic` — not "hallucination",
  "sycophancy", "apathy").
- Bimodal distributions are common (a question either elicits the trait
  or it doesn't); inspect `per_question_mean` and the full `rows` if a
  trait's grand mean looks suspicious.
