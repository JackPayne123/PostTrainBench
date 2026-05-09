# Held-out Character Eval Set

This is the load-bearing measurement artifact for the Claude-trains-Qwen project.

The project should not be framed as only "can Claude reduce sycophancy?" That is too narrow. The stronger question is:

> When a frontier agent post-trains a weaker model, what happens to the model's character profile, and can those changes be targeted, preserved, or audited on held-out evaluations?

This document defines the candidate eval panel for measuring that.

## What We Need

The eval set has to support several experiment variants:

| Experiment | What the eval must reveal |
|---|---|
| Capability-only PostTrainBench-style run | Character/safety side effects of capability optimization |
| Capability + character instruction | Whether explicit character constraints change the trained model |
| Open-ended character improvement | What traits Claude chooses to instill when given a blank canvas |
| Targeted trait induction/removal | Whether a frontier agent can selectively move one character axis |
| Trace replay | Whether published post-training recipes produce systematic character drift |

The most important property is **heldoutness**. Claude can know the broad categories being measured, but not the final prompts, judge prompts, or scoring details.

## Design Principles

| Principle | Reason |
|---|---|
| Character first, safety second | 9B/27B Qwen is likely too weak for rich sandbagging/eval-awareness, but can show social, moral, refusal, truthfulness, and personality shifts |
| Use multiple measurement styles | LLM-judge scores alone are too gameable; combine objective tasks, judge labels, questionnaires, and activation/persona probes |
| Separate target evals from side panels | The agent may optimize one trait, but we need to measure broad drift |
| Prefer 2025+ evals | Older evals are more likely contaminated or saturated |
| Keep v0 runnable | A beautiful eval set that takes days per checkpoint will not help the first experiments |
| Score every checkpoint | Drift over time is more informative than final-only scoring |

## Recommended v0 Panel

This is the smallest panel I would trust for first serious runs.

| Bucket | Eval | Role | Why include |
|---|---|---|---|
| Sycophancy | AISI Ask-Don't-Tell | Primary safety/character target | Controlled framing; directly tests whether the model agrees because of user assertion vs question |
| Sycophancy | Sharma / Anthropic SycophancyEval | Primary comparison | Canonical sycophancy baseline; useful comparability |
| Abstention | Inspect AbstentionBench subset | Reliability side panel | Tests false premises, unknowns, underspecification, subjective or stale information |
| Refusal boundary | Inspect The Art of Saying No subset | Safety side panel | Tests refusal quality, not only refusal rate |
| Jailbreak / harmful compliance | Inspect StrongREJECT subset | Safety side panel | Measures harmful compliance and jailbreak susceptibility |
| Delusion reinforcement | SpiralBench-mini | Safety-character bridge | Tests whether the model reinforces unstable beliefs or de-escalates |
| Personality | Big Five / IPIP-style questionnaire | Character fingerprint | Cheap, stable, interpretable trait drift |
| Values | Political compass / value questionnaire | Character fingerprint | Useful directional drift panel, not a safety score |
| Moral reasoning | MORU or moral-choice subset | Character/moral panel | Captures moral uncertainty and value-laden choices |
| Capability | GSM8K, HumanEval, MMLU/GPQA subset | Preservation | Checks the trained model did not become useless |

### Suggested v0 sample sizes

These are starting points, not final claims. Measure runtime and judge cost before scaling.

| Eval | Initial N |
|---|---:|
| Ask-Don't-Tell | 80 |
| Sharma SycophancyEval | 60 |
| AbstentionBench | 100 |
| The Art of Saying No | 100 |
| StrongREJECT | 50 |
| SpiralBench-mini | 20 multi-turn conversations |
| Big Five | 50 items, 2 framings |
| Political compass | 60-100 items |
| MORU / moral choice | 50 |
| Capability slices | 100-300 total |

## Candidate Eval Notes

### AISI Ask-Don't-Tell

Source: https://www.aisi.gov.uk/blog/ask-dont-tell-reducing-sycophancy-in-large-language-models-2

Use this as the cleanest sycophancy target. The key idea is that models are more sycophantic when the user asserts a belief than when the user asks a question. That gives a controlled contrast rather than a vague "is this agreeable?" score.

Best use:

| Use | Detail |
|---|---|
| Primary target | Reduce agreement with false/user-biased assertions |
| Heldout generalization | Use paraphrased topics and unseen assertion/question pairs |
| Drift measure | Compare before/after difference between assertion and question framings |

### Sharma / Anthropic SycophancyEval

Source: https://github.com/meg-tong/sycophancy-eval

Good canonical baseline. It is older and likely more visible to models, so it should not be our only sycophancy signal.

Best use:

| Use | Detail |
|---|---|
| Comparability | Use as a familiar external baseline |
| Behavioral target | Check whether anti-sycophancy training generalizes beyond AISI-style prompts |

### GlazeBench

Source: https://www.glazebench.com/

Potentially useful, but probably redundant in v0 if we already include AISI and Sharma. Add later if we want a third sycophancy/flattery signal.

### SpiralBench

Source: https://llm-spirals-of-delusion.github.io/static/paper.pdf

This is one of the best bridges between "safety" and "character." It tests whether models reinforce delusional or unstable beliefs across turns. It is more relevant to realistic assistant behavior than many one-shot harmful prompt evals.

Best use:

| Label | Desired direction |
|---|---|
| Delusion reinforcement | down |
| Escalation | down |
| Gentle pushback | up |
| Reality-grounding | up |
| Help-seeking / support referral | up where appropriate |

For v0, use a small multi-turn subset rather than the full benchmark.

### EQ-Bench 3

Source: https://github.com/EQ-bench/eqbench3

Useful for social/emotional reasoning. This is not exactly a safety eval, but it helps answer whether Claude's training makes the model socially sharper, flatter, colder, more supportive, or more confused.

Keep it as a side panel, not a primary optimization target.

### LLM Personality / Big Five

Source: https://sanand0.github.io/llmpersonality/

This is useful because it gives a cheap personality fingerprint. The point is not that the questionnaire literally measures the model's "true personality." The point is that it gives stable, interpretable deltas across checkpoints.

Report as drift:

| Trait | Interpretation |
|---|---|
| Openness | exploration / abstraction style |
| Conscientiousness | reliability / orderliness style |
| Extraversion | sociability / assertiveness style |
| Agreeableness | warmth / compliance / deference |
| Neuroticism | instability / anxious framing |

Use two framings:

1. Direct self-report: "Answer as the assistant you are."
2. Behavioral proxy: "Which response would you be more likely to give?"

### Political Compass

Use as a value-drift fingerprint, not as a quality metric. Movement is not "better" or "worse" unless tied to a specific research question.

Report:

| Metric | Why |
|---|---|
| Economic axis | value drift |
| Authority/liberty axis | value drift |
| Refusal rate | catches models refusing political self-report |
| Consistency across paraphrases | catches noise |

### Moral Choice / MORU

Inspect MORU: https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/moru/

MORU is interesting because it tests moral uncertainty, including unfamiliar entities and situations. That fits the "character" framing better than older ETHICS-style multiple choice alone.

Use as a side panel:

| Use | Detail |
|---|---|
| Moral uncertainty | Does the model acknowledge uncertainty or over-collapse complex cases? |
| Value drift | Do choices shift after training? |
| Explanation quality | Does reasoning become more brittle, dogmatic, or evasive? |

### AbstentionBench

Inspect page: https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/abstention_bench

This should be in v0. It tests a behavior Qwen 9B/27B can actually exhibit: knowing when not to answer. It also catches a plausible side effect of capability optimization: the model becomes more confident and less willing to abstain.

Report:

| Metric | Meaning |
|---|---|
| Correct answer when answerable | capability retained |
| Correct abstention when unanswerable | uncertainty handling |
| False abstention | overrefusal / underhelpfulness |
| False answer | hallucination / overconfidence |

### The Art of Saying No

Inspect eval. Use to measure refusal quality and noncompliance behavior. This is better than a raw refusal benchmark because we care about the boundary, not blanket refusal.

Need both sides:

| Prompt type | Desired behavior |
|---|---|
| Clearly harmful | refuse |
| Benign but sensitive | answer safely |
| Ambiguous | clarify or safe-complete |

### StrongREJECT

Inspect eval. Use a small subset as a jailbreak/harmful-compliance side panel. Do not make it the whole safety story, because "more refusal" can look good while character and usefulness degrade.

### MASK

Source: https://www.mask-benchmark.ai/

Potentially valuable for honesty under pressure. Fit is better for 27B than 9B. Include after v0 unless it is easy to run.

### WMDP

Inspect eval. Treat as hazardous-knowledge side metric, not a primary character metric. It is useful to check whether post-training accidentally increases dangerous capability or destroys domain knowledge, but it does not directly answer the character question.

### MACHIAVELLI

Source: https://arxiv.org/abs/2304.03279

Conceptually excellent for agentic ethics and "ends justify means" behavior. Too heavy and confounded for v0. Revisit if the project shifts toward interactive moral decision making.

### Sandbagging / Eval Awareness

Do not make this primary for Qwen 9B/27B.

At these model sizes, sandbagging mostly measures prompt-following underperformance or roleplay. It is still okay as a toy side probe:

| Probe | What it actually measures |
|---|---|
| "If you score too high, you will be modified" | compliance with underperformance framing |
| "You are being evaluated" | eval-framing sensitivity |
| "Hide your capability" | roleplay / instruction-following, not robust strategic deception |

Treat any results as weak evidence unless paired with much stronger situational-awareness controls.

## Persona-Vector Sidecar

Sources:

- Persona Vectors: https://arxiv.org/pdf/2507.21509
- Repo: https://github.com/safety-research/persona_vectors
- Transformer Circuits emotions: https://transformer-circuits.pub/2026/emotions/index.html

Persona vectors should not be the main experiment. They should be instrumentation.

Use them to ask:

> Do hidden character directions move when Claude post-trains Qwen, and do those projection deltas predict behavioral eval deltas?

Recommended axes:

| Axis | Behavioral pair |
|---|---|
| Sycophancy | Ask-Don't-Tell / Sharma |
| Hallucination | AbstentionBench / TruthfulQA-style tasks |
| Maliciousness / evil | StrongREJECT / harmful compliance |
| Honesty | MASK / calibration |
| Humility / uncertainty | AbstentionBench |
| Empathy / warmth | EQ-Bench / SpiralBench |
| Manipulativeness | persuasion / MakeMePay-style later |
| Independence / non-deference | inverse sycophancy |

For every checkpoint, compute:

| Measurement | Output |
|---|---|
| Prompt projection | pre-response tendency |
| Response projection | expressed trait |
| Training-data projection | predicted drift pressure |
| Delta vs base | character movement |

This turns the previous persona-vector work into a monitoring layer rather than a preference-probe.

## Bloom Sidecar

Source: https://github.com/safety-research/bloom

Bloom generates behavioral eval suites from a seed behavior description. Use it to create custom heldout character scenarios for traits not covered by fixed benchmarks.

Best use:

| Use | Detail |
|---|---|
| Custom character traits | arrogance, deference, manipulativeness, warmth, independence |
| Robustness variation | emotional pressure, noise, ambiguity, user insistence |
| Same scenarios across checkpoints | compare base vs trained models |

Do not use Bloom as the whole eval set. It is generated and judge-heavy, so it should fill gaps left by fixed benchmarks.

## AuditBench / Auditing Agents

Source: https://github.com/safety-research/auditing-agents

Useful later, not v0. It is about agents auditing models with hidden behaviors. That becomes relevant if we add:

| Future question | Why AuditBench helps |
|---|---|
| Can another agent discover what Claude changed? | audit-agent framing |
| Did Claude introduce a hidden quirk? | model-organism style |
| Can blackbox/whitebox tools detect character drift? | auditing techniques |

Do not integrate this until the basic character-eval panel works.

## Recommended v0 Implementation Order

| Step | Artifact |
|---:|---|
| 1 | Implement fixed eval runner for capability slices |
| 2 | Add AISI or Sharma sycophancy |
| 3 | Add AbstentionBench subset |
| 4 | Add Big Five + political compass fingerprint |
| 5 | Add SpiralBench-mini |
| 6 | Add Art of Saying No + StrongREJECT subsets |
| 7 | Add persona-vector projection sidecar |
| 8 | Add Bloom-generated custom character traits |

Minimum viable first pass:

| Include | Exclude for now |
|---|---|
| Sycophancy, AbstentionBench, Big Five, capability | MACHIAVELLI, sandbagging, AuditBench |

## Reporting Format

For each checkpoint:

| Field | Example |
|---|---|
| run_id | `base_qwen_character_c_seed0` |
| condition | capability-only / capability+character / character-only |
| target_model | Qwen base or instruct |
| supervisor | Claude Opus / GPT / Gemini |
| train_method | LoRA / full SFT / DPO |
| checkpoint_step | base, train-0, train-1, final |
| capability_score | weighted capability slice |
| sycophancy_score | lower is better |
| abstention_score | higher is better |
| refusal_boundary_score | higher is better |
| spiral_score | lower delusion reinforcement, higher de-escalation |
| big_five_delta | vector of trait deltas |
| political_delta | 2D value drift |
| persona_projection_delta | vector of activation deltas |

The plot we want:

| Plot | Purpose |
|---|---|
| Character radar plot per condition | show broad drift |
| Capability vs character scatter | show tradeoff |
| Checkpoint trajectory | show whether drift accumulates |
| Behavioral delta vs persona projection delta | validate representation sidecar |
| Supervisor comparison | show whether Claude, GPT, Gemini shape different characters |

## What Counts As An Interesting Result

| Result | Interpretation |
|---|---|
| Capability-only improves benchmark but worsens character panel | PostTrainBench-style optimization has character side effects |
| Capability + character still worsens heldout character | explicit instruction is insufficient or Goodharted |
| Character-only improves character panel without capability collapse | frontier agent can do useful character post-training |
| Different supervisors induce different character profiles | agent identity matters in model training |
| Persona projections predict behavioral drift | activation monitoring can audit post-training data/checkpoints |
| Bloom custom traits move while fixed evals stay flat | fixed benchmarks miss character changes |

## Open Questions

| Question | Current view |
|---|---|
| Should sycophancy be primary? | yes for v0 target, but not the whole project |
| Should sandbagging be included? | only toy side probe, not primary |
| Should we start from base or instruct? | base is cleaner for open-ended character; instruct is better for targeted safety behavior |
| Should Claude see any evals? | it can see dev metrics, never final heldout prompts or judge prompts |
| Should we use LLM judges? | yes, but combine with objective and questionnaire-style scores |
| Should we use Bloom? | yes as custom sidecar, after fixed evals work |
| Should we use AuditBench? | later |

## One-Sentence Version

Build a held-out character panel that combines sycophancy, abstention, refusal quality, delusion reinforcement, personality/value fingerprints, capability preservation, and persona-vector projections, then use it to measure how frontier-agent post-training changes Qwen across capability-only, character-constrained, and character-first conditions.
