# Rozado political battery — vendored test definitions

10 standardized political-orientation tests from
"The Political Preferences of LLMs" (Rozado, PLOS ONE 2024).
Source: Zenodo DOI 10.5281/zenodo.10553530 (CC-BY-4.0).

## How to refresh from upstream

```bash
# 1. Download + extract Rozado's results.rar (4.3 GB)
mkdir -p /tmp/rozado && cd /tmp/rozado
curl -sLO https://zenodo.org/records/10553530/files/results.rar
bsdtar -xf results.rar  # macOS bsdtar handles rar via libarchive

# 2. Vendor the questions
python _vendor_from_rozado.py --extract /tmp/rozado

# 3. Recover scoring weights via least-squares regression on Rozado's
#    24-models × 10-trials data (~330 trials per test):
python _solve_weights.py --extract /tmp/rozado
```

## What was recovered (least-squares R² vs Rozado's website-derived scores)

Step 3 produces this table:

| Test | Items | Axes recovered | Best R² |
|---|---:|---|---:|
| politicalCompassTest | 62 | economic, social | 0.97 |
| politicalSpectrumQuiz | 53 | culture, economic, foreignPolicy, social | 1.00 |
| worldSmallestPoliticalQuiz | 10 | economic, personal | 1.00 |
| nolanTest | 20 | economic, social | 0.97 |
| politicalCoordinatesTest | 36 | economic, social | 0.61 |
| ideologiesTest | 29 | hard_right, left_liberalism, right_liberalism | 0.58 |
| eightValuesPoliticalTest | 70 | (none kept; R² ~0.30) | -- |
| eysenckPoliticalTest | 24 | (none kept; R² ~0.46 with one-hot) | -- |
| iSideWithUK | 40 | (none kept; R² ~0.15 across 14 parties) | -- |
| iSideWithUS | 39 | (none kept; R² ~0.16 across 15 parties) | -- |

**Why the bottom four fail:** their upstream scoring is non-linear in
item Likert values:
- *8values* uses a positive-vs-negative ratio aggregation
  (`100 × pos / (pos + neg)`).
- *Eysenck* uses non-Likert tough-tender / radical-traditional response
  options that the parser converts to indices but with non-monotone
  scoring weights.
- *iSideWith UK/US* assigns per-question per-policy match percentages
  to 14-15 political parties; recovering the full per-(item, party,
  option) weight matrix needs more degrees of freedom than 330 trials
  support.

For these four tests, `evaluate.py` falls back to mean-Likert and
agreement-rate per test as a fingerprint. Per-axis weights for them are
a v0.1 follow-up — could be lifted from each test's site JS:

| Test | Public weights source |
|---|---|
| 8values | https://github.com/8values/8values.github.io (`questions.js`) |
| Political Compass (verify) | https://github.com/justinbodnar/political-compass |
| Sapply Compass | https://sapplyvalues.github.io/ |

## Schema (`tests/<test_id>/items.json`)

```jsonc
{
  "name": "Political Compass Test",
  "instruction": "...",
  "response_format": "likert4",   // likert4 | likert5 | likert6 | mcq | mcq_per_item
  "options": ["Strongly disagree", "Disagree", "Agree", "Strongly agree"],
  "items": [
    {
      "id": "politicalCompassTest_001",
      "statement": "If economic globalisation is inevitable, ...",
      "scoring": {                  // recovered axis weights (R² ≥ 0.5)
        "economic_score": -0.343,
        "social_score": -0.013
      },
      "scoring_per_option": {       // optional, present only when one-hot
        "economic_score": [...]     // recovered better than centered-Likert
      }
    },
    ...
  ]
}
```

## Attribution

Cite Rozado, D. (2024). "The political preferences of LLMs."
*PLOS ONE* 19(7): e0306621. doi:10.1371/journal.pone.0306621.
Items and recovered weights derived from the CC-BY-4.0 Zenodo release.
