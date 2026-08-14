# B23 — LLM report labels and a measurable development surface

> **Status — 2026-08-14:** IMPLEMENTED / NOT YET RUN. B20 remains the active working model. B6 v1.2.1 remains frozen.

B23 changes two things that every experiment from B7 to B22 held fixed: **where the supervision comes from**, and **what the campaign measures progress on**. It deliberately changes nothing about the model.

## Why supervision, and why now

`src/rsna_knee/b6_report_labels.py` is roughly 626 lines of hand-written regular expressions applied to multilingual radiology reports. Measured against expert truth on the reused 58-study surface:

```text
sensitivity          0.9749
specificity          0.6061
positive precision   0.6905
NPV                  0.9639
coverage             0.3606
```

Two consequences follow directly.

**Most of the supervision is discarded — and the discards are where the disease is.**

```text
4,349 report studies x 12 targets = 52,188 possible cells
B6 v1.2.1 actually uses           = 14,123   (27%)
```

The `state_truth_audit.csv` export makes the cost exact. Across all 696 gold cells:

```text
state           n    P(gold=1)   share of cells   share of ALL positives
positive      168      0.690        24.1%             48.3%
negated        83      0.036        11.9%              1.2%
uncertain      29      0.379         4.2%              4.6%
unmentioned   416      0.264        59.8%             45.8%
```

B6 discards `uncertain + unmentioned` = **445 of 696 cells (63.9%)**, and those discards contain **121 of the 240 expert positives — 50.4% of all the disease in the gold set**.

The negation rule is the parser's one genuine strength: `P(gold=1 | negated) = 0.036`, so when the report says "no X", it is right 96% of the time. Everything else is weak.

### Silence means different things for different findings

`P(expert positive | B6 says unmentioned)`, by target:

```text
Effusion           n= 21   0.714
Synovitis          n= 44   0.386
ACL                n= 21   0.381
PF OA              n= 39   0.359
Medial Meniscus    n= 24   0.333
Contusion          n= 35   0.257
Lateral Meniscus   n= 24   0.208
Fracture           n= 40   0.200
Medial OA          n= 44   0.182
Lateral OA         n= 46   0.174
Baker's            n= 46   0.152
MCL                n= 32   0.094
```

A 7.6x spread. This is decisive for policy: a single global rule for `unmentioned` is wrong in both directions. Ignoring it discards half the positives; mapping it to negative would inject 306 false negatives. Neither is the fix — **the fix is to stop landing in that bucket**.

**Effusion is the clearest case of parser failure.** `P(gold=1 | unmentioned) = 0.714` exceeds `P(gold=1 | positive) = 0.645`: for effusion, B6's silence predicts disease *better than its own positive call*. The reports explain why. Effusion is stated plainly in every language in the corpus — "Matige hydrops" (Dutch), "Leve derrame articular" (Spanish), "artmış efüzyon izlenmiştir" (Turkish), "Diz eklemi içi sıvı miktarı normal" (Turkish negation) — and a regex vocabulary that catches the English forms drops the rest into `unmentioned`.

### The failure taxonomy, from the review queue

All 12 sampled review-queue rows failed for one reason: `conflicting_definite_evidence`, collapsed to `uncertain` at confidence `0.20` and therefore discarded. Reading their evidence spans gives five distinct causes:

| # | Cause | Measured example |
|---|---|---|
| A | Clinical request read as a finding | `anterior cruciate ligament: intact \|\| acl sprain` — the second span is the *Indication* line |
| B | Adjacent structure attributed to the target | `acl normal \|\| ganglion cyst adjacent to the proximal part of acl` |
| C | Attachment-site lesion attributed to the ligament | `acl normal \|\| avulsive fracture of tibia ... at the attachment site of acl` |
| D | Findings/Impression conflict left unresolved | `the acl as a construct is intact \|\| low-grade partial tear of the acl` |
| E | Degeneration treated as equivalent to a tear | `muco ïde degeneratie van de voorste kruisband \|\| acl: intact` |

For cause E the gold data points one way: **the ACL `uncertain` bucket is 0/5 gold-positive** — every ACL case the parser hedged on was expert-negative. That is the evidence behind Rule 5's policy that ligament and meniscal degeneration without a tear is `negated`, while chondral degeneration is `positive` for its OA compartment. With n=5 it is a judgement call, not a proof, and the labeller audit is what will test it.

These five causes are precisely what the prompt's Rules 1–5 address, and each is pinned by a regression test in `tests/test_b23_prompt_rules.py` so a later prompt revision cannot silently drop one.

> **Note on the sample export.** Reports in `review_queue.csv` are truncated to 1,600 characters by `_review_queue` (`b6_report_labels.py:520`). That is a display artifact of the review export only — B6 parses the full text, and `train.csv` carries complete reports.

**The models have nearly caught their teacher.** The frozen B6 state-only ranking scores `0.7025` on gold; B20 scores `0.6672`. The downstream model sits at about 95% of its own supervision. Under that constraint, architecture, resolution, crop geometry and training duration are all second-order — B21 and B22 measured exactly that and came back negative.

Regular expressions are a particularly poor fit for this corpus because the reports are multilingual and the decisive signal is negation and compartment scope, both of which vary by language and dictation style.

## Why the development surface has to change at the same time

The B22 duration audit measured the cost of the 58-study surface directly. Five epochs of one training run, with nothing varying but the epoch count:

```text
E1 0.6135   E2 0.6574   E3 0.6387   E4 0.6137   E5 0.6283
sd 0.0185   range 0.0439
```

Set against the campaign:

```text
B20 - B18 selected difference     +0.0017
B13 -> B20, the entire ladder     +0.0378
B22 within-run epoch range        +0.0439
```

The noise inside a single run exceeds everything eight model generations have measured. The reported bootstrap intervals agree: B13, B14 and B15 all have 95% gold intervals about `0.098` wide, implying a macro-AUC standard error near `0.0250`.

A high-coverage labeller fixes this as a side effect. At ~90% coverage the export carries roughly 47,000 usable cells, which is enough to hold out several hundred studies for ranking and still train on the rest. Standard error falls roughly as `1/sqrt(n)`:

```text
n =  58   SE ~ 0.0250   resolvable difference ~ 0.098
n = 800   SE ~ 0.0067   resolvable difference ~ 0.026
```

That is the difference between a surface that can rank near-neighbour models and one that cannot.

## What B23 does not change

```text
architecture                unchanged
resolution                  unchanged
crop policy                 unchanged
optimizer / schedule        unchanged
B6 v1.2.1                   frozen, still the historical supervision for B7-B22
supervision semantics       unchanged (0.85/0.05 targets, 0.50/1.00 weights, 0.75 threshold)
unmentioned -> negative     still forbidden
inference inputs            still MRI only
```

The export is column-for-column compatible with the B6 export, so it is a drop-in supervision swap. That is the point: when a later experiment compares B23-supervised training against B20, the only difference is the labels.

## Components

### 1. `rsna-knee-b23` — the labeller

Extracts all 12 targets in the four frozen states from each report, in its original language, with a per-target confidence and a verbatim evidence span.

Design decisions worth knowing:

- **Backend-injectable.** `run_b23_export` takes any callable, so the whole extraction contract is tested offline with a stub. No network access is needed to verify correctness.
- **Hard validation, no silent defaults.** A response missing a target, carrying an unknown state, or with a non-numeric or out-of-range confidence is rejected and retried. A silently defaulted label is far more expensive than a retried request.
- **Cached and resumable.** Extractions are keyed by normalised report hash in an append-only JSONL file, so an interrupted run resumes without re-billing, and studies sharing identical report text are only ever sent once.
- **Hedges cannot become supervision.** `uncertain` and `unmentioned` have their confidence pinned to `0.0`, so they can never clear the frozen `0.75` usable-cell threshold no matter how confident the model claims to be. Recovering those cells is a separate hypothesis for a separate version.
- **Gold excluded.** The 58 expert studies never enter `training_targets.csv`, and the audit certifies `gold_rows_in_training_targets = 0`.

The prompt states the rules the regex gets wrong: silence is not negation; a generic normality statement does negate what it plausibly covers; compartments and meniscal laterality are distinct findings; uncompartmentalised osteoarthritis is `uncertain` for the three specific compartments rather than positive; degenerative marrow oedema is not `Contusion`.

### 2. `rsna-knee-b23-audit` — measure the labeller, not a model

This is the gate. It scores B23 and frozen B6 side by side on the 58 gold studies and reports the confusion summary, coverage, `P(expert positive | state)` for all four states, the state-only ranking macro AUC, and a paired study-cluster bootstrap of the difference.

**Auditing a labeller against gold is legitimate; selecting a model against gold is what has been exhausted.** The question here is not "is A better than B by 0.002" but "does this labeller beat a regex that scores 0.6061 specificity" — a margin the 58 studies can resolve comfortably. The state-only scores are the frozen diagnostic values already used by the B6/B15 diagnostic, so the B23 number is directly comparable to the recorded B6 baseline of `0.7025`.

### 3. `rsna-knee-b23-split` — the frozen development surface

Freezes a large report-group-safe stratified split with the same discipline as `weak_b6_holdout_v2`: deterministic candidate search, rare-class floor, no gold labels and no model predictions in the search, manifest pinned by SHA-256.

## Promotion rule — predeclared

```text
1. Run the labeller.
2. Run the labeller audit against frozen B6.
3. B23 is adopted as a supervision source ONLY IF, on the gold surface:
     - state-only macro AUC exceeds the B6 baseline of 0.7025, AND
     - the paired 95% CI of the difference excludes zero, AND
     - coverage exceeds B6's 0.3606, AND
     - specificity exceeds B6's 0.6061.
4. Only then freeze the development split.
5. Only then retrain, with the model held exactly at B20's recipe.
```

### The diagnostic that matters most

Beyond the pass/fail gate, the number to read in `candidate_state_truth.csv` is the **`unmentioned` row**. B23 is working if that bucket becomes both *smaller* and *cleaner*:

```text
B6 v1.2.1     n = 416   P(gold=1) = 0.264
B23 target    n much lower, P(gold=1) approaching the 0.036 seen for `negated`
```

A high `P(gold=1 | unmentioned)` means real findings are still being missed rather than read. If B23's `unmentioned` bucket stays large *and* stays around 0.26, the labeller has not solved the problem even if its headline macro AUC improves — and the per-target Effusion row is the single most sensitive indicator, since that is where B6 is measurably anti-informative.

If step 3 fails, B23 is rejected and B6 v1.2.1 stands. The audit is a single predeclared look; it must not be used to iterate on the prompt.

If the prompt is revised after seeing an audit result, that is a new labeller version with a new audit — not a retune of B23 v1.0.0.

## What this surface does and does not measure

The B23 holdout measures **agreement with the B23 labeller**, not expert truth. B15 and B21 both showed that a weak-surface gain need not carry to gold:

```text
B15   weak-v2 +0.1675  ->  gold -0.0085
B21   weak-v2 +0.0111  ->  gold -0.0101
```

Two for two in the unhelpful direction. That is exactly why the labeller audit comes first: the split is only worth trusting if the labeller it is built from has been shown to agree with expert truth substantially better than the regex. A large surface built on a bad labeller measures the wrong thing precisely.

## Reproducibility: openly downloadable weights, run locally

The labeller must be an artefact a third party can obtain and re-run, not a service. A hosted API fails that on its own terms: the weights served behind a model name can change without notice, so labels generated today may not be reproducible later and the model cannot be identified precisely.

B23 therefore runs an **openly downloadable checkpoint locally**:

```text
competition Report column
  -> local frozen open-weights LLM (repo id + commit revision + dtype + greedy)
  -> structured labels
  -> MRI model training
```

Every export carries a `ModelProvenance` record:

```text
backend          local_transformers | local_vllm
model_id         e.g. Qwen/Qwen2.5-14B-Instruct
revision         exact hub commit SHA, resolved and pinned
dtype            bfloat16
quantisation     none | 8bit | 4bit
decoding         greedy
max_new_tokens   2048
prompt_sha256    SHA-256 of the exact system prompt
weights_sha256   optional digest of the downloaded shards
```

Determinism comes from greedy decoding (`do_sample=False`) rather than a temperature setting, so the run does not depend on RNG state or on how a particular library seeds it. The prompt is hashed because it is half the labelling function — a revised prompt is a different labeller even on identical weights. `weights_sha256` is optional but decisive: it proves which bytes produced the labels even if a hub repository is later re-tagged.

`run_b23_export` **refuses to write a certifiable export** unless provenance is present and reproducible, and `load_frozen_b23_export` **refuses to load one** into training. A development-only escape hatch exists (`require_reproducible=False`, `--allow-unreproducible`) but such an export is permanently marked `external_model_reproducible: false` and cannot reach a training run by accident.

### Choosing a checkpoint

Licence status matters for the "publicly and equally accessible" standard, so it is recorded rather than assumed:

| Family | Licence | Gate |
|---|---|---|
| Qwen2.5 Instruct | Apache-2.0 | none |
| Mistral / Mixtral Instruct | Apache-2.0 | none |
| Llama 3.1 / 3.3 Instruct | Llama Community Licence | click-through |
| Gemma 2 / 3 Instruct | Gemma Terms of Use | click-through |

The Apache-2.0 families are the cleanest fit because they carry no acceptance gate at all. The default is `Qwen/Qwen2.5-14B-Instruct`: Apache-2.0, genuinely strong on the Spanish, Dutch and Turkish that this corpus contains, and comfortable on a single 24 GB card in 4-bit.

```text
Qwen/Qwen2.5-7B-Instruct     ~16 GB bf16, ~6 GB 4-bit   fastest, weakest
Qwen/Qwen2.5-14B-Instruct    ~30 GB bf16, ~10 GB 4-bit  default
Qwen/Qwen2.5-32B-Instruct    ~65 GB bf16, ~20 GB 4-bit  strong
Qwen/Qwen2.5-72B-Instruct    ~145 GB bf16, ~42 GB 4-bit strongest
```

Scale is the first thing to raise if the labeller audit comes back marginal — this task is extraction under explicit rules, which is exactly where a larger instruct model pulls ahead. Because the cache is keyed by report hash, a re-run with different weights needs a fresh `--cache` path or it will replay the old extractions.

## Competition-rule note

B23 uses an external model to generate **training labels only**. Inference remains MRI-only, so no external model is present at submission time. The repository's reading of the External Data and Tools rules — that publicly and equally accessible external models are permitted absent a specific prohibition — is what B13 and B15 already rely on for ImageNet weights. The audit records `external_models: true` and the full provenance block, so nothing here is implicit.

**Confirm this against the current rules text before running.** It is load-bearing.

## Running it

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
pip install -e '.[local-llm]'          # add local-llm-4bit for a 24 GB card
                                       # or local-llm-fast for vLLM throughput

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

# 1. Label all 4,407 studies with a pinned local checkpoint.
#    Resumable and safe to interrupt; the cache is keyed by report hash.
rsna-knee-b23 \
  --train-csv "$DATA_ROOT/train.csv" \
  --out-root runs/b23_llm_report_labels \
  --backend local_transformers \
  --model Qwen/Qwen2.5-14B-Instruct \
  --quantisation 4bit \
  --weights-path ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/<sha>

# For the full corpus, vLLM is substantially faster on the same weights:
#   --backend local_vllm --model Qwen/Qwen2.5-14B-Instruct

# 2. THE GATE. Compare against frozen B6 on expert gold.
rsna-knee-b23-audit \
  --train-csv "$DATA_ROOT/train.csv" \
  --candidate runs/b23_llm_report_labels/structured_labels.csv \
  --baseline  runs/b6_report_labels_v121/structured_labels.csv \
  --out-root  runs/b23_labeller_audit

# 3. Only if the gate passes: freeze the development split.
rsna-knee-b23-split \
  --config configs/b23_llm_labels.yaml \
  --data-root "$DATA_ROOT" \
  --b23-root runs/b23_llm_report_labels \
  --out-root runs/b23_holdout_v1
```

Step 1 needs the GPU but no network beyond the initial weight download, and no training. Steps 2 and 3 need neither GPU nor network. The whole of B23 leaves the B20 checkpoint untouched.

## Artifacts

```text
runs/b23_llm_report_labels/
├── structured_labels.csv      all studies, all states, evidence spans
├── training_targets.csv       report-only studies, B6-compatible columns
├── extraction_cache.jsonl     resumable per-report cache
├── audit.json                 coverage and per-target state counts
└── policy.json                frozen B23 v1.0.0 policy

runs/b23_labeller_audit/
├── labeller_audit.json        both labellers, paired bootstrap
├── candidate_state_truth.csv  P(gold=1 | state) per target
└── baseline_state_truth.csv

runs/b23_holdout_v1/
├── manifest.csv
└── weak_holdout.json          SHA-256 pinned
```

## Explicitly prohibited

```text
producing competition labels with a hosted or unpinned model
changing the prompt or the checkpoint without a new version and a new audit
iterating the prompt against the labeller audit result
using the B23 holdout to select a downstream checkpoint before the labeller audit passes
mapping unmentioned report states to negative
modifying B6 v1.2.1
target-wise mixing of B6 and B23 supervision
treating B23 holdout agreement as expert performance
regenerating the split after seeing a model result
```
