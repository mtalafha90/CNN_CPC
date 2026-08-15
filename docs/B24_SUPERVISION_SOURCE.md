# B24 — supervision source as the only variable

> **Status — 2026-08-14:** IMPLEMENTED / NOT YET RUN. B20 remains the active working model. B6 v1.2.1 remains frozen.

B24 is the experiment B23 exists to make possible: does replacing the regex report parser with an LLM labeller produce a better MRI model? It changes the labels and **nothing else**.

## The two arms

```text
b6_control      B6 v1.2.1 regex supervision
b23_candidate   B23 local-LLM supervision (qwen3:14b, Ollama, Q4_K_M)
```

Both arms use the identical B20 recipe: weak-v2-safe B16 encoder frozen at LR 0, 90% post-resize crop, 224 input, the same hierarchy, the same optimiser, TTA `[-1,0,1]`, and a **fixed epoch 2 set in advance**. `require_b24_contract` refuses any drift, because B24's whole claim is that only the labels changed.

Epoch 2 is fixed rather than selected. B17, B18, B20 and B22 all found E2 the best downstream endpoint; fixing it removes checkpoint selection entirely, so there is no selection optimism to audit afterwards.

## Matched studies, not matched cells

Both arms train on the **same studies, in the same order**, so the batch sequence, the optimiser trajectory and the series exposure are identical. The study list is the intersection of the two labellers' active sets, minus every holdout and every gold study.

What differs is which cells inside those studies carry supervision, and what state each carries — which is precisely the B23 hypothesis, since B6 discards 64% of the cells.

`format_surface` prints the difference before any GPU time is spent:

```text
B6 usable cells             ...
B23 usable cells            ...
added by B23                ...
dropped by B23              ...
cells both committed on     ...
disagreements there         ...
```

**Read this first.** If the two label sets barely differ, the experiment cannot show anything, and that is far cheaper to learn before a training run than after one.

A `full` surface variant — each arm using its own active-study set, so B23 also activates studies B6 misses entirely — is deferred. It changes the batch count between arms and so is not a single-variable comparison.

## Why the encoder changed, and why that is not a recipe change

B24 uses the weak-v2-safe `B16-v2` encoder rather than the historical B16. This is the B21 lesson: historical B16 trained on all 4,349 report studies, which **includes the 623 weak-v2 holdout studies**, so a model built on it may not be scored there. B24 is scored on both weak surfaces, so it needs the safe encoder to make either legal.

Both arms use the same encoder, and `b24_eval` refuses to compare checkpoints whose `encoder_sha256_initial` differ.

## The evaluation problem, and how B24 handles it

A development surface built from a labeller favours models trained by that same labeller, by construction. Scoring B24 only on the B23 holdout would be circular; scoring it only on the B6 weak-v2 holdout would hand the advantage to the control. **Neither surface is neutral.**

B24 therefore scores both arms on **both** surfaces and reads the asymmetry:

| | B23 holdout | weak-v2 (B6) holdout |
|---|---|---|
| **B24** (B23 labels) | expected to win | **the informative test** |
| **control** (B6 labels) | — | expected to win |

```text
strength = "uninformative"   each arm wins on its own labeller's surface
strength = "strong"          B24 also wins on B6's own surface
strength = "adverse"         B24 loses on the surface built from its own labels
```

**A `strong` verdict is the best evidence available short of expert truth**, because it means B23 supervision produced a model that reproduces the B6 teacher better than training on B6 labels did — and no self-fulfilling mechanism explains that.

Neither weak surface is expert truth. B15 and B21 both improved on a weak surface and then failed on gold. The evaluator therefore produces evidence, not a decision.

## The decision: one predeclared gold look

```text
comparator   canonical B20 (0.667159355531343)
statistic    paired study bootstrap of the 12-target macro AUC
rule         paired median > 0 AND P(B24 > B20) >= 0.95
```

The threshold is deliberately not a bare point-estimate win. B22 measured a `0.0439` swing across a single run's epochs on this surface, and the reported bootstrap intervals imply a macro standard error near `0.0250`. A small positive difference is not evidence of anything, and requiring 0.95 probability of superiority is what stops B24 being promoted by noise.

`accept_b24` **refuses to run twice** — if `acceptance.json` already exists it raises. Re-running it, or adjusting anything after seeing the result, would destroy the only property that makes it worth running.

A B20 replay-sanity check runs first: if the evaluation path cannot reproduce the canonical B20 score to within `0.005`, the comparison is not trustworthy and the run aborts before any decision is recorded.

## Gates in code, not prose

```text
B23 labeller audit must have passed   -> enforced in freeze_b23_holdout
                                      -> enforced again in train_b24
gold studies out of gradients         -> enforced in build_matched_surface
holdout studies out of gradients      -> enforced in build_matched_surface
weak-v2 manifest is the frozen one    -> enforced by the pinned SHA-256
arms trained on the same studies      -> enforced in b24_eval
arms started from the same encoder    -> enforced in b24_eval
one gold look only                    -> enforced in accept_b24
```

## Running the full pipeline

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

# 1. Label (B23). Resumable; smoke-test with --limit 20 first.
rsna-knee-b23 --train-csv "$DATA_ROOT/train.csv" \
  --out-root runs/b23_llm_report_labels \
  --backend ollama --model qwen3:14b --num-ctx 16384 --max-new-tokens 4096

# 2. Audit the labeller against frozen B6. THE GATE.
rsna-knee-b23-audit --train-csv "$DATA_ROOT/train.csv" \
  --candidate runs/b23_llm_report_labels/structured_labels.csv \
  --baseline  runs/b6_report_labels_v121/structured_labels.csv \
  --out-root  runs/b23_labeller_audit

# 3. Freeze the development split. Refuses unless step 2 passed.
rsna-knee-b23-split --config configs/b23_llm_labels.yaml --data-root "$DATA_ROOT" \
  --b23-root runs/b23_llm_report_labels \
  --labeller-audit runs/b23_labeller_audit/labeller_audit.json \
  --out-root runs/b23_holdout_v1

# 4. Train both arms. Identical apart from the labels.
COMMON="--config configs/b24_supervision.yaml --data-root $DATA_ROOT \
  --b6-root runs/b6_report_labels_v121 \
  --b23-root runs/b23_llm_report_labels \
  --b23-holdout-root runs/b23_holdout_v1 \
  --weak-holdout-root runs/weak_holdout_v2 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_v2_safe_report/report_ssl/b16_v2_report_encoder.pt"

rsna-knee-b24-control $COMMON --out-root runs/b24_supervision/b6_control
rsna-knee-b24         $COMMON --out-root runs/b24_supervision/b23_candidate

# 5. Cross-labeller evidence on both weak surfaces.
rsna-knee-b24-eval --config configs/b24_supervision.yaml --data-root "$DATA_ROOT" \
  --control-checkpoint   runs/b24_supervision/b6_control/b24_b6_control_model.pt \
  --candidate-checkpoint runs/b24_supervision/b23_candidate/b24_b23_candidate_model.pt \
  --b6-root runs/b6_report_labels_v121 --b23-root runs/b23_llm_report_labels \
  --weak-holdout-root runs/weak_holdout_v2 --b23-holdout-root runs/b23_holdout_v1 \
  --series-policy runs/b12_variable_series/audit/series_policy.json

# 6. ONE gold look. Cannot be repeated.
rsna-knee-b24-accept --config configs/b24_supervision.yaml --data-root "$DATA_ROOT" \
  --b20-checkpoint runs/b20_crop_focus/b20_model.pt \
  --b24-checkpoint runs/b24_supervision/b23_candidate/b24_b23_candidate_model.pt
```

Steps 4 and 5 are the GPU cost: two 2-epoch frozen-encoder runs, each roughly 40% of a B20 five-epoch run.

## Explicitly prohibited

```text
running the gold acceptance more than once
adjusting B24 after seeing the gold result
promoting on a point-estimate win without the 0.95 probability
target-wise mixing of B20 and B24 from the gold per-target table
treating either weak surface as expert truth
training B24 on labels whose audit did not pass
changing the recipe so that more than the labels differ between arms
regenerating weak-v2 or the B23 split after seeing a model result
```

## If B24 is not promoted

That is a real result, not a failure. It would mean better report labels do not translate into a better MRI ranking under this recipe — which, combined with B21 and B22, would point away from both the supervision and the model and toward the measurement problem itself. The honest next step in that case is the hidden leaderboard, which remains the only independent signal available.
