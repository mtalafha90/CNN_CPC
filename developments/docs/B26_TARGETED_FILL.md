# B26 — targeted supervision fill and manual quality audit

> **Status — 2026-08-16:** raw targeted extraction COMPLETE; exact B20 fill-surface audit COMPLETE; 80-case manual quality audit FAILED for raw negative-label precision. **Do not train B26 v1.0 as extracted. B20 remains the active working model.**

## Purpose

B26 was introduced after the B25X mechanism analysis and the training-label-only balance audit. The design intentionally makes only the narrow intervention supported by those results:

1. the target scope comes from the balance audit (`targets_needing_fill`), not from a hard-coded target name;
2. existing B6 supervision is always preserved;
3. new supervision is allowed only where B6 is silent;
4. the labeller is not instructed to search for negatives or to manufacture class balance.

The full frozen B6 audit selected exactly one target under the declared forward-looking balance rule:

```text
Synovitis
B6 positive       399
B6 negative        17
B6 usable         416
majority share   95.9%
```

## Implementation

Canonical implementation:

```text
developments/src/rsna_knee/b26_targeted_fill.py
```

Initial implementation commit:

```text
7ecf246c509ffbc40a2fc38f8460509ca6631f60
```

The implementation preserves the B24X-Density principle: fill B6-silent cells only and raise if an existing B6 target or weight is changed.

## Full B26 extraction

The full Qwen extraction completed on all 4,407 studies in about 122 minutes.

Provenance:

```text
backend          ollama
model            qwen3:14b
revision/digest  bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8
quantisation     Q4_K_M
decoding         greedy
max_new_tokens   4096
seed             2026
prompt SHA-256   4aaa5e7df804108a4250060607fc9b5cbec6866ab1a88414b7eaed4a1b0b3e00
```

Corpus accounting:

```text
all studies                    4407
expert-gold excluded             58
report-only                    4349
```

Raw Synovitis states on the 4,349 report-only studies:

```text
positive          564
negated           691
uncertain          57
unmentioned      3037
```

The raw extraction therefore showed that Qwen can emit both positive and negated Synovitis states without being instructed to seek negatives. However, raw state counts are not sufficient evidence that the negated calls are semantically valid.

## Exact B20 fill-surface audit

The B26 output was aligned to the exact frozen B20/B6 training surface:

```text
B20/B6 training studies       3120
B6 usable cells              14123
audit-selected targets           1
selected target          Synovitis
```

For Synovitis:

```text
B6 positive                    399
B6 negative                     17
B6 usable                      416

B26 added positive             113
B26 added negative             518
B26 added total                631
B26 definite calls skipped
because B6 was occupied        413

raw B26-fill final positive    512
raw B26-fill final negative    535
raw B26-fill final usable     1047

B6 cells dropped                 0
B6 cells overridden              0
```

Thus the fill mechanism itself worked exactly as intended: all original B6 supervision was preserved and 631 new B6-silent Synovitis cells were proposed.

## 80-case manual quality audit

Before training, a deterministic review sample (`seed=2026`) was drawn from the actual B20-surface additions:

```text
60 newly added negated calls
20 newly added positive calls
```

Each sampled case was reviewed against the original report, B26 state and quoted evidence.

### Positive calls

```text
reviewed                         20
supported                        16
unsupported                       4
observed precision              80.0%
```

The four unsupported positives were dominated by semantic overreach: effusion alone, or synovial-fluid leakage, was treated as evidence of Synovitis even though the report did not establish synovitis or a qualifying synovial abnormality.

### Negated calls

```text
reviewed                         60
accepted                         10
rejected                         50
observed accepted fraction      16.7%
```

Of the ten accepted negations:

```text
explicit target-specific synovium/synovitis negation      1
accepted from an unqualified global-normal conclusion      9
```

The dominant failure mode was unsupported negation. B26 frequently inferred absence of Synovitis from findings that do not negate Synovitis, including:

```text
no/trace joint effusion
normal bone marrow / no bone bruise
normal menisci or ligaments
no intra-articular body
normal surrounding soft tissues
```

These should generally be `unmentioned`, not `negated`, unless the report separately negates Synovitis/synovial abnormality or gives a genuinely global unqualified normal conclusion.

One reviewed case also showed a polarity/conflict failure: the findings contained a negative statement but the impression stated diffuse synovitis. Under the frozen rule that the impression wins, the correct state is positive rather than negated.

## Decision

The attractive raw fill count (`113` positive + `518` negative) **must not be used for training as-is**. The manual review shows that the raw negative count is heavily inflated by an effusion/normal-structure-to-Synovitis-negation shortcut.

Therefore:

```text
B26 v1.0 raw extraction       COMPLETE
B26 exact fill construction   PASSED engineering invariants
B26 manual label-quality gate FAILED for raw negative precision
B26 v1.0 training             NOT ALLOWED
B20 active working model      unchanged
```

The raw cache, targeted labels, prompt hash and provenance should be preserved unchanged as the completed B26-v1 extraction record.

## Next step: B26.1 evidence adjudication

The next development step is a stricter evidence adjudication layer applied only to the 631 proposed B6-silent definite calls. It should not rerun the complete 4,407-report extraction.

The adjudication rule must be frozen before any model evaluation and should require:

```text
positive:
  explicit synovitis OR qualifying synovial abnormality
  (e.g. thickening/hypertrophy/proliferation/pannus)

negated:
  explicit no-synovitis / normal-or-unremarkable synovium
  OR a genuinely unqualified global-normal / no-intra-articular-pathology conclusion

not sufficient for negated:
  no effusion
  trace effusion
  no bone bruise
  normal ligament/meniscus
  no intra-articular body
  normal surrounding soft tissue

conflicts:
  impression/conclusion overrides findings

otherwise:
  unmentioned / do not add supervision
```

B26.1 is explicitly a post-B26 quality-control refinement motivated by the manual audit. It is not retroactively prospective evidence for B26 v1.0 and must not be evaluated on weak-v2 or reused gold until its label-quality gate is completed.

## Local artifacts

The following local artifacts contain the completed run/audit data and are intentionally not committed with report text:

```text
runs/b26_fill/audit.json
runs/b26_fill/targeted_labels.csv
runs/b26_fill/fill_surface_audit.json
runs/b26_fill/synovitis_blinded_review_80.csv
runs/b26_fill/synovitis_blinded_review_80_annotated.csv
```

The manual review CSV contains original report text and remains a local audit artifact rather than repository source material.
