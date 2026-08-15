# B26.2 — deterministic evidence gate

> **Status — 2026-08-16:** IMPLEMENTED / NOT YET QUALITY-ACCEPTED / TRAINING BLOCKED. B20 remains the active working model.

## Why B26.2 exists

B26.1 applied a strict second LLM pass to the 631 exact B20-surface B26 proposals. It retained 281 same-polarity candidates:

```text
raw B26 candidates             631
raw positive                   113
raw negated                    518

B26.1 accepted positive         83
B26.1 accepted negated         198
B26.1 accepted total           281
polarity flips rejected          2
```

If used directly, B26.1 would produce:

```text
B6 Synovitis                  399 positive / 17 negative
B26.1 additions               83 positive / 198 negative
final                         482 positive / 215 negative
```

However, the required **fresh 80-case manual audit** (excluding the original B26 manual-review UIDs) did not pass for negative precision:

```text
                    correct    reviewed    observed precision
positive               19         20            95.0%
negated                36         60            60.0%
overall                55         80            68.8%
```

The positive side was strong. The remaining negative errors were systematic: B26.1 still sometimes treated `no effusion`, normal cartilage/bone/ligaments/menisci, no intra-articular injury, normal capsule, or absence of regional inflammation as if those statements directly negated Synovitis.

Therefore **B26.1 is not approved for training**.

## B26.2 policy

B26.2 is a deterministic, precision-first whitelist over the frozen B26.1 output. It does not call an LLM and does not inspect model predictions, weak-v2, or gold outcomes.

It can only **remove** B26.1 proposals:

```text
B26.1 accepted candidate
        |
        +-- positive -> retain only explicit Synovitis or qualifying synovial abnormality
        |
        +-- negated  -> retain only explicit target-specific negation
        |              OR a vetted whole-exam global-normal conclusion
        |
        `-- otherwise -> unmentioned / no added supervision
```

It never:

```text
creates a label
flips polarity
changes an existing B6 cell
drops an existing B6 cell
reads weak-v2 outcomes
reads expert-gold outcomes
```

### Positive whitelist

Accepted only when the B26.1 quoted evidence is verbatim in the original report and contains a direct qualifying concept such as:

```text
synovitis / sinovitis / sinovit
Reizsynovialitis
synovial thickening
synovial hypertrophy
synovial proliferation
pannus
validated multilingual equivalents observed in the reviewed corpus
```

Generic synovial findings such as a small synovial recess/outpouching or synovial-fluid leakage are not sufficient.

### Negative whitelist

Accepted when either:

1. the quoted evidence is verbatim and explicitly negates Synovitis or a qualifying synovial abnormality, for example `no synovitis`, `Synovialis nicht verdickt`, `keine Verdickung der Synovia`, `synovium unremarkable`, `no synovial thickening`; or
2. the original report contains a vetted unqualified whole-exam conclusion such as `Conclusion: Normal`, `Normal study`, `No significant abnormality identified`, `Normal MR examination of the ... knee`, `Geen afwijkingen aangetoond`, and equivalent frozen forms.

The following remain insufficient by themselves:

```text
no / trace effusion
normal bone marrow
normal meniscus / ligament
normal cartilage
normal surrounding soft tissue
no intra-articular body
no intra-articular injury
normal / non-thickened capsule
no perimeniscal inflammation
```

## Implementation

```text
developments/src/rsna_knee/b26_2_deterministic_gate.py
developments/tests/test_b26_2_deterministic_gate.py
```

The run records SHA-256 hashes of the B26.1 candidate file and balance-audit input.

## Required next quality gate

B26.2 must be run on the completed B26.1 candidate file, then a new manual review must be performed on its accepted calls. The review set excludes both previous audited UID sets when those files are supplied.

Training remains explicitly blocked until that fresh B26.2 manual review is complete.

No weak-v2 or reused-gold result should be inspected before the label-quality decision is frozen.
