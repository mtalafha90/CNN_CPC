# Where to go after B51 — a decision built from the whole archive

**Written:** 2026-08-30, after reading the 137 experiment documents, the audit
chain, and the measurements taken today. Nothing here is a new result. It is a
reading of existing evidence and a ranked decision.

## The one number that reframes the project

`POST_B45_PLATEAU_RETROSPECTIVE.md` records the closest published comparator:

> Qiu et al. 2024, *Learning co-plane attention across MRI sequences for
> diagnosing twelve types of knee abnormalities*, Nature Communications 15,
> 7637. Overall AUC **0.812** on 1,748 subjects. **Its AUC falls to roughly
> `0.721-0.726` on external datasets with different sequence settings.**

Same twelve findings. Same modality. A purpose-built co-plane attention
architecture. And it loses about **0.09** when the acquisition changes.

This project's hidden score is **0.714**.

`RAISING_AUC_TO_080.md` set its target from "published work ... roughly
0.73-0.81", flagging that the range "carries no citation and should be treated
as a working impression". The citation was found later, and it splits: `0.812`
is the *in-domain* figure, `0.721-0.726` the *external* one. The 0.80 target was
therefore set against a number measured under conditions this competition may
not offer.

If the hidden test is genuinely cross-site — the challenge is built from 19
sites and about a dozen languages — then **0.714 is already at the published
external-generalisation level for this task**, and the remaining headroom is
around `+0.01`, not `+0.09`.

That single fact changes what counts as a good decision.

## Step 0 — the cheapest and most decision-relevant action, and it is not mine

**Look at the public leaderboard.** Two minutes, and it settles the strategy.

```text
top scores near 0.75-0.80   the external reading holds; 0.714 is close to the
                            practical ceiling; pursue +0.01 consolidations and
                            stop funding large architecture bets

top scores near 0.90+       the external reading is wrong, the hidden test is
                            closer to in-domain, and something structural is
                            missing that incremental work will never reach
```

`EXPERIMENT_STATUS` carries an unverified `0.924` claim. It has never been
checked, and it is the difference between "nearly done" and "wrong approach".
Everything below assumes the first branch; if the leaderboard says otherwise,
this document should be rewritten rather than followed.

## What the archive has already closed

Eight consecutive architecture experiments returned no support:

| | Mechanism | Result |
|---|---|---|
| B38 | global-only 448 tail | `0.66441` vs base `0.66875` |
| B40 | one more epoch | loss fell, expert AUC fell |
| B41/B42 | in-plane geometry | no visible hidden separation |
| B44 | 32 → 64 centres | weak targets unmoved |
| B45 | plane router | `−0.003944` vs B42 |
| B46 | gold anchoring | `−0.004946`, `P=0.13` |
| B48 | global conditioning | ceiling `0.0015`, underpowered |
| B49 | native tiling | ceiling `0.0024`, hidden `0.707` |

B50 was the first powered positive since B37 (`+0.011221`, 12/12 targets). B51
carried it to the full population and lost `−0.011785` against B42 on expert
truth with 3/12 targets improved. **B50's gain was measured against
report-derived labels; B51's loss against expert truth.** That is the third
instance of the pattern B40 first recorded.

## What today's measurements closed

Four cheap measurements, no GPU, all negative — which is progress by
elimination:

| Question | Answer |
|---|---|
| Did the LLM fix label quality? | **No.** precision `0.6647` vs B6's `0.6736` |
| Is the merged teacher better? | **No.** precision `0.6527`, coverage `0.6580` |
| Do bad per-target labels explain ACL/MCL? | **No.** ACL is among the best labelled (precision `0.774`) |
| Can LLM self-confidence filter errors? | **No.** AUC `0.576`; costs 40% coverage for `+0.038` |

What survives is one aggregate fact: the teacher has sensitivity `0.977` against
specificity `0.499`. It over-calls positives systematically, and about a third of
its positives disagree with the expert.

But note what this does **not** license. `RAISING_AUC_TO_080.md` Tier 1 —
"train on LLM labels and evaluate honestly" — **was already executed**. B42
trains through `load_fill_merged_export` on the `llm_fill` arm, and it scored
`0.714`. The teacher upgrade has happened. It did not produce `0.80`.

**Stop spending effort on the teacher.** Four measurements and one completed
hidden submission now point the same way.

## What is genuinely untested

Three items from `RAISING_AUC_TO_080.md` Tier 0 were never done, and they are the
cheapest unexplored surface in the repository.

### 1. Slice coverage on the long-series tail — CPU, minutes

`tools/slice_coverage.py` exists and has never been run. Phases 2 and 3 both
recorded `Investigate >78-slice series structurally: GO` and it was never
actioned.

The stake, in the tool's own words: *"a structure four slices thick can fall
entirely between two samples."* ACL and MCL are exactly such structures, and they
are exactly the two weakest targets — **both below chance on expert truth for
essentially the whole lineage**. Today's audit removed the label explanation for
ACL, which makes the sampling explanation more, not less, likely.

If a cruciate ligament is not in the sampled slices, no architecture and no
teacher can fix it. This measures whether the model is being asked about
something it was never shown.

### 2. The cosine that never anneals — one run, same compute

```text
scheduler   CosineAnnealingLR(T_max=5)
epochs run  2
epoch 1     1.000e-04   100.0% of peak
epoch 2     9.055e-05    90.5% of peak
STOP
```

Training halts having never once trained at a meaningfully reduced learning
rate. `T_max=2` costs the identical 90 minutes and anneals properly. B22 tested
epoch *counts* and found E2 best; it never tested whether E2 with a completed
schedule differs from E2 without one.

This is the rarest thing in the archive: an untested change that costs nothing.

### 3. Real test-time augmentation — no training at all

Current TTA is three views offset by `±1` slice on a comb of median stride
`1.9`. The three views are nearly the same slices; it is close to a no-op. B39's
five offsets `[-2,-1,0,1,2]` broaden it slightly and were never scored.

**One warning the archive does not state.** `RAISING_AUC_TO_080.md` suggests
horizontal flip. **Do not use it.** Four of the twelve targets are laterality-
specific — Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA — and a
horizontal flip of a coronal knee exchanges medial with lateral. On sagittal
series it exchanges anterior with posterior, which bears directly on the
cruciates. Flip TTA would corrupt exactly the targets that are already weakest.

Multi-scale and multi-crop are safe. Flip is not.

## The decision

```text
0. Read the leaderboard.                      2 minutes. Gates everything.
1. Run slice_coverage.py.                     CPU, minutes. Tests the ACL/MCL
                                              hypothesis before any GPU.
2. Do not submit B51.                         −0.012 on the only expert surface.
3. T_max=2 rerun of the B42 endpoint.         One run, same compute as before.
4. Only if 1 and 3 justify it: B47.           Scores evidence on the 14x14 the
                                              encoder produced instead of the
                                              6x6 it is pooled to, recovering
                                              5.44x of discarded localisation.
5. Stop: teacher work, gold anchoring,        All closed by direct measurement.
   plane routing, centre counts, crop
   fractions, top-k, extra epochs.
```

### On B51 specifically

Do not submit it. B50's supporting evidence was measured against report labels;
against expert truth B51 is `−0.011785` with 3/12 targets improved and
`P(better) = 0.081`. Expert-58 is too blunt to resolve `±0.011` and has never
predicted a hidden difference — but a submission needs a positive reason, and
after today there is not one. B42 remains the endpoint.

### What would change this decision

- **The leaderboard top is near 0.90.** Then the external-generalisation reading
  is wrong, incremental consolidation is futile, and the project needs a
  different capability rather than a better version of this one.
- **`slice_coverage` shows good coverage of short structures.** Then Ceiling 4
  is a story rather than a mechanism, ACL/MCL weakness is unexplained, and the
  next question is why the images do not support those two findings.
- **`T_max=2` moves the endpoint materially.** Then the fixed-E2 policy,
  inherited unquestioned from B26 through Phase 9, was a real constraint and
  every prior architecture comparison ran on an under-annealed model.

## The honest summary

The project's controls are unusually strict and its bookkeeping is sound. The
plateau is not carelessness. It is that `0.714` may simply be close to what this
task allows across sites, and the archive set its target from an in-domain
number measured elsewhere.

The remaining work worth doing is small, cheap and diagnostic. The expensive
work — another architecture, another teacher — has been tried eight times and
has not moved the hidden score once since B37.
