# Dataset contract audit — Phase 2 result

> **Descriptive data audit only.** This result does not train or select a model, alter B6 supervision, or use PV1/PV2 target-wise outcomes for architecture design.

## Exact release and scan coverage

The Phase-2 package was generated from the same frozen local release as Phase 1:

```text
train.csv SHA256
8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33

train_series.csv SHA256
573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3
```

All listed series were found on disk:

```text
listed/scanned series          24,371
missing series directories          0
zero-DICOM series                  0
```

The scan therefore covers the complete training series table rather than a sample.

## Physical DICOM slice-count distribution

```text
series                         24,371
mean slices/series             33.61
minimum                            11
1st percentile                     15
5th percentile                     18
25th percentile                    25
median                             30
75th percentile                    34
90th percentile                    39
95th percentile                    45
97th percentile                    88
98th percentile                   144
99th percentile                   160
maximum                           320
```

The distribution has a clear long upper tail. Exact threshold counts are:

| Slice-count threshold | Series above threshold | Fraction of series |
|---:|---:|---:|
| 32 | 7,332 | 30.08% |
| 45 | 1,157 | 4.75% |
| 48 | 992 | 4.07% |
| 64 | 790 | 3.24% |
| 78 | 763 | 3.13% |
| 100 | 709 | 2.91% |
| 160 | 183 | 0.75% |
| 200 | 88 | 0.36% |

The 763 series above 78 slices are only 3.13% of series but contain 124,802 of the 819,078 listed DICOM slices, or 15.24% of all training slices. The 709 series above 100 slices alone contain 14.65% of all slices.

The tail is not smooth. Common large counts include 120, 128, 144, 160, 186 and 320 slices; 85 series contain exactly 320 DICOM files. This pattern should be characterized by plane/protocol/header metadata before interpreting the long tail as ordinary through-plane sampling.

## What the current 16-position 2.5D sampler actually sees

The current preprocessing does not use only 16 individual images. It places 16 distributed centers through a series and constructs a 3-channel triplet around each center. At ordinary evaluation (`gap=1`), one view can therefore contain up to 48 distinct source slice indices.

The frozen evaluation policy additionally uses center-offset TTA `[-1,0,+1]`. Across those three views, the union can contain up to 78 distinct source slice indices.

For the exact deterministic center policy currently implemented:

```text
one ordinary evaluation view
  full source-slice coverage for every series with <= 48 slices

three-view TTA [-1,0,+1]
  full source-slice coverage for every series with <= 78 slices
```

On this release:

```text
series fully covered by one eval view     95.93%
series fully covered by frozen TTA         96.87%
```

Averaging the fraction of source slices represented within each series gives:

```text
mean per-series coverage, one view        97.78%
mean per-series coverage, frozen TTA      98.52%
median per-series coverage                100% for both
```

Because the small number of very long series contains a disproportionate number of images, slice-weighted coverage is lower:

```text
unique source-slice indices represented / all listed slices
one view                                  89.04%
frozen TTA                               92.03%
```

Among only the 763 series longer than 78 slices, the mean TTA source-slice coverage is about 52.75%, the median is about 54.17%, and the minimum is 24.38% for 320-slice series. A single ordinary view represents only 15% of a 320-slice series; the three-view TTA union represents 78/320 = 24.38%.

These percentages describe distinct source slice indices touched by preprocessing. They are not a claim that every slice carries independent diagnostic information, nor do they measure feature-space information retention.

## Interpretation

### No immediate reason to replace the 16-position sampler globally

For the overwhelming majority of training series, the present 16-center 2.5D design is much less sparse than the phrase “16 slices” suggests. The triplet construction plus distributed centers gives complete source-slice coverage through 48 slices in one view, and the frozen three-view TTA gives complete coverage through 78 slices. Therefore a global increase in the number of centers would add substantial runtime for relatively little additional source coverage on roughly 97% of series.

The Phase-2 result is therefore **NO-GO for an unconditional global increase in sampled positions based on slice count alone**.

### The long tail remains a real data question

The 3.13% of series above 78 slices are different: they contain 15.24% of all DICOM files and can be heavily subsampled. Before proposing an adaptive sampler, we need to know what these large series actually are. In particular, the next audit should determine whether the tail clusters by anatomical plane, acquisition metadata, matrix size, field strength, manufacturer/model, slice thickness, pixel spacing, transfer syntax, or multi-frame/volumetric acquisition characteristics.

A future adaptive sampling experiment would be justified only if the long-tail series correspond to clinically meaningful acquisitions for which the current distributed triplets systematically omit substantial anatomical information. It should not be defined merely because `N > 78`.

## Training-time caveat

Training uses the same 16 centers but may choose gap 1 or 2 and adds center jitter. Repeated epochs can therefore expose a long series to more source indices than one deterministic evaluation view. The Phase-2 calculation above intentionally evaluates the frozen deterministic inference contract and does not estimate the stochastic union seen across training epochs.

## Decision after Phase 2

```text
Global increase from 16 positions              NO-GO from slice-count evidence alone
Keep current 16-position policy                YES while data audit continues
Investigate >78-slice series structurally      GO
Define B35 from this result                    NO-GO
Change B6 from this result                     NO-GO
```

The next data step is a representative-header audit over all 24,371 training series, with explicit stratification of the >78-, >100- and >200-slice tails. This remains descriptive and precedes any new modelling experiment.
