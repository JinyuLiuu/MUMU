# Complete Results

All numbers are source-data proxy metrics. `B` is label-aware mAP@[0.5:0.95],
`C = 0.4 CIDEr + 0.3 SPICE + 0.3 CLIPScore`, and raw score is
`0.3 A + 0.4 B + 0.3 C`. The MUMU efficiency multiplier is omitted because its
`alpha` and `beta` values are unspecified.

## Validation

| Model | A Macro-F1 | B mAP | B class-agnostic mAP | CIDEr | SPICE | CLIPScore | C composite | Raw score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Florence-2-base | 0.1947 | 0.0314 | 0.0492 | 1.0491 | 0.1951 | 0.7854 | 0.7138 | 0.2851 |
| Florence-2-base-ft | 0.1589 | 0.0198 | 0.0389 | 1.0431 | 0.1931 | 0.7599 | 0.7031 | 0.2665 |
| Florence-2-large | 0.1998 | **0.0357** | **0.0553** | 1.1149 | 0.2023 | 0.7900 | 0.7436 | **0.2973** |
| Florence-2-large-ft | 0.1592 | 0.0189 | 0.0398 | **1.1522** | **0.2047** | 0.7772 | **0.7555** | 0.2820 |
| LFM2.5-VL-450M | **0.4459** | 0.0037 | 0.0043 | 0.6099 | 0.1537 | **0.7949** | 0.5285 | 0.2938 |

## Public Test

| Model | A Macro-F1 | B mAP | B class-agnostic mAP | CIDEr | SPICE | CLIPScore | C composite | Raw score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Florence-2-base | 0.1555 | 0.0111 | 0.0488 | 0.9406 | 0.1856 | 0.7793 | 0.6657 | 0.2508 |
| Florence-2-base-ft | 0.1195 | 0.0061 | 0.0396 | 0.9178 | 0.1824 | 0.7568 | 0.6489 | 0.2329 |
| Florence-2-large | 0.1672 | **0.0138** | **0.0541** | 0.9563 | 0.1912 | 0.7839 | 0.6751 | 0.2582 |
| Florence-2-large-ft | 0.1494 | 0.0063 | 0.0409 | **0.9865** | **0.1944** | 0.7703 | **0.6840** | 0.2526 |
| LFM2.5-VL-450M | **0.3795** | 0.0016 | 0.0073 | 0.5912 | 0.1596 | **0.7959** | 0.5232 | **0.2715** |

## Runtime and Integrity

| Model | Parameters | Dtype | Total inference time | Peak allocated GPU memory | Rows | Failures |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Florence-2-base | 231,414,016 | FP32 | 5,053.1 s | 1.18 GiB | 6,000 | 0 |
| Florence-2-base-ft | 231,414,016 | FP32 | 3,392.3 s | 1.19 GiB | 6,000 | 0 |
| Florence-2-large | 776,465,408 | FP32 | 5,733.2 s | 3.56 GiB | 6,000 | 0 |
| Florence-2-large-ft | 770,173,952 | FP32 | 4,080.9 s | 3.53 GiB | 6,000 | 0 |
| LFM2.5-VL-450M | 448,718,848 | BF16 | not retained as one uninterrupted run | 1.07 GiB | 6,000 | 0 |

For LFM2.5, Task B generation took 1,205.5 seconds and the selected Task C run
took 345.8 seconds. Task A completed before a parser-driven restart, so an exact
full-run time is not claimed.

Each selected prediction file contains exactly 6,000 unique IDs with task counts
`A/B/C = 2,001/2,000/1,999`. Florence-2 predictions retain the official parser's
four zero-width boxes and dense repeated detections because no NMS or box
filtering was requested. LFM2.5 produced 1,805 valid boxes on 1,796 of 2,000
Task B images, showing that low detection recall is its main weakness.

## LFM2.5 Caption Prompt Comparison

| Prompt | Split | CIDEr | SPICE | CLIPScore | C composite | Raw score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Initial verbose caption | Validation | 0.4429 | 0.1885 | 0.8139 | 0.4779 | 0.2786 |
| Selected 8–12 word COCO prompt | Validation | 0.6099 | 0.1537 | 0.7949 | **0.5285** | **0.2938** |
| Initial verbose caption | Public test | 0.4500 | 0.1872 | 0.8147 | 0.4805 | 0.2587 |
| Selected 8–12 word COCO prompt | Public test | 0.5912 | 0.1596 | 0.7959 | **0.5232** | **0.2715** |

Full-precision values and per-dataset breakdowns are in `results/metrics/`.
Compressed raw predictions are in `results/predictions/`. The selected LFM2.5
file combines Task A/B from the initial full run with Task C from the COCO-style
prompt run; this composition is recorded in its summary JSON.

## Interpretation

LFM2.5 more than doubles Florence-2-base's lexical Task A proxy and achieves the
best public-test raw score among these five checkpoints. Florence-2 remains much
stronger on Tasks B and C. Task A is not an equal prompting comparison:
LFM2.5 receives the explicit source-data vocabulary, whereas Florence-2 labels
are extracted lexically from detailed captions. The two Florence-2 large models
are also outside the 0.5B parameter limit. These results should therefore guide
model selection, not be presented as an official challenge leaderboard.
