# Evaluation Dataset

The 6,000-image public package is hosted at
[JinyuLiu/MUMU-Eval-6000](https://huggingface.co/datasets/JinyuLiu/MUMU-Eval-6000)
and mirrored at
[hf-mirror.com](https://hf-mirror.com/datasets/JinyuLiu/MUMU-Eval-6000).
No dataset files are stored in this GitHub repository.
The verified upload revision is `1dbd1e38f77f5e34b080746db4439537bd772ca0`.

## Public Package

| Split | Images | Task A | Task B | Task C | Ground truth |
| --- | ---: | ---: | ---: | ---: | --- |
| Validation | 1,000 | 334 | 333 | 333 | Included |
| Public test | 5,000 | 1,667 | 1,667 | 1,666 | Included |
| Total | 6,000 | 2,001 | 2,000 | 1,999 | — |

All 6,000 images have distinct SHA-256 values. The deterministic sampling seed
is `20260817`. Both manifests contain IDs, task, source dataset, relative image
path, dimensions, format, SHA-256, source ID, source split, prompt, and
source-data ground truth.

```bash
HF_ENDPOINT=https://hf-mirror.com \
  hf download JinyuLiu/MUMU-Eval-6000 \
  --repo-type dataset --local-dir ./mumu_eval_v1
```

## Composition

| Task | Dataset | Validation | Public test | Source used to prepare the split |
| --- | --- | ---: | ---: | --- |
| A | KonIQ-10k | 67 | 334 | [HF collection](https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets) / [official page](http://database.mmsp-kn.de/koniq-10k-database.html) |
| A | SPAQ | 67 | 334 | [HF collection](https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets) / [official GitHub](https://github.com/h4nwei/SPAQ) |
| A | LIVE Challenge | 67 | 333 | [HF collection](https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets) / [official page](https://live.ece.utexas.edu/research/ChallengeDB/index.html) |
| A | Places365 | 67 | 333 | [HF](https://huggingface.co/datasets/ljnlonoljpiljm/places365-256px) / [official GitHub](https://github.com/CSAILVision/places365) |
| A | NUS-WIDE | 66 | 333 | [labels](https://huggingface.co/datasets/lxyhaha/NUS-WIDE) / [images](https://huggingface.co/datasets/moneyzz432/nus_wide) / [official page](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) |
| B | Objects365 | 111 | 556 | [annotations](https://huggingface.co/datasets/jxu124/objects365) / [images](https://huggingface.co/datasets/guozonghao96/objects365) / [official GitHub](https://github.com/sshao0516/Objects365) |
| B | LVIS | 111 | 556 | [HF](https://huggingface.co/datasets/winvoker/lvis) / [official GitHub](https://github.com/lvis-dataset/lvis-api) |
| B | Visual Genome | 111 | 555 | [HF](https://huggingface.co/datasets/ranjaykrishna/visual_genome) / [official site](https://visualgenome.org/) |
| C | COCO Captions | 111 | 556 | [HF](https://huggingface.co/datasets/Multimodal-Fatima/COCO_captions_validation) / [official site](https://cocodataset.org/) |
| C | Flickr30k | 111 | 555 | [HF](https://huggingface.co/datasets/noonamkha/flickr30k-karpathy) / [project page](https://shannon.cs.illinois.edu/DenotationGraph/) |
| C | Conceptual Captions | 111 | 555 | [HF metadata](https://huggingface.co/datasets/google-research-datasets/conceptual_captions) / [official GitHub](https://github.com/google-research-datasets/conceptual-captions) |

The mirror form of any Hugging Face URL is obtained by replacing
`https://huggingface.co` with `https://hf-mirror.com`.

## Rebuilding

`scripts/build_mumu_eval.py` reconstructs the same allocation from the source
layout documented in the original `datasets/README.md`. It writes public
manifests, internal annotations, and image files. Conceptual Captions provides
URLs rather than hosted image bytes; `scripts/download_conceptual_captions_sample.py`
performs deterministic, validated downloads before the builder runs.

```bash
python scripts/build_mumu_eval.py \
  --datasets-root /path/to/source-datasets \
  --output /path/to/mumu_eval_v1
```

This is a multi-source research split. Original dataset and image licenses
remain in force; users are responsible for complying with every source's terms.
