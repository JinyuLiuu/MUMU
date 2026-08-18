# Model Weights

Weights are not stored in this repository. The table records the exact Hugging
Face revisions used for the published runs.

| Model | Parameters | Exact revision | Hugging Face | Mirror |
| --- | ---: | --- | --- | --- |
| Florence-2-base | 231,414,016 | `5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac` | [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base/tree/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac) | [mirror](https://hf-mirror.com/microsoft/Florence-2-base) |
| Florence-2-base-ft | 231,414,016 | `f6c1a25888ffc1d945ee8a1a77ac833c7303d46e` | [microsoft/Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft/tree/f6c1a25888ffc1d945ee8a1a77ac833c7303d46e) | [mirror](https://hf-mirror.com/microsoft/Florence-2-base-ft) |
| Florence-2-large | 776,465,408 | `21a599d414c4d928c9032694c424fb94458e3594` | [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large/tree/21a599d414c4d928c9032694c424fb94458e3594) | [mirror](https://hf-mirror.com/microsoft/Florence-2-large) |
| Florence-2-large-ft | 770,173,952 | `4a12a2b54b7016a48a22037fbd62da90cd566f2a` | [microsoft/Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft/tree/4a12a2b54b7016a48a22037fbd62da90cd566f2a) | [mirror](https://hf-mirror.com/microsoft/Florence-2-large-ft) |
| LFM2.5-VL-450M | 448,718,848 | `fc6221ca597f3315e4f82fc2df606783267b34ba` | [LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/tree/fc6221ca597f3315e4f82fc2df606783267b34ba) | [mirror](https://hf-mirror.com/LiquidAI/LFM2.5-VL-450M) |
| CLIP ViT-B/32 (metric only) | — | current local snapshot | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | [mirror](https://hf-mirror.com/openai/clip-vit-base-patch32) |

The Florence-2 reference code supplied for this evaluation is
[anyantudre/Florence-2-Vision-Language-Model](https://github.com/anyantudre/Florence-2-Vision-Language-Model).
Model loading and post-processing use the upstream Hugging Face remote code in
each pinned checkpoint.

## Download

```bash
export HF_ENDPOINT=https://hf-mirror.com

hf download microsoft/Florence-2-base \
  --revision 5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac \
  --local-dir models/Florence-2-base

hf download LiquidAI/LFM2.5-VL-450M \
  --revision fc6221ca597f3315e4f82fc2df606783267b34ba \
  --local-dir models/LFM2.5-VL-450M

hf download openai/clip-vit-base-patch32 \
  --local-dir models/clip-vit-base-patch32
```

Repeat the Florence command with the other IDs and revisions from the table.
Do not mix the two inference environments: Florence-2 was run with
Transformers 4.41.2, while LFM2.5 was run with Transformers 5.1.0.
