# FastOpd

FastOpd is an online preference distillation framework for FashionIQ composed image retrieval. A BLIP-2 MaxSim student retrieves and reranks candidate images, while a Qwen3-VL teacher provides ranking and counterfactual supervision.

The training pipeline batches teacher requests, uses student-order speculative decoding for ranking, and generates counterfactual instructions through parallel A/B/C branches. The student and teacher share a GPU through vLLM sleep/wake and optimizer-state offloading.

## Layout

- `scripts/`: full-training launcher.
- `blip2_maxsim/train.py`: training entry point.
- `blip2_maxsim/data/`: FashionIQ loading and image paths.
- `blip2_maxsim/model/`: BLIP-2, MaxSim, and TCB models.
- `blip2_maxsim/teacher/`: Qwen worker, ranking, speculative decoding, and counterfactual generation.
- `blip2_maxsim/training/`: training loop, losses, checkpoints, and evaluation.

## Full training

Install the dependencies in `pyproject.toml` in the student environment. The teacher environment needs vLLM, `qwen-vl-utils`, PyTorch, NumPy, and Transformers.

```bash
export FASTOPD_DATASET_ROOT=/path/to/fashioniq
export FASTOPD_BASE_CHECKPOINT=/path/to/warmup_end.pt
export FASTOPD_TRAIN_ANCHOR_DIR=/path/to/train_warmup_end
export FASTOPD_QWEN_MODEL=/path/to/Qwen3-VL-32B-Instruct-FP8
export FASTOPD_QWEN_PYTHON=/path/to/teacher-env/bin/python
export FASTOPD_GPU=0
bash scripts/train_full.sh --execute
```

The launcher trains across `dress`, `shirt`, and `toptee` for one epoch. Set `FASTOPD_OUTPUT_DIR` to choose an output directory; the default is `outputs/full_fashioniq`.
