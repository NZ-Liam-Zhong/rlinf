#!/usr/bin/env python3
"""End-to-end test: LIBERO dataset -> DreamZero model -> loss + backward.

Usage:
  CUDA_VISIBLE_DEVICES=1 python test_dreamzero_sft_pipeline.py

Prints shapes at every stage so you can see the full data flow.
"""

import sys, os, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("PYTHONPATH", os.path.dirname(__file__))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DATA_PATH = "/mnt/public/znz/datasets/libero_10"
MODEL_PATH = "/mnt/public/znz/checkpoints/DreamZero-AgiBot"
TOKENIZER_PATH = "/mnt/public/znz/checkpoints/umt5-xxl"
ACTION_HORIZON = 192
VIDEO_HORIZON = 33
STATE_HORIZON = 4

SEP = "=" * 72


def stage1_raw_dataset():
    """Stage 1: Read one raw sample from LeRobot dataset (before any transform)."""
    print(f"\n{SEP}")
    print("STAGE 1: Raw LeRobot v3 LIBERO sample")
    print(SEP)

    import lerobot.datasets.lerobot_dataset as ld

    metadata = ld.LeRobotDatasetMetadata(DATA_PATH)
    print(f"  fps            = {metadata.fps}")
    print(f"  total_episodes = {metadata.total_episodes}")

    ds = ld.LeRobotDataset(
        DATA_PATH,
        delta_timestamps={
            "observation.images.image": [t / metadata.fps for t in range(VIDEO_HORIZON)],
            "observation.images.wrist_image": [t / metadata.fps for t in range(VIDEO_HORIZON)],
            "observation.state": [8 * t / metadata.fps for t in range(STATE_HORIZON)],
            "action": [t / metadata.fps for t in range(ACTION_HORIZON)],
        },
        video_backend="pyav",
    )
    sample = ds[0]
    print(f"\n  sample keys = {sorted(sample.keys())}")
    for k, v in sorted(sample.items()):
        if hasattr(v, "shape"):
            print(f"  {k:40s}  shape={str(v.shape):20s}  dtype={v.dtype}")
        elif isinstance(v, (int, float, str, bool)):
            print(f"  {k:40s}  value={v!r}")
    return sample


def stage2_dataset_getitem():
    """Stage 2: DreamZeroLiberoDataset.__getitem__ output (after transform)."""
    print(f"\n{SEP}")
    print("STAGE 2: DreamZeroLiberoDataset[0] output")
    print(SEP)

    from rlinf.models.embodiment.dreamzero.sft_data import DreamZeroLiberoDataset

    dataset = DreamZeroLiberoDataset(
        data_path=DATA_PATH,
        action_horizon=ACTION_HORIZON,
        video_horizon=VIDEO_HORIZON,
        max_action_dim=32,
        max_state_dim=64,
    )
    print(f"  dataset length = {len(dataset)}")
    print(f"  task map       = {dataset._tasks}")

    item = dataset[0]
    print(f"\n  __getitem__ output keys:")
    for k, v in sorted(item.items()):
        if isinstance(v, np.ndarray):
            print(f"    {k:35s}  shape={str(v.shape):20s}  dtype={v.dtype}")
        elif isinstance(v, (np.generic,)):
            print(f"    {k:35s}  value={v!r}  dtype={type(v).__name__}")
        else:
            val_repr = repr(v) if len(repr(v)) < 80 else repr(v)[:77] + "..."
            print(f"    {k:35s}  {val_repr}")

    print(f"\n  images min/max       = {item['images'].min()}, {item['images'].max()}")
    print(f"  state non-zero count = {(item['state'] != 0).sum()} / {item['state'].size}")
    print(f"  action non-zero count= {(item['action'] != 0).sum()} / {item['action'].size}")
    print(f"  action_mask true cnt = {item['action_mask'].sum()} / {item['action_mask'].size}")
    print(f"  text                 = {item['text']!r}")
    return dataset


def stage3_collator(dataset):
    """Stage 3: DreamZeroCollator batch output."""
    print(f"\n{SEP}")
    print("STAGE 3: DreamZeroCollator batch (batch_size=1)")
    print(SEP)

    from rlinf.models.embodiment.dreamzero.sft_data import DreamZeroCollator

    collator = DreamZeroCollator(
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=512,
    )
    items = [dataset[0]]
    batch = collator(items)

    print(f"\n  batch keys:")
    for k, v in sorted(batch.items()):
        if isinstance(v, torch.Tensor):
            print(f"    {k:40s}  shape={str(tuple(v.shape)):20s}  dtype={v.dtype}")
        else:
            print(f"    {k:40s}  type={type(v).__name__}")

    print(f"\n  text token sample    = {batch['text'][0, :10].tolist()}")
    print(f"  text_attn_mask sum   = {batch['text_attention_mask'].sum(dim=-1).tolist()}")
    return batch


def stage4_model_forward(batch):
    """Stage 4: Load DreamZeroPolicy and run sft_forward."""
    print(f"\n{SEP}")
    print("STAGE 4: DreamZeroPolicy.sft_forward()")
    print(SEP)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device = {device}")
    if device == "cuda":
        # Debug fallback for environments where cuDNN 3D conv init is unstable.
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    print(f"  Loading model from {MODEL_PATH} ...")
    t0 = time.time()

    from rlinf.models.embodiment.dreamzero.dreamzero_policy import DreamZeroPolicy

    policy = DreamZeroPolicy(
        model_path=MODEL_PATH,
        device=device,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=512,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  policy.model type = {type(policy.model).__name__}")

    batch_gpu = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_gpu[k] = v.to(device)
        else:
            batch_gpu[k] = v

    print(f"\n  batch_gpu keys on device:")
    for k, v in sorted(batch_gpu.items()):
        if isinstance(v, torch.Tensor):
            print(f"    {k:40s}  device={v.device}  shape={tuple(v.shape)}")

    print(f"\n  Running sft_forward ...")
    t0 = time.time()
    if device == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = policy.sft_forward(data=batch_gpu)
    else:
        loss = policy.sft_forward(data=batch_gpu)
    print(f"  sft_forward done in {time.time() - t0:.1f}s")
    print(f"  loss = {loss}")
    print(f"  loss.shape = {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
    print(f"  loss.dtype = {loss.dtype}")
    print(f"  loss.requires_grad = {loss.requires_grad}")

    return loss, policy


def stage5_backward(loss):
    """Stage 5: Backward pass."""
    print(f"\n{SEP}")
    print("STAGE 5: Backward pass")
    print(SEP)

    t0 = time.time()
    loss.backward()
    print(f"  backward done in {time.time() - t0:.1f}s")
    print(f"  SUCCESS - gradient flows end to end")


def main():
    print(f"\n{'#' * 72}")
    print("# DreamZero SFT Pipeline Test on LIBERO")
    print(f"# Data:  {DATA_PATH}")
    print(f"# Model: {MODEL_PATH}")
    print(f"{'#' * 72}")

    sample = stage1_raw_dataset()
    dataset = stage2_dataset_getitem()
    batch = stage3_collator(dataset)
    loss, policy = stage4_model_forward(batch)
    stage5_backward(loss)

    print(f"\n{SEP}")
    print("ALL 5 STAGES PASSED")
    print(SEP)


if __name__ == "__main__":
    main()
