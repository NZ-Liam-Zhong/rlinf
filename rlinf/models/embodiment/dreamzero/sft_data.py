"""DreamZero SFT data utilities for LIBERO.

Provides DreamZeroLiberoDataset and DreamZeroCollator that convert
LeRobot v3 LIBERO data into the batch format expected by VLA.forward().
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

LIBERO_PROMPT_TEMPLATE = (
    "As a robot, perform the manipulation task: {task}. "
    "Observe the scene from both a third-person and wrist camera view."
)


class DreamZeroLiberoDataset(Dataset):
    """Map LeRobot v3 LIBERO samples to DreamZero training inputs."""

    VIDEO_CHUNK_OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14]
    VIDEO_CHUNK_STRIDE = 16
    ACTION_CHUNK_SIZE = 16

    def __init__(
        self,
        data_path: str | list[str],
        action_horizon: int = 64,
        num_chunks: int = 4,
        max_action_dim: int = 32,
        max_state_dim: int = 64,
    ):
        import lerobot.datasets.lerobot_dataset as lerobot_dataset

        if isinstance(data_path, (list, tuple)):
            if len(data_path) == 0:
                raise ValueError("DreamZeroLiberoDataset requires at least one data path.")
            data_path = data_path[0]
        self.data_path = str(data_path)
        self.num_chunks = num_chunks
        self.action_horizon = action_horizon
        self.video_frames_per_chunk = len(self.VIDEO_CHUNK_OFFSETS)
        self.video_horizon = self.video_frames_per_chunk * num_chunks + 1
        self.state_horizon = num_chunks
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.embodiment_id = 17

        metadata = lerobot_dataset.LeRobotDatasetMetadata(self.data_path)
        self._fps = metadata.fps
        self._tasks = self._load_task_texts(Path(self.data_path) / "meta")

        video_steps: list[int] = []
        for c in range(num_chunks):
            base = c * self.VIDEO_CHUNK_STRIDE
            video_steps.extend(base + o for o in self.VIDEO_CHUNK_OFFSETS)
        video_steps.append(video_steps[-1] + 2)

        state_steps = [c * self.VIDEO_CHUNK_STRIDE for c in range(num_chunks)]
        action_steps = list(range(action_horizon))

        delta_timestamps = {
            "observation.images.image": [t / self._fps for t in video_steps],
            "observation.images.wrist_image": [t / self._fps for t in video_steps],
            "observation.state": [t / self._fps for t in state_steps],
            "action": [t / self._fps for t in action_steps],
        }
        self.dataset = lerobot_dataset.LeRobotDataset(
            self.data_path,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
        )

    @staticmethod
    def _load_task_texts(meta_dir: Path) -> dict[int, str]:
        import pandas as pd

        task_map: dict[int, str] = {}

        tasks_jsonl = meta_dir / "tasks.jsonl"
        if tasks_jsonl.exists():
            with open(tasks_jsonl, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    task_id = int(entry.get("task_index", 0))
                    task_text = str(entry.get("task", ""))
                    task_map[task_id] = task_text
            if task_map:
                return task_map

        task_path = meta_dir / "tasks.parquet"
        if not task_path.exists():
            return {}

        tasks_df = pd.read_parquet(task_path)

        if (
            list(tasks_df.columns) == ["task_index"]
            and tasks_df.index.dtype.kind in ("U", "O", "S")
        ):
            for text, row in tasks_df.iterrows():
                task_map[int(row["task_index"])] = str(text)
            return task_map

        text_col = None
        for candidate in ("task", "task_text", "language", "instruction", "prompt"):
            if candidate in tasks_df.columns:
                text_col = candidate
                break
        if text_col is None:
            cols = [c for c in tasks_df.columns if c != "task_index"]
            text_col = cols[0] if cols else None

        for _, row in tasks_df.iterrows():
            task_id = int(row.get("task_index", 0))
            if text_col is None:
                task_text = ""
            else:
                value = row.get(text_col, "")
                task_text = "" if value is None else str(value)
            task_map[task_id] = task_text
        return task_map

    @staticmethod
    def _to_hwc_uint8(image: Any) -> np.ndarray:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    def _build_video_grid(self, main_frames: np.ndarray, wrist_frames: np.ndarray) -> np.ndarray:
        """Horizontally concatenate main and wrist views: (T, 256, 256, 3) × 2 → (T, 256, 512, 3)."""
        images = []
        for idx in range(main_frames.shape[0]):
            main = self._to_hwc_uint8(main_frames[idx])
            wrist = self._to_hwc_uint8(wrist_frames[idx])
            merged = np.concatenate([main, wrist], axis=1)
            images.append(merged)
        return np.stack(images, axis=0)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]

        main_frames = np.asarray(sample["observation.images.image"])
        wrist_frames = np.asarray(sample["observation.images.wrist_image"])
        if main_frames.ndim == 3:
            main_frames = main_frames[None, ...]
        if wrist_frames.ndim == 3:
            wrist_frames = wrist_frames[None, ...]
        images = self._build_video_grid(main_frames, wrist_frames).astype(np.uint8)

        state = np.asarray(sample["observation.state"], dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        state = state[: self.state_horizon]
        if state.shape[0] < self.state_horizon:
            pad = np.zeros((self.state_horizon - state.shape[0], state.shape[1]), dtype=np.float32)
            state = np.concatenate([state, pad], axis=0)
        state_pad = np.zeros((self.state_horizon, self.max_state_dim), dtype=np.float32)
        state_dim = min(state.shape[-1], self.max_state_dim)
        state_pad[:, :state_dim] = state[:, :state_dim]
        state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
        state_mask[:, :state_dim] = True

        action = np.asarray(sample["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if action.shape[0] < self.action_horizon:
            pad = np.zeros((self.action_horizon - action.shape[0], action.shape[1]), dtype=np.float32)
            action = np.concatenate([action, pad], axis=0)
        action = action[: self.action_horizon]
        action = np.clip(action, -1.0, 1.0)
        action_pad = np.zeros((self.action_horizon, self.max_action_dim), dtype=np.float32)
        action_dim = min(action.shape[-1], self.max_action_dim)
        action_pad[:, :action_dim] = action[:, :action_dim]
        action_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
        action_mask[:, :action_dim] = True

        prompt = sample.get("task")
        if prompt is None:
            task_idx = int(sample.get("task_index", 0))
            prompt = self._tasks.get(task_idx, "")
        prompt = str(prompt)

        return {
            "images": images,
            "state": state_pad,
            "state_mask": state_mask,
            "action": action_pad,
            "action_mask": action_mask,
            "embodiment_id": np.int64(self.embodiment_id),
            "has_real_action": np.bool_(True),
            "has_lapa_action": np.bool_(False),
            "is_cotrain_instance": np.bool_(False),
            "segmentation_target": np.zeros((2,), dtype=np.float32),
            "segmentation_target_mask": np.zeros((1,), dtype=np.float32),
            "lapa_action": np.zeros_like(action_pad),
            "lapa_action_mask": np.zeros_like(action_mask),
            "text": prompt,
        }


class DreamZeroCollator:
    """Collate DreamZero samples: stack tensors and tokenize text."""

    def __init__(self, tokenizer_path: str, max_seq_len: int):
        from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer

        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=max_seq_len,
            clean="whitespace",
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        for key in [
            "images", "state", "state_mask", "action", "action_mask",
            "embodiment_id", "has_real_action", "has_lapa_action",
            "is_cotrain_instance", "segmentation_target",
            "segmentation_target_mask", "lapa_action", "lapa_action_mask",
        ]:
            values = [f[key] for f in features]
            batch[key] = torch.as_tensor(np.stack(values, axis=0))

        raw_texts = [str(f["text"]) for f in features]
        text_values = [LIBERO_PROMPT_TEMPLATE.format(task=t) for t in raw_texts]
        text_ids, text_mask = self.tokenizer(
            text_values, return_mask=True, add_special_tokens=True
        )
        batch["text"] = torch.as_tensor(text_ids)
        batch["text_attention_mask"] = torch.as_tensor(text_mask)
        return batch


def build_dreamzero_sft_dataloader(
    cfg,
    world_size: int,
    rank: int,
    data_paths: str,
    eval_dataset: bool = False,
):
    """Build DreamZero SFT dataloader -- callable from FSDPVlaSftWorker."""
    model_cfg = cfg.actor.model
    tokenizer_path = model_cfg.get("tokenizer_path", "google/umt5-xxl")
    max_seq_len = int(model_cfg.get("max_seq_len", 512))
    action_chunk_size = int(model_cfg.get("dreamzero_action_horizon", 16))
    num_chunks = int(model_cfg.get("dreamzero_num_chunks", 4))
    effective_action_horizon = action_chunk_size * num_chunks
    max_action_dim = int(model_cfg.get("dreamzero_max_action_dim", 32))
    max_state_dim = int(model_cfg.get("dreamzero_max_state_dim", 64))

    dataset = DreamZeroLiberoDataset(
        data_path=data_paths,
        action_horizon=effective_action_horizon,
        num_chunks=num_chunks,
        max_action_dim=max_action_dim,
        max_state_dim=max_state_dim,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        drop_last=not eval_dataset,
    )
    num_workers = int(cfg.actor.get("dataloader_num_workers", 4))
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=not eval_dataset,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=DreamZeroCollator(
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
        ),
    )
    return data_loader, {"num_samples": len(dataset)}
