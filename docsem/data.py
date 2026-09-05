"""Shared data loading helpers: tasks, labels, pdf paths, blocks caches."""
from __future__ import annotations

import json
from pathlib import Path

from .config import BLOCKS_DIR, PAGES_DIR, TRAIN_LABELS, TRAIN_TASKS, VAL_TASKS


def load_tasks(tasks_path: str | Path) -> list[dict]:
    with open(tasks_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(labels_path: str | Path = TRAIN_LABELS) -> dict[str, dict]:
    with open(labels_path, encoding="utf-8") as f:
        return {row["instance_id"]: row for row in (json.loads(line) for line in f if line.strip())}


def tasks_path_for(split: str, data_dir: Path) -> Path:
    return data_dir / split / "tasks.jsonl"


def pdf_path_for(task: dict, data_dir: Path, split: str) -> Path:
    rel = task["document_pdf"]
    p = data_dir / rel
    if not p.exists():
        # original gsm-sem layout: "documents/task_000001.pdf" relative to split dir
        p = data_dir / split / "documents" / rel.rsplit("/", 1)[-1]
    return p


def page_dir_for(task_id: str, split: str, data_dir: Path) -> Path:
    return data_dir / "pages" / split / task_id


def blocks_path_for(task_id: str, split: str, data_dir: Path) -> Path:
    return data_dir / "blocks" / split / f"{task_id}.json"


def blocks_exist(task_id: str, split: str, data_dir: Path) -> bool:
    return blocks_path_for(task_id, split, data_dir).exists()


def split_tasks(split: str, data_dir: Path, limit: int = 0) -> list[dict]:
    tasks = load_tasks(tasks_path_for(split, data_dir))
    if limit:
        tasks = tasks[:limit]
    return tasks


def val_tasks(data_dir: Path, limit: int = 0) -> list[dict]:
    return split_tasks("val", data_dir, limit)


def train_tasks(data_dir: Path, limit: int = 0) -> list[dict]:
    return split_tasks("train", data_dir, limit)