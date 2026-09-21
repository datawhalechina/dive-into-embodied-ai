"""Download RLinf's published SFT/train/eval splits and prepare local LeRobot copies."""

import argparse
import json
import os
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATASET_ID = "RLinf/RECAP-Libero10-Task0-48succ-Data"
DATASET_REVISION = "75b382d2c066bcedd8b030285b45c856913f1497"
TASK = "put both the alphabet soup and the tomato sauce in the basket"
REPOS = {
    "sft": ("RECAP_SFT_REPO_ID", "local/libero10_fewshot_sft"),
    "train": ("RECAP_REPO_ID", "local/libero10_task0_train"),
    "eval": ("RECAP_EVAL_REPO_ID", "local/libero10_task0_eval"),
}
EXPECTED_EPISODES = {"sft": 30, "train": 4096, "eval": 64}
COLUMNS = ("state", "actions", "timestamp", "frame_index", "episode_index", "index", "task_index")


def dataset_root(repo_id):
    from lerobot.common.constants import HF_LEROBOT_HOME

    if len(repo_id.split("/")) != 2 or any(part in ("", ".", "..") for part in repo_id.split("/")):
        raise ValueError("Use a local repo id of the form local/dataset_name")
    return HF_LEROBOT_HOME / repo_id


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def validate_download(raw_dir):
    """Reject Hugging Face's offline fallback to an empty or partial local_dir."""
    for split, expected in EXPECTED_EPISODES.items():
        source = raw_dir / f"libero10_task0_{split}"
        info_path = source / "meta/info.json"
        episodes_path = source / "meta/episodes.jsonl"
        tasks_path = source / "meta/tasks.jsonl"
        if not all(path.is_file() for path in (info_path, episodes_path, tasks_path)):
            raise FileNotFoundError(f"{split}: metadata is incomplete")
        info = json.loads(info_path.read_text())
        rows = read_jsonl(episodes_path)
        if info.get("total_episodes") != expected or len(rows) != expected:
            raise ValueError(f"{split}: expected {expected} episodes, found {len(rows)}")
        for row in rows:
            ep = row["episode_index"]
            fields = {"episode_chunk": ep // info["chunks_size"], "episode_index": ep}
            data_path = source / info["data_path"].format(**fields)
            videos = [source / info["video_path"].format(**fields, video_key=key) for key in ("image", "wrist_image")]
            if not data_path.is_file() or not all(path.is_file() for path in videos):
                raise FileNotFoundError(f"{split}: episode {ep} data or video is incomplete")


def download(raw_dir):
    from huggingface_hub import snapshot_download

    raw_dir.mkdir(parents=True, exist_ok=True)
    marker = raw_dir / "INCOMPLETE"
    marker.touch()
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    try:
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            allow_patterns=[f"libero10_task0_{split}/*" for split in REPOS],
            local_dir=raw_dir,
        )
        validate_download(raw_dir)
    except Exception as error:
        raise RuntimeError(
            f"Dataset download from {endpoint} is incomplete. Keep {raw_dir} and rerun to resume after fixing network access."
        ) from error
    marker.unlink()
    print(f"Downloaded and verified {DATASET_ID}@{DATASET_REVISION} from {endpoint} to {raw_dir}")


def prepare(raw_dir, split, repo_id):
    """Keep the published split; copy tables/metadata and link the large video directory."""
    if (raw_dir / "INCOMPLETE").exists():
        raise ValueError("Finish data download first")
    source = (raw_dir / f"libero10_task0_{split}").resolve()
    info = json.loads((source / "meta/info.json").read_text())
    rows = read_jsonl(source / "meta/episodes.jsonl")
    tasks = read_jsonl(source / "meta/tasks.jsonl")
    task_map = {row["task_index"]: row["task"] for row in tasks}
    if split != "sft" and set(task_map.values()) != {TASK}:
        raise ValueError("The rollout train/eval split must contain only LIBERO-Long Task 0")
    if [row["episode_index"] for row in rows] != list(range(len(rows))) or len(rows) != info["total_episodes"]:
        raise ValueError("Episode metadata must be complete and contiguous")
    root = dataset_root(repo_id)
    if root.exists():
        raise FileExistsError(f"Output exists: {root}. Choose another --repo-id; existing data is never deleted.")
    root.mkdir(parents=True)
    (root / "INCOMPLETE").touch()
    (root / "meta").mkdir()
    stats_rows, total = [], 0
    for row in rows:
        ep, length = row["episode_index"], row["length"]
        relative = info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        table = pq.read_table(source / relative)
        success = row.get("is_success")
        if type(success) is not bool or (split == "sft" and not success):
            raise ValueError(f"Episode {ep}: expected explicit is_success; SFT must contain successes")
        if len(table) != length or not np.array_equal(np.asarray(table["frame_index"]), np.arange(length)):
            raise ValueError(f"Episode {ep}: incomplete or unordered frames")
        if not np.all(np.asarray(table["episode_index"]) == ep):
            raise ValueError(f"Episode {ep}: mismatched episode indices")
        actual_tasks = {task_map[int(t)] for t in np.unique(np.asarray(table["task_index"]))}
        if actual_tasks != set(row["tasks"]) or len(actual_tasks) != 1:
            raise ValueError(f"Episode {ep}: inconsistent task labels")
        if split != "sft":
            if not bool(table["done"][-1].as_py()) or bool(np.any(np.asarray(table["is_success"]))) != success:
                raise ValueError(f"Episode {ep}: terminal flags disagree with outcome metadata")
        for key in ("image", "wrist_image"):
            video = source / info["video_path"].format(
                episode_chunk=ep // info["chunks_size"], episode_index=ep, video_key=key
            )
            if not video.is_file():
                raise FileNotFoundError(video)
        # Ignore published return/reward annotations: recompute this recipe's targets.
        # Remove stale HF schema metadata (some files declare absent image columns).
        table = table.select(COLUMNS).replace_schema_metadata(None)
        table = table.set_column(table.column_names.index("index"), "index", pa.array(np.arange(total, total + length)))
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination)
        episode_stats = {}
        for key, width in (("state", 8), ("actions", 7)):
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if values.shape != (length, width) or not np.isfinite(values).all():
                raise ValueError(f"Episode {ep}: invalid {key}")
            episode_stats[key] = {
                "min": values.min(0).tolist(),
                "max": values.max(0).tolist(),
                "mean": values.mean(0).tolist(),
                "std": values.std(0).tolist(),
                "count": [length],
            }
        stats_rows.append({"episode_index": ep, "stats": episode_stats})
        row.update(success=success, source=split)
        total += length
    # Use the v2.1 metadata layout expected by the pinned LeRobot loader.
    # These descriptive stats are separate from OpenPI norm_stats, computed next.
    info.update(
        codebase_version="v2.1",
        total_frames=total,
        total_episodes=len(rows),
        total_tasks=len(tasks),
        total_chunks=(len(rows) + info["chunks_size"] - 1) // info["chunks_size"],
        total_videos=2 * len(rows),
        splits={"train": f"0:{len(rows)}"},
        features={key: info["features"][key] for key in (*COLUMNS, "image", "wrist_image")},
    )
    (root / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    write_jsonl(root / "meta/episodes.jsonl", rows)
    write_jsonl(root / "meta/tasks.jsonl", tasks)
    write_jsonl(root / "meta/episodes_stats.jsonl", stats_rows)
    (root / "videos").symlink_to(source / "videos", target_is_directory=True)
    (root / "recap_source.json").write_text(
        json.dumps(
            {"dataset": DATASET_ID, "revision": DATASET_REVISION, "split": split, "raw_dir": str(source)}, indent=2
        )
        + "\n"
    )
    (root / "INCOMPLETE").unlink()
    print(
        f"Prepared {split}: {len(rows)} episodes, {sum(row['success'] for row in rows)} successes, {total} frames: {root}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "prepare"])
    parser.add_argument("--raw-dir", type=pathlib.Path, default=pathlib.Path("data/recap/rlinf"))
    parser.add_argument(
        "--split", choices=list(REPOS), default="train", help="Split to prepare; download fetches all three"
    )
    parser.add_argument("--repo-id", help="Local output repo id; defaults to the corresponding RECAP_*_REPO_ID")
    args = parser.parse_args()
    if args.command == "download":
        download(args.raw_dir)
    else:
        variable, default = REPOS[args.split]
        prepare(args.raw_dir, args.split, args.repo_id or os.environ.get(variable, default))


if __name__ == "__main__":
    main()
