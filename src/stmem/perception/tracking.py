"""Optional on-disk SAM3 cache, bound to the exact input video bytes."""

import hashlib
import json
from pathlib import Path

import numpy as np

from stmem.dataset import bundle_path


def video_digest(video: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(video).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_tracking_cache(tracks: dict, video: str, output: str, prompts: list[str]):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rows = {}
    for frame_id, objects in tracks.items():
        entries = []
        for index, obj in enumerate(objects):
            filename = f"frame_{frame_id}_object_{index}.npy"
            np.save(root / filename, obj["mask"], allow_pickle=False)
            entries.append({**{k: v for k, v in obj.items() if k != "mask"}, "mask_path": filename})
        rows[str(frame_id)] = entries
    payload = {
        "schema_version": 1,
        "video_sha256": video_digest(video),
        "prompts": prompts,
        "frames": rows,
    }
    path = root / "tracking.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_tracking_cache(path: str | Path, video: str, prompts: list[str]):
    path = Path(path)
    payload = json.loads(path.read_text())
    if (
        payload["schema_version"] != 1
        or payload["video_sha256"] != video_digest(video)
        or payload["prompts"] != prompts
    ):
        raise ValueError("tracking cache version, input video or prompts do not match")
    return {
        int(frame_id): [
            {
                **{key: value for key, value in obj.items() if key != "mask_path"},
                "mask": np.load(bundle_path(path.parent, obj["mask_path"]), allow_pickle=False),
            }
            for obj in objects
        ]
        for frame_id, objects in payload["frames"].items()
    }
