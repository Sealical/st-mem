"""SAM3 video + ViPE artifacts + DINOv2 -> a portable observation bundle.

Run after ViPE's separate process. The memory pipeline handles interval captioning.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from stmem.dataset import bundle_path, load_rgb
from stmem.observations import (
    ObjectObservation,
    ObservationFrame,
    ObservationSequence,
)
from stmem.perception.runtime import seed_everything
from stmem.types import CameraModel


def prepare_observations(
    *,
    video_bundle: str,
    vipe_root: str,
    output: str,
    prompts: list[str],
    sam3_checkpoint: str | None = None,
    dinov2_checkpoint: str,
    dinov2_repo: str,
    device="cuda:0",
    scene_label="unknown",
    seed=42,
    tracking_cache: str | None = None,
):
    if not tracking_cache and not sam3_checkpoint:
        raise ValueError("Provide --sam3-checkpoint or a matching --tracking-cache")
    # Seed before importing any CUDA model; the caller chooses the visible physical GPU.
    seed_everything(seed)
    from stmem.perception.dinov2 import DINOv2Encoder
    from stmem.perception.geometry import ViPEArtifacts
    from stmem.perception.sam3 import Sam3VideoTracker

    bundle = Path(video_bundle).resolve()
    metadata = json.loads((bundle / "video.json").read_text())
    if metadata.get("schema_version") != 1 or not metadata.get("frames"):
        raise ValueError("Invalid sampled-video metadata")
    from stmem.perception.tracking import video_digest

    if metadata.get("video_sha256") != video_digest(bundle / "sampled.mp4"):
        raise ValueError("Sampled-video metadata does not match the clip")
    geometry = ViPEArtifacts(vipe_root, "sampled")
    geometry.validate_video(bundle / "sampled.mp4")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "frames").mkdir()
    if tracking_cache:
        from stmem.perception.tracking import load_tracking_cache

        tracks = load_tracking_cache(tracking_cache, str(bundle / "sampled.mp4"), prompts)
    else:
        tracker = Sam3VideoTracker(sam3_checkpoint, device=device)
        try:
            tracks = tracker.forward({"video": str(bundle / "sampled.mp4"), "prompts": prompts})[
                "frames"
            ]
        finally:
            tracker.close()
    import torch

    torch.cuda.empty_cache()
    encoder = DINOv2Encoder(dinov2_checkpoint, device=device, repo=dinov2_repo)
    frames = []
    try:
        for row in metadata["frames"]:
            frame_id = row["frame_id"]
            rgb_path = bundle_path(bundle, row["rgb_path"])
            rgb = load_rgb(rgb_path)
            height, width = rgb.shape[:2]
            geo = geometry.forward({"frame_id": frame_id})
            depth, pose, intrinsics = geo["depth"], geo["T_world_camera"], geo["intrinsics"]
            if depth.shape != (height, width):
                raise ValueError(
                    "ViPE output resolution differs from the sampled video; use the same video for both passes"
                )
            fx, fy, cx, cy = intrinsics
            camera = CameraModel(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
            objects = []
            for track in tracks.get(frame_id, []):
                mask = track["mask"]
                if mask.shape != depth.shape:
                    raise ValueError("SAM3 mask and ViPE depth resolutions differ")
                valid = mask & np.isfinite(depth) & (depth > 0)
                if not valid.any():
                    raise ValueError(
                        f"no valid geometry for {track['object_id']} at frame {frame_id}"
                    )
                v, u = np.nonzero(valid)
                z = depth[valid]
                points_camera = np.column_stack([(u - cx) * z / fx, (v - cy) * z / fy, z])
                center = np.median(points_camera, axis=0)
                position_world = (pose @ np.append(center, 1))[:3]
                x1, y1, x2, y2 = track["bbox_xyxy"]
                embedding = encoder.forward({"image": rgb[y1:y2, x1:x2]})["embedding"]
                objects.append(
                    ObjectObservation(
                        object_id=track["object_id"],
                        label=track["label"],
                        bbox_xyxy=track["bbox_xyxy"],
                        position_world=tuple(position_world),
                        visual_feature=embedding,
                        confidence=track["confidence"],
                    )
                )
            relative = f"frames/{frame_id:06d}.png"
            shutil.copyfile(rgb_path, root / relative)
            frames.append(
                ObservationFrame(
                    frame_id=frame_id,
                    timestamp_s=row["timestamp_s"],
                    rgb_path=relative,
                    camera=camera,
                    T_world_camera=pose.tolist(),
                    objects=objects,
                    scene_id="scene_0",
                    scene_label=scene_label,
                )
            )
            print(f"[stmem preprocess] frame {frame_id}: {len(objects)} objects", flush=True)
    finally:
        encoder.close()
    sequence = ObservationSequence(
        video_id=Path(metadata["source_video"]).stem,
        feature_space=encoder.feature_space,
        provenance={
            "tracking": "SAM3 video",
            "geometry": "ViPE metric depth + T_world_camera",
            "features": encoder.feature_space,
            "scene_label": "user-provided" if scene_label != "unknown" else "unknown",
        },
        frames=frames,
    )
    path = root / "observations.json"
    path.write_text(sequence.model_dump_json(indent=2) + "\n")
    return path
