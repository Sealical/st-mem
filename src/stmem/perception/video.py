"""Bounded constant-frame-rate sampling shared by tracking and geometry."""

import json
import math
from pathlib import Path

from stmem.perception.tracking import video_digest


def sample_video(*, video, output, fps=2.0, max_frames=32, start_seconds=0.0, max_size=640):
    import cv2

    if (
        not math.isfinite(fps)
        or not math.isfinite(start_seconds)
        or fps <= 0
        or max_frames < 2
        or start_seconds < 0
        or max_size < 32
    ):
        raise ValueError("fps/max-frames/start-seconds/max-size are out of range")
    if Path(output).exists():
        raise FileExistsError("Choose a new sampled-video output directory")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(source_fps) or source_fps <= 0 or fps > source_fps:
        capture.release()
        raise ValueError("sample fps must be positive and no greater than source fps")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frames = output / "frames"
    frames.mkdir()
    writer = None
    rows = []
    try:
        for index in range(max_frames):
            source_id = round((start_seconds + index / fps) * source_fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_id)
            ok, bgr = capture.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            scale = min(1.0, max_size / max(h, w))
            width, height = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output / "sampled.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("could not create sampled video")
            writer.write(bgr)
            rows.append(
                {
                    "frame_id": index,
                    "timestamp_s": source_id / source_fps,
                    "source_frame_id": source_id,
                }
            )
    finally:
        capture.release()
        if writer:
            writer.release()
    if len(rows) < 2:
        raise ValueError("need at least two readable frames")
    # Decode the exact sampled video so downstream crops and geometry see identical pixels.
    sampled = cv2.VideoCapture(str(output / "sampled.mp4"))
    try:
        for row in rows:
            ok, bgr = sampled.read()
            if not ok:
                raise ValueError("sampled video frame count differs from metadata")
            relative = f"frames/{row['frame_id']:06d}.png"
            if not cv2.imwrite(str(output / relative), bgr):
                raise OSError(f"could not save frame {relative}")
            row["rgb_path"] = relative
    finally:
        sampled.release()
    payload = {
        "schema_version": 1,
        "source_video": Path(video).name,
        "video_sha256": video_digest(output / "sampled.mp4"),
        "sample_fps": fps,
        "frames": rows,
    }
    (output / "video.json").write_text(json.dumps(payload, indent=2) + "\n")
    return output / "video.json"
