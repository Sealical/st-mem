"""Small synthetic observations for a no-weights installation check.

The images, captions, geometry and identity vectors are synthetic fixtures,
not model predictions or paper benchmark results.
"""

from pathlib import Path

import numpy as np

from stmem.observations import (
    CaptionObservation,
    ObjectObservation,
    ObservationFrame,
    ObservationSequence,
)


def create_demo_bundle(output: str | Path) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    times = [0, 1, 2, 3, 4, 5, 6, 7, 10]
    for index, timestamp in enumerate(times):
        rgb = np.full((64, 96, 3), 220, dtype=np.uint8)
        rgb[36:52, 55:85] = [25, 90, 180]
        objects = [
            ObjectObservation(
                object_id="book_1",
                label="book",
                bbox_xyxy=(55, 36, 85, 52),
                position_world=(2, 0, 1),
                visual_feature=[0, 1, 0],
            )
        ]
        if index not in (6, 7):
            x = min(index, 2) * 0.5 if index < 8 else 1.4
            left = 5 + int(x * 12)
            rgb[12:32, left : left + 16] = [200, 45, 40]
            objects.append(
                ObjectObservation(
                    object_id="cup_1",
                    label="cup",
                    bbox_xyxy=(left, 12, left + 16, 32),
                    position_world=(x, 0, 1),
                    visual_feature=[1, 0, 0],
                )
            )
        filename = f"frame_{index:03d}.npy"
        np.save(output / filename, rgb, allow_pickle=False)
        frames.append(
            ObservationFrame(
                frame_id=index,
                timestamp_s=timestamp,
                rgb_path=filename,
                scene_id="office" if index == 8 else "kitchen",
                scene_label="office" if index == 8 else "kitchen",
                objects=objects,
            )
        )
    sequence = ObservationSequence(
        video_id="synthetic_stmem_demo",
        feature_space="synthetic-identity-v1",
        provenance={
            "source": "synthetic fixture",
            "geometry": "synthetic meters",
            "features": "synthetic identity vectors",
        },
        frames=frames,
        captions=[
            CaptionObservation(
                object_id="cup_1",
                t_start=0,
                t_end=2,
                text="The red cup moved from the sink to the table in the kitchen.",
            ),
            CaptionObservation(
                object_id="cup_1",
                t_start=2,
                t_end=5,
                text="The red cup remained stationary on the kitchen table.",
            ),
            CaptionObservation(
                object_id="cup_1",
                t_start=10,
                t_end=10,
                text="The red cup was seen again in the office.",
            ),
            CaptionObservation(
                object_id="book_1",
                t_start=0,
                t_end=10,
                text="The blue book remained stationary on the table.",
            ),
        ],
    )
    manifest = output / "observations.json"
    manifest.write_text(sequence.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest
