"""Run ViPE in a separate environment with the Metric3D-small depth backend."""

import hashlib
import json
import random
from pathlib import Path


def run_vipe(*, video, output, depth_model="metric3d-small", seed=42):
    if not Path(video).is_file():
        raise FileNotFoundError(video)
    if depth_model != "metric3d-small":
        raise ValueError("This integration supports the validated metric3d-small backend")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new ViPE output directory")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    from vipe import make_pipeline
    from vipe.config import parse_typed_config
    from vipe.streams.base import ProcessedVideoStream
    from vipe.streams.raw_mp4_stream import RawMp4Stream

    config = parse_typed_config(
        "default",
        hydra_args=[
            "pipeline=default",
            "pipeline.init.instance=null",
            "pipeline.init.async_prefetch=false",
            f"pipeline.slam.keyframe_depth={depth_model}",
            f"pipeline.post.depth_align_model=adaptive_{depth_model}",
            "pipeline.output.save_artifacts=true",
            "pipeline.output.save_viz=false",
            "pipeline.slam.visualize=false",
            f"pipeline.output.path={output}",
        ],
    )
    stream = ProcessedVideoStream(RawMp4Stream(Path(video)), []).cache(desc="ST-Mem video")
    make_pipeline(config.pipeline).run(stream)
    with Path(video).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    (output / "stmem_vipe.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "video_sha256": digest,
                "video_name": Path(video).stem,
                "depth_model": depth_model,
                "seed": seed,
            },
            indent=2,
        )
        + "\n"
    )

    return output / "stmem_vipe.json"
