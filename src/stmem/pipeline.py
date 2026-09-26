"""Build a self-contained memory without an external workflow framework."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from stmem import MemoryViews, QueryEngine, STMemBuilder, STMemParams, load_memory, save_memory
from stmem.dataset import ObservationDataset
from stmem.fixtures import create_demo_bundle


def view_counts(memory) -> dict[str, int]:
    views = MemoryViews(memory)
    return {
        name: len(getattr(views, name))
        for name in ("objects", "scenes", "texts", "events", "images")
    }


def build_memory(
    observations: str | Path,
    output: str | Path,
    *,
    params: STMemParams | None = None,
    captioner: Callable | None = None,
) -> Path:
    """Build from an observation manifest into a new directory. Never overwrite outputs."""
    manifest = Path(observations).resolve()
    source = ObservationDataset(str(manifest.parent), manifest.name)
    params = params or STMemParams()
    if params.captions == "vlm" and captioner is None:
        raise ValueError("VLM captions require an explicit captioner or --vlm-checkpoint")
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    builder = STMemBuilder(
        source.sequence.video_id,
        params=params,
        feature_space=source.sequence.feature_space,
        captions=source.sequence.captions,
        image_getter=source.image_by_id,
        captioner=captioner,
        provenance=source.sequence.provenance,
    )
    for packet in source:
        builder.forward({"packet": packet})
    memory = builder.finalize(directory / "crops")
    return save_memory(memory, directory / "memory.json")


def run_demo(output: str | Path) -> dict:
    """Generate synthetic inputs, build memory, reload it, and query all four paths."""
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = create_demo_bundle(directory / "observations")
    memory_path = build_memory(manifest, directory / "memory")
    memory = load_memory(memory_path)
    engine = QueryEngine(memory)
    source = "Synthetic fixture. No model inference or benchmark evaluation."
    results = {
        "source": source,
        "memory": "memory/memory.json",
        "nlq": engine.query_nlq("red cup moved sink kitchen"),
        "vq2d": engine.query_vq2d([1, 0, 0], feature_space="synthetic-identity-v1"),
        "str": engine.query_str(object_id="cup_1"),
        "lor": engine.query_lor("cup"),
    }
    queries = directory / "queries.json"
    queries.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    feature = directory / "query_feature.json"
    feature.write_text(
        json.dumps({"feature_space": memory.feature_space, "feature": [1, 0, 0]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "source": source,
        "memory": str(memory_path),
        "queries": str(queries),
        "feature_file": str(feature),
        "views": view_counts(memory),
        "query_counts": {name: len(results[name]) for name in ("nlq", "vq2d", "str", "lor")},
    }
