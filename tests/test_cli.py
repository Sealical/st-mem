"""Public CLI acceptance checks, including portable replay in fresh processes."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stmem import STMemParams, load_memory
from stmem.pipeline import build_memory, run_demo
from stmem.verify import verify


def invoke(*args, cwd, check=True):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-m", "stmem", *map(str, args)],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=check,
    )


@pytest.fixture
def demo(tmp_path):
    result = invoke("demo", "--output", tmp_path / "demo", cwd=tmp_path)
    return json.loads(result.stdout), tmp_path


def test_demo_builds_five_views_and_queries(demo):
    result, root = demo
    assert result["views"] == {"objects": 2, "scenes": 2, "texts": 8, "events": 5, "images": 7}
    assert result["query_counts"] == {"nlq": 5, "vq2d": 4, "str": 4, "lor": 1}
    payload = json.loads(Path(result["queries"]).read_text())
    assert payload["source"].startswith("Synthetic fixture.")
    assert (root / "demo" / payload["memory"]).is_file()
    assert payload["lor"][0]["timestamp_s"] == 10
    inspect = invoke("inspect", "--memory", result["memory"], cwd=root)
    assert json.loads(inspect.stdout)["views"] == result["views"]


@pytest.mark.parametrize("query_type", ["nlq", "vq2d", "str", "lor"])
def test_query_commands(demo, query_type):
    result, root = demo
    options = {
        "nlq": ["--text", "cup moved sink", "--object-id", "cup_1"],
        "vq2d": ["--feature-file", result["feature_file"]],
        "str": ["--object-id", "cup_1", "--scene", "kitchen", "--start", "0", "--end", "2"],
        "lor": ["--label", "cup", "--as-of", "7", "--lookback", "3"],
    }
    process = invoke(
        "query", "--memory", result["memory"], "--type", query_type, *options[query_type], cwd=root
    )
    payload = json.loads(process.stdout)
    assert payload and all(row["query_type"] == query_type for row in payload)
    if query_type == "lor":
        assert payload[0]["timestamp_s"] == 5


def test_build_replays_identical_memory(demo):
    result, root = demo
    process = invoke(
        "build",
        "--observations",
        root / "demo/observations/observations.json",
        "--output",
        root / "rebuilt",
        cwd=root,
    )
    assert load_memory(json.loads(process.stdout)["memory"]) == load_memory(result["memory"])


def test_saved_memory_survives_moving_source_and_fresh_process_replay(demo):
    result, root = demo
    source_memory = Path(result["memory"])
    shutil.copytree(source_memory.parent, root / "portable")
    (root / "demo").rename(root / "original-moved")
    report = verify(root / "portable/memory.json", "cup_1", "cup")
    assert report["portable_reload"] == "passed"
    assert all(report["fresh_process_queries"].values())
    process = invoke("verify", "--memory", root / "portable", cwd=root)
    assert json.loads(process.stdout)["query_counts"] == {"nlq": 4, "vq2d": 4, "str": 4, "lor": 1}


def test_outputs_are_never_overwritten(demo):
    result, root = demo
    before = Path(result["memory"]).read_bytes()
    with pytest.raises(FileExistsError):
        run_demo(root / "demo")
    with pytest.raises(FileExistsError):
        build_memory(root / "demo/observations/observations.json", root / "demo/memory")
    process = invoke("demo", "--output", root / "demo", cwd=root, check=False)
    assert process.returncode == 2
    assert Path(result["memory"]).read_bytes() == before


@pytest.mark.parametrize(
    "options",
    [
        ["--type", "nlq"],
        ["--type", "nlq", "--text", "cup", "--top-k", "0"],
        ["--type", "str", "--start", "1"],
        ["--type", "str", "--start", "2", "--end", "1"],
        ["--type", "lor", "--lookback", "-1"],
        ["--type", "lor", "--start", "1", "--end", "2"],
        ["--type", "vq2d"],
    ],
)
def test_invalid_query_is_an_explicit_failure(demo, options):
    result, root = demo
    process = invoke("query", "--memory", result["memory"], *options, cwd=root, check=False)
    assert process.returncode == 2 and process.stderr


def test_imports_do_not_load_models_or_private_framework(tmp_path):
    # Test installed imports from a directory that contains no source code.
    code = (
        "import sys, stmem, stmem.cli, stmem.pipeline; "
        "assert not any(name == 'torch' or name.startswith('embodied_slam') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
    assert importlib.util.find_spec("stmem") is not None


def test_vlm_build_requires_explicit_backend_and_runs_injected_captioner(demo):
    _result, root = demo
    observations = root / "demo/observations/observations.json"
    with pytest.raises(ValueError, match="explicit captioner"):
        build_memory(observations, root / "missing", params=STMemParams(captions="vlm"))
    calls = []

    def captioner(label, kind, samples, image_getter):
        calls.append((label, kind))
        assert image_getter(samples[0].frame_id) is not None
        return f"The {label} is visible."

    path = build_memory(
        observations, root / "captioned", params=STMemParams(captions="vlm"), captioner=captioner
    )
    assert calls and load_memory(path).objects
    process = invoke(
        "build",
        "--observations",
        observations,
        "--output",
        root / "invalid",
        "--captions",
        "vlm",
        cwd=root,
        check=False,
    )
    assert process.returncode == 2 and "--vlm-checkpoint" in process.stderr
    assert not (root / "invalid").exists()


def test_image_query_requires_explicit_dino_checkpoint(demo):
    result, root = demo
    process = invoke(
        "query",
        "--memory",
        result["memory"],
        "--type",
        "vq2d",
        "--image",
        "missing.png",
        cwd=root,
        check=False,
    )
    assert process.returncode == 2 and "--dinov2-checkpoint" in process.stderr
