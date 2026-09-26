"""ST-Mem invariants: geometry, gaps, cross-view links and portable query replay."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from stmem import (
    MemoryViews,
    QueryEngine,
    STMemBuilder,
    STMemParams,
    load_memory,
    save_memory,
)
from stmem.dataset import ObservationDataset, bundle_path
from stmem.fixtures import create_demo_bundle
from stmem.lte import (
    LTEParams,
    TrackSample,
    douglas_peucker_3d,
    segment_trajectory,
)
from stmem.observations import (
    ObjectObservation,
    ObservationFrame,
    ObservationSequence,
)
from stmem.types import CameraModel, FramePacket


@pytest.fixture
def built_memory(tmp_path):
    manifest = create_demo_bundle(tmp_path / "observations")
    source = ObservationDataset(str(manifest.parent))
    builder = STMemBuilder(
        source.sequence.video_id,
        feature_space=source.sequence.feature_space,
        captions=source.sequence.captions,
        image_getter=source.image_by_id,
    )
    for packet in source:
        builder.forward({"packet": packet})
    directory = tmp_path / "memory"
    memory = builder.finalize(directory / "crops")
    return memory, save_memory(memory, directory / "memory.json"), source


def test_motion_stop_is_split_without_tracking_gap(built_memory):
    memory, _, _ = built_memory
    intervals = memory.objects["cup_1"].intervals
    assert [(iv.kind, iv.t_start, iv.t_end) for iv in intervals[:3]] == [
        ("motion", 0, 2),
        ("static", 2, 5),
        ("unobserved", 6, 7),
    ]
    assert intervals[0].spatial_anchors[0].position_world == (0, 0, 1)
    assert intervals[0].spatial_anchors[-1].position_world == (1, 0, 1)
    assert len(intervals[1].spatial_anchors) == 1
    transitions = [e for e in memory.events if e.kind == "motion_transition"]
    assert any(e.timestamp_s == 2 and e.object_id == "cup_1" for e in transitions)


def test_gap_holds_actual_last_observation_and_no_visual_evidence(built_memory):
    memory, _, _ = built_memory
    gap = memory.objects["cup_1"].intervals[2]
    assert gap.kind == "unobserved"
    assert gap.spatial_anchors[0].timestamp_s == 5
    assert not gap.spatial_anchors[0].observed
    assert gap.visual_anchor_ids == []
    engine = QueryEngine(memory)
    assert engine.query_lor("cup", as_of=7)[0]["timestamp_s"] == 5
    assert engine.query_lor("cup", as_of=7, lookback_seconds=1) == []
    assert engine.query_lor("cup")[0]["timestamp_s"] == 10


def test_all_five_views_share_valid_references_and_frame_zero_crop(built_memory):
    memory, path, _ = built_memory
    views = MemoryViews(memory)
    assert all((views.objects, views.scenes, views.events, views.texts, views.images))
    assert {e["scene_ids"][0] for e in views.texts if e["object_id"] is None} == {
        "kitchen",
        "office",
    }
    first = next(a for a in views.images if a.object_id == "cup_1" and a.frame_id == 0)
    crop = np.load(path.parent / first.crop_path, allow_pickle=False)
    assert crop.shape == (20, 16, 3)
    np.testing.assert_array_equal(crop[0, 0], [200, 45, 40])


def test_four_queries_and_filters(built_memory):
    memory, _, _ = built_memory
    engine = QueryEngine(memory)
    nlq = engine.query_nlq("cup moved sink kitchen", object_id="cup_1", scene="kitchen")
    assert nlq[0]["t_start"] == 0 and nlq[0]["t_end"] == 2
    assert engine.query_nlq("elephant skateboard") == []
    visual = engine.query_vq2d([1, 0, 0], feature_space="synthetic-identity-v1")
    assert visual and {r["object_id"] for r in visual} == {"cup_1"}
    trajectory = engine.query_str(object_id="cup_1", scene="kitchen", time_range=(0, 2))
    assert trajectory[0]["kind"] == "motion"
    assert trajectory[0]["spatial_anchors"][-1]["position_world"] == (1, 0, 1)
    assert engine.query_str(object_id="cup_1", scene="bedroom") == []
    assert engine.query_str(object_id="cup_1", region=([8, 8, 8], [9, 9, 9])) == []
    assert engine.query_lor("book")[0]["object_id"] == "book_1"


@pytest.mark.parametrize(
    "space,vector",
    [
        ("another-model", [1, 0, 0]),
        ("synthetic-identity-v1", [1, 0]),
        ("synthetic-identity-v1", [0, 0, 0]),
    ],
)
def test_visual_model_mismatch_is_an_error(built_memory, space, vector):
    with pytest.raises(ValueError):
        QueryEngine(built_memory[0]).query_vq2d(vector, feature_space=space)


def test_save_reload_portable_without_original_data_or_models(built_memory, tmp_path):
    memory, path, source = built_memory
    portable = tmp_path / "portable"
    shutil.copytree(path.parent, portable)
    # The replay copy must be self-contained even if the source directory is moved.
    source.root.rename(tmp_path / "source_moved")
    replay = load_memory(portable)
    assert replay == memory
    assert QueryEngine(replay).query_lor("cup")[0]["position_world"] == (1.4, 0, 1)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stmem.cli",
            "query",
            "--memory",
            str(portable),
            "--type",
            "lor",
            "--label",
            "cup",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout)[0]["timestamp_s"] == 10


def test_bad_assets_schema_and_links_are_rejected(built_memory, tmp_path):
    memory, path, _ = built_memory
    payload = memory.model_dump()
    payload["schema_version"] = 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_memory(bad)
    payload = memory.model_dump()
    payload["objects"]["cup_1"]["intervals"][0]["visual_anchor_ids"] = ["missing"]
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_memory(bad)
    payload = memory.model_dump()
    payload["objects"]["cup_1"]["visual_anchors"][0]["crop_path"] = "../../outside.npy"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="escapes"):
        load_memory(bad, validate_assets=False)
    with pytest.raises(FileExistsError):
        save_memory(memory, path)
    with pytest.raises(ValueError):
        bundle_path(tmp_path, "../elsewhere")


def test_metric_geometry_without_room_label_and_input_ownership(tmp_path):
    camera = CameraModel(width=8, height=8, fx=4, fy=4, cx=4, cy=4)
    pose = np.eye(4)
    pose[0, 3] = 10
    obj = ObjectObservation(object_id="cup", label="cup", bbox_xyxy=(2, 2, 6, 6))
    frame = ObservationFrame(frame_id=0, timestamp_s=0, objects=[obj])
    packet = FramePacket(
        frame_id=0,
        timestamp_s=0,
        camera=camera,
        depth=np.full((8, 8), 2),
        raw_pose_world_camera=pose,
        extras={"stmem": frame},
    )
    builder = STMemBuilder(
        "metric",
        params=STMemParams(captions="template", require_visual_features=False, require_crops=False),
    )
    builder.forward({"packet": packet})
    memory = builder.finalize(tmp_path / "crops")
    assert memory.objects["cup"].visual_anchors[0].position_world == (10, 0, 2)
    assert obj.position_world is None
    assert frame.scene_label == "unknown"


def test_missing_geometry_fails_instead_of_inventing_world_position():
    frame = ObservationFrame(
        frame_id=0,
        timestamp_s=0,
        objects=[ObjectObservation(object_id="cup", label="cup", bbox_xyxy=(1, 1, 3, 3))],
    )
    builder = STMemBuilder("missing", params=STMemParams(require_visual_features=False))
    with pytest.raises(ValueError, match="3D memory needs"):
        builder.forward({"packet": FramePacket(frame_id=0, timestamp_s=0, extras={"stmem": frame})})


def test_observation_order_and_finiteness_validation():
    frame = ObservationFrame(frame_id=0, timestamp_s=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        ObservationSequence(video_id="bad", frames=[frame, frame])
    with pytest.raises(ValueError):
        ObjectObservation(
            object_id="a", label="cup", bbox_xyxy=(0, 0, 1, 1), position_world=(float("nan"), 0, 0)
        )


def test_rdp_preserves_turn_and_handles_long_line_without_recursion():
    assert douglas_peucker_3d(np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0]]), 0.1) == [0, 1, 2]
    points = np.column_stack([np.arange(5000), np.zeros((5000, 2))])
    assert douglas_peucker_3d(points, 0.1) == [0, 4999]


def test_stationary_detection_uses_seconds_not_frame_ids():
    obj = ObjectObservation(
        object_id="a", label="cup", bbox_xyxy=(0, 0, 1, 1), position_world=(1, 0, 1)
    )
    samples = [TrackSample(i * 100, float(i), "room", obj) for i in range(4)]
    assert segment_trajectory(samples, LTEParams())[0][0] == "static"


def test_core_import_does_not_load_torch():
    code = "import sys; import stmem; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_viewer_scene_change_during_gap_does_not_move_the_object(tmp_path):
    obj = ObjectObservation(
        object_id="cup", label="cup", bbox_xyxy=(0, 0, 1, 1), position_world=(1, 0, 1)
    )
    builder = STMemBuilder(
        "gap",
        params=STMemParams(captions="template", require_visual_features=False, require_crops=False),
    )
    for i, scene in enumerate(["kitchen", "kitchen", "office", "hall"]):
        frame = ObservationFrame(
            frame_id=i,
            timestamp_s=i,
            scene_id=scene,
            scene_label=scene,
            objects=[obj] if i == 0 else [],
        )
        builder.forward({"packet": FramePacket(frame_id=i, timestamp_s=i, extras={"stmem": frame})})
    memory = builder.finalize(tmp_path / "crops")
    intervals = memory.objects["cup"].intervals
    assert len(intervals) == 2
    gap = intervals[-1]
    assert gap.kind == "unobserved" and gap.scene_ids == ["kitchen"]
    assert (gap.t_start, gap.t_end) == (1, 3)
    assert QueryEngine(memory).query_lor("cup", scene="office") == []
    assert QueryEngine(memory).query_lor("cup", scene="kitchen")[0]["timestamp_s"] == 0


def test_str_visual_evidence_obeys_the_time_window(built_memory):
    memory, _, _ = built_memory
    result = QueryEngine(memory).query_str(object_id="cup_1", time_range=(0, 1))
    anchors = {a.anchor_id: a for a in MemoryViews(memory).images}
    assert result
    assert all(
        0 <= anchors[aid].timestamp_s <= 1 for row in result for aid in row["visual_anchor_ids"]
    )
