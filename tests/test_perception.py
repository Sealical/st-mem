"""Optional adapter contracts without GPU or model dependencies."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from stmem.perception.captioner import MotionCaptioner
from stmem.perception.sam3 import Sam3VideoTracker


def test_tracking_cache_is_bound_to_video_prompts_and_safe_assets(tmp_path):
    from stmem.perception.tracking import load_tracking_cache, save_tracking_cache

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"synthetic video fingerprint fixture")
    tracks = {0: [{"object_id": "cup", "mask": np.ones((2, 3), dtype=bool)}]}
    path = save_tracking_cache(tracks, str(video), str(tmp_path / "tracking"), ["cup"])
    result = load_tracking_cache(path, str(video), ["cup"])
    np.testing.assert_array_equal(result[0][0]["mask"], tracks[0][0]["mask"])
    with pytest.raises(ValueError, match="do not match"):
        load_tracking_cache(path, str(video), ["book"])
    video.write_bytes(b"different video")
    with pytest.raises(ValueError, match="do not match"):
        load_tracking_cache(path, str(video), ["cup"])
    video.write_bytes(b"synthetic video fingerprint fixture")
    payload = json.loads(path.read_text())
    payload["frames"]["0"][0]["mask_path"] = "../../outside.npy"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inside its root"):
        load_tracking_cache(path, str(video), ["cup"])


def test_vipe_frame_indices_are_not_array_positions(tmp_path):
    from stmem.perception.geometry import ViPEArtifacts

    path = tmp_path / "pose.npz"
    np.savez(path, inds=np.array([5, 0]), data=np.array([[50, 51], [0, 1]]))
    indexed = ViPEArtifacts._indexed(path)
    np.testing.assert_array_equal(indexed[0], [0, 1])
    np.testing.assert_array_equal(indexed[5], [50, 51])
    np.savez(path, inds=np.array([1, 1]), data=np.ones((2, 2)))
    with pytest.raises(ValueError, match="invalid ViPE frame index"):
        ViPEArtifacts._indexed(path)


@pytest.mark.parametrize("indices", [[0.5, 1.5], [-1, 0]])
def test_vipe_rejects_non_integer_or_negative_frame_indices(tmp_path, indices):
    from stmem.perception.geometry import ViPEArtifacts

    path = tmp_path / "pose.npz"
    np.savez(path, inds=np.array(indices), data=np.ones((2, 2)))
    with pytest.raises(ValueError, match="invalid ViPE frame index"):
        ViPEArtifacts._indexed(path)


def test_vipe_provenance_rejects_a_different_same_named_video(tmp_path):
    from stmem.perception.geometry import ViPEArtifacts
    from stmem.perception.tracking import video_digest

    video = tmp_path / "sampled.mp4"
    video.write_bytes(b"synthetic video fixture")
    artifacts = ViPEArtifacts.__new__(ViPEArtifacts)
    artifacts.root, artifacts.video_name = tmp_path, "sampled"
    with pytest.raises(ValueError, match="missing ViPE provenance"):
        artifacts.validate_video(video)
    (tmp_path / "stmem_vipe.json").write_text(
        json.dumps(
            {"schema_version": 1, "video_sha256": video_digest(video), "video_name": "sampled"}
        )
    )
    artifacts.validate_video(video)
    video.write_bytes(b"different same-named video")
    with pytest.raises(ValueError, match="do not match"):
        artifacts.validate_video(video)


def test_all_adapters_import_without_heavy_models(tmp_path):
    code = """
import importlib, sys
for name in ('sam3', 'geometry', 'dinov2', 'qwen', 'captioner', 'prepare', 'video', 'vipe', 'runtime', 'cli'):
    importlib.import_module('stmem.perception.' + name)
assert not {'torch', 'transformers', 'sam3', 'vipe', 'cv2', 'OpenEXR'} & sys.modules.keys()
assert not any(name.startswith('embodied_slam') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


@pytest.mark.parametrize("stage", ["sample", "track", "vipe", "prepare"])
def test_video_help_without_model_runtime(stage, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "stmem", "video", stage, "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--output" in result.stdout


def test_sam3_tracks_keep_ids_and_always_close_sessions(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fixture")

    class Predictor:
        def __init__(self):
            self.requests = []
            self.closed = False
            self.fail = False

        def handle_request(self, request):
            self.requests.append(request)
            return {"session_id": "fixture"}

        def handle_stream_request(self, request):
            if self.fail:
                raise RuntimeError("upstream failure")
            for frame in (0, 1):
                yield {
                    "frame_index": frame,
                    "outputs": {
                        "out_obj_ids": [7],
                        "out_binary_masks": [np.array([[0, 1, 1], [0, 1, 1]])],
                        "out_probs": [0.9],
                    },
                }

        def shutdown(self):
            self.closed = True

    tracker = Sam3VideoTracker("unused")
    predictor = Predictor()
    tracker.predictor = predictor
    result = tracker.forward({"video": video, "prompts": ["cup", "book"]})["frames"]
    assert len(result) == 2
    assert [row["object_id"] for row in result[0]] == ["prompt_0/track_7", "prompt_1/track_7"]
    assert result[1][0]["bbox_xyxy"] == [1, 0, 3, 2]
    assert sum(row["type"] == "close_session" for row in predictor.requests) == 2
    predictor.fail = True
    with pytest.raises(RuntimeError, match="upstream failure"):
        tracker.forward({"video": video, "prompts": ["cup"]})
    assert predictor.requests[-1]["type"] == "close_session"
    tracker.close()
    assert predictor.closed and tracker.predictor is None


def test_sam3_rejects_duplicate_prompts_before_loading(tmp_path):
    video = tmp_path / "sample.mp4"
    video.touch()
    tracker = Sam3VideoTracker("unused")
    with pytest.raises(ValueError, match="unique"):
        tracker.forward({"video": video, "prompts": ["cup", "cup"]})
    assert tracker.predictor is None


def test_captioner_uses_real_times_and_boxes_without_fps_assumptions():
    captioner = MotionCaptioner("unused", "cpu")
    recorded = {}

    def capture(inputs):
        recorded.update(inputs)
        return {"text": "A cup moves on the desk."}

    captioner.model.forward = capture
    samples = [
        SimpleNamespace(
            frame_id=i * 100, timestamp_s=t, observation=SimpleNamespace(bbox_xyxy=(1, 2, 4, 5))
        )
        for i, t in enumerate((0.2, 1.7, 9.0))
    ]
    result = captioner("cup", "moving", samples, lambda _: np.zeros((6, 6, 3), np.uint8))
    content = recorded["messages"][0]["content"]
    assert result == "A cup moves on the desk."
    assert [row["text"].split(",")[0] for row in content[1:] if row["type"] == "text"] == [
        "t=0.2s",
        "t=1.7s",
        "t=9s",
    ]
    assert len([row for row in content if row["type"] == "image"]) == 3
    with pytest.raises(ValueError, match="observed samples"):
        captioner("cup", "moving", [], lambda _: None)


def test_preparation_matches_masks_depth_and_world_coordinates(tmp_path, monkeypatch):
    from stmem.observations import ObservationSequence
    from stmem.perception import dinov2, geometry, prepare
    from stmem.perception.tracking import save_tracking_cache, video_digest

    bundle = tmp_path / "video"
    bundle.mkdir()
    video = bundle / "sampled.mp4"
    video.write_bytes(b"fixture")
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(bundle / "0.png")
    metadata = {
        "schema_version": 1,
        "source_video": "sample.mp4",
        "video_sha256": video_digest(video),
        "frames": [{"frame_id": 0, "timestamp_s": 2.5, "rgb_path": "0.png"}],
    }
    (bundle / "video.json").write_text(json.dumps(metadata))
    tracks = {
        0: [
            {
                "object_id": "prompt_0/track_0",
                "label": "cup",
                "bbox_xyxy": [0, 0, 4, 4],
                "confidence": 0.9,
                "mask": np.ones((4, 4), dtype=bool),
            }
        ]
    }
    cache = save_tracking_cache(tracks, str(video), str(tmp_path / "tracks"), ["cup"])

    class Geometry:
        def __init__(self, *_):
            pass

        def validate_video(self, video_path):
            assert video_path == video

        def forward(self, inputs):
            assert inputs == {"frame_id": 0}
            pose = np.eye(4)
            pose[0, 3] = 10
            return {
                "depth": np.ones((4, 4)),
                "T_world_camera": pose,
                "intrinsics": np.array([2, 2, 1.5, 1.5]),
            }

    class Encoder:
        feature_space = "test-features"

        def __init__(self, *_args, **_kwargs):
            pass

        def forward(self, inputs):
            assert inputs["image"].shape == (4, 4, 3)
            return {"embedding": [1, 0, 0]}

        def close(self):
            pass

    monkeypatch.setattr(geometry, "ViPEArtifacts", Geometry)
    monkeypatch.setattr(dinov2, "DINOv2Encoder", Encoder)
    monkeypatch.setattr(prepare, "seed_everything", lambda _: None)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
    )
    options = dict(
        video_bundle=str(bundle),
        vipe_root="unused",
        prompts=["cup"],
        sam3_checkpoint=None,
        dinov2_checkpoint="unused",
        dinov2_repo="unused",
        tracking_cache=str(cache),
        scene_label="office",
        device="cpu",
    )
    path = prepare.prepare_observations(output=str(tmp_path / "observations"), **options)
    sequence = ObservationSequence.model_validate_json(path.read_text())
    assert sequence.frames[0].timestamp_s == 2.5
    assert sequence.frames[0].objects[0].position_world == (10.0, 0.0, 1.0)
    assert sequence.frames[0].scene_label == "office"
    assert sequence.feature_space == "test-features"
    with pytest.raises(FileExistsError):
        prepare.prepare_observations(output=str(path.parent), **options)
    metadata["video_sha256"] = "wrong"
    (bundle / "video.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata does not match"):
        prepare.prepare_observations(output=str(tmp_path / "mismatch"), **options)
    assert not (tmp_path / "mismatch").exists()


def test_sampling_decodes_same_video_and_preserves_original_timestamps(tmp_path, monkeypatch):
    from stmem.perception.tracking import video_digest
    from stmem.perception.video import sample_video

    class Capture:
        def __init__(self, path):
            self.sampled = path.endswith("sampled.mp4")

        def isOpened(self):
            return True

        def get(self, _):
            return 20.0

        def set(self, _key, _value):
            pass

        def read(self):
            return True, np.zeros((4, 6, 3), np.uint8)

        def release(self):
            pass

    class Writer:
        def __init__(self, path, *_):
            Path(path).write_bytes(b"encoded video fixture")

        def isOpened(self):
            return True

        def write(self, _):
            pass

        def release(self):
            pass

    def save(path, bgr):
        Image.fromarray(bgr).save(path)
        return True

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoCapture=Capture,
            VideoWriter=Writer,
            VideoWriter_fourcc=lambda *_: 0,
            CAP_PROP_FPS=1,
            CAP_PROP_POS_FRAMES=2,
            INTER_AREA=3,
            resize=lambda image, _size, **_kwargs: image,
            imwrite=save,
        ),
    )
    path = sample_video(
        video="source.mp4",
        output=tmp_path / "video",
        fps=2,
        max_frames=3,
        start_seconds=1.5,
        max_size=480,
    )
    data = json.loads(path.read_text())
    assert [row["timestamp_s"] for row in data["frames"]] == [1.5, 2.0, 2.5]
    assert data["source_video"] == "source.mp4"
    assert data["video_sha256"] == video_digest(path.parent / "sampled.mp4")
    assert all((path.parent / row["rgb_path"]).exists() for row in data["frames"])
    with pytest.raises(FileExistsError):
        sample_video(video="source.mp4", output=path.parent)
    with pytest.raises(ValueError, match="out of range"):
        sample_video(video="source.mp4", output=tmp_path / "nan", fps=float("nan"))
