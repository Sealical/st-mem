from __future__ import annotations

import os
import subprocess
import sys
from contextlib import nullcontext
from types import ModuleType

import numpy as np
import pytest

from stmem.perception.qwen import Qwen3VLModel


def test_qwen3_vl_constructs_without_loading() -> None:
    model = Qwen3VLModel(checkpoint="unused", device="cpu")
    assert model.network is None
    assert model.processor is None


def test_qwen3_vl_unknown_op_before_load() -> None:
    model = Qwen3VLModel(checkpoint="unused", device="cpu")
    with pytest.raises(KeyError, match="unknown vlm op"):
        model.forward({"op": "nope", "prompt": "hi"})
    assert model.network is None


def test_qwen3_vl_requires_messages_or_image_before_load() -> None:
    model = Qwen3VLModel(checkpoint="unused", device="cpu")
    with pytest.raises(KeyError, match="messages.*image"):
        model.forward({"op": "generate", "prompt": "hi"})
    assert model.network is None


def test_qwen3_vl_forward_contract_with_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = ModuleType("torch")
    torch.inference_mode = nullcontext  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    class FakeNetwork:
        def __init__(self) -> None:
            self.max_new_tokens: int | None = None

        def generate(self, **kwargs: object) -> np.ndarray:
            self.max_new_tokens = int(kwargs.pop("max_new_tokens"))
            assert set(kwargs) == {"input_ids", "pixel_values"}
            return np.array([[10, 11, 12, 42, 43]], dtype=np.int64)

    class FakeBatch(dict[str, object]):
        moved_to: str | None = None

        def to(self, device: str) -> FakeBatch:
            self.moved_to = device
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] | None = None
            self.template_kwargs: dict[str, object] | None = None
            self.batch: FakeBatch | None = None

        def apply_chat_template(
            self,
            messages: list[dict[str, object]],
            **kwargs: object,
        ) -> FakeBatch:
            self.messages = messages
            self.template_kwargs = kwargs
            self.batch = FakeBatch(
                input_ids=np.array([[1, 2, 3]], dtype=np.int64),
                pixel_values="pixels",
            )
            return self.batch

        def batch_decode(self, tokens: list[np.ndarray], **kwargs: object) -> list[str]:
            assert [token.tolist() for token in tokens] == [[42, 43]]
            assert kwargs == {
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            }
            return ["a chair"]

    network = FakeNetwork()
    processor = FakeProcessor()
    model = Qwen3VLModel(checkpoint="unused", device="cuda:1", max_new_tokens=128)
    model.network = network
    model.processor = processor

    result = model.forward(
        {
            "op": "generate",
            "image": np.zeros((2, 3, 3), dtype=np.uint8),
            "prompt": "What is shown?",
            "max_new_tokens": 7,
        }
    )

    assert result == {"text": "a chair"}
    assert network.max_new_tokens == 7
    assert processor.messages is not None
    content = processor.messages[0]["content"]
    assert isinstance(content, list)
    assert content[1] == {"type": "text", "text": "What is shown?"}
    assert processor.template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert processor.batch is not None
    assert processor.batch.moved_to == "cuda:1"


def test_qwen3_vl_missing_key_does_not_import_heavy_runtime(tmp_path) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    script = r"""
import sys
from stmem.perception.qwen import Qwen3VLModel

assert "torch" not in sys.modules
assert "transformers" not in sys.modules
model = Qwen3VLModel(checkpoint="unused", device="cpu")
try:
    model.forward({"op": "invalid"})
except KeyError as exc:
    assert "unknown vlm op" in str(exc)
else:
    raise AssertionError("unknown op did not fail")
try:
    model.forward({"op": "generate", "prompt": "hi"})
except KeyError as exc:
    assert "messages" in str(exc) and "image" in str(exc)
else:
    raise AssertionError("missing messages/image did not fail")
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
assert model.network is None
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
