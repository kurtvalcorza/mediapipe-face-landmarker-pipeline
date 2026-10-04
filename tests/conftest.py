import builtins

import numpy as np
import pytest
from PIL import Image

MODEL_LIBRARIES = {"mediapipe", "cv2", "matplotlib", "sounddevice"}


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initialising model libraries (fleet RTM-001)."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in MODEL_LIBRARIES:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def synthetic_image(*, width: int = 320, height: int = 240, seed: int = 0, mode: str = "RGB") -> Image.Image:
    """A smooth, seeded image with no face in it."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack([np.sin(x / 40 + seed), np.cos(y / 30 - seed), np.sin((x + y) / 60)], axis=-1)
    array = ((base + rng.normal(0, 0.05, base.shape) + 1.5) / 3.0 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array).convert(mode)
