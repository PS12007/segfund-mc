import importlib.util

import pytest


def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("triton") is None:
        return False
    import torch

    return torch.cuda.is_available()


def pytest_collection_modifyitems(config, items):
    if _cuda_available():
        return
    skip = pytest.mark.skip(reason="needs CUDA + torch + triton")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
