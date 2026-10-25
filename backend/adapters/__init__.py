from __future__ import annotations

from typing import Optional

from .gpt2 import Gpt2Adapter

_REGISTRY = {
    "gpt2": Gpt2Adapter,
}


def get_adapter(family: str) -> Optional[Gpt2Adapter]:
    cls = _REGISTRY.get(family)
    if cls is None:
        return None
    return cls()


def supported_families() -> list[str]:
    return sorted(_REGISTRY.keys())
