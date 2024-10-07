from __future__ import annotations

import torch


class Prof:
    profiler: torch.profiler.profile | None = None

    @classmethod
    def set_profiler(cls, prof: torch.profiler.profile) -> None:
        assert cls.profiler is None, "profiler is already set"
        cls.profiler = prof

    @classmethod
    def get_profiler(cls):
        assert cls.profiler is not None, "profiler is not set"
        return cls.profiler
