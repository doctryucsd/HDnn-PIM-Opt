from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from sim.flow import run
from sim.utils import Prof


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    profile: bool = cfg.get("profile", False)
    if profile:
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                skip_first=1, wait=0, warmup=0, active=1, repeat=10
            ),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(".log/"),
            with_stack=True,
            profile_memory=True,
            record_shapes=True,
        ) as profiler:
            Prof.set_profiler(profiler)
            run(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
