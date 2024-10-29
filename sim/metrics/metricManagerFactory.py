from __future__ import annotations

from .metricArgs import MetricArgs
from .metricManager import MetricManager


def metric_manager_factory(
    typ: str, args: MetricArgs, device: str | int
) -> MetricManager:
    if typ == "cimloop":
        from .cimloop.cimloop import CiMLoop

        return CiMLoop(args, device)

    # FIXME: Implement the following classes
    # elif typ == "analytical":
    #     from .analytical.analytical import Analytical
    #     return Analytical(args, device)

    # elif typ == "neurosim":
    #     from .neurosim.neurosim import NeuroSim
    #     return NeuroSim(args, device)

    else:
        raise ValueError(f"Invalid metric manager type: {typ}")
