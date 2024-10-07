from __future__ import annotations

from .metricArgs import MetricArgs
from .metricManager import MetricManager


def metric_manager_factory(typ: str, args: MetricArgs) -> MetricManager:
    if typ == "analytical":
        from .analytical.analytical import Analytical

        return Analytical(args)
    elif typ == "neurosim":
        from .neurosim.neurosim import NeuroSim

        return NeuroSim(args)
    elif typ == "cimloop":
        from .cimloop.cimloop import CIMLoop

        return CIMLoop(args)
    else:
        raise ValueError(f"Invalid metric manager type: {typ}")
