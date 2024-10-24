from __future__ import annotations
from typing import Any, Dict, List, Tuple
from ..metrics.metricArgs import MetricArgs
from ..metrics.metricManagerFactory import metric_manager_factory
from ..metrics.metricManager import MetricManager
from logging import Logger
import multiprocessing as mp

class Evaluator:
    def __init__(self,
        typ: str,
        metric_args: MetricArgs,
    ) -> None:
        devices: List[str] = metric_args.hardware_args["devices"]

        # evaluator
        self.metric_managers: List[MetricManager] = [metric_manager_factory(typ, metric_args, device) for device in devices]

    def evaluate(self, params: List[Dict[str, Any]], logger: Logger) -> List[Dict[str, Any]]:
        # evaluate each evaluator in parallel
        with mp.Pool() as pool:
            results: List[Dict[str, Any]] = pool.starmap(self._evaluate, [(params, logger, evaluator) for evaluator in self.metric_managers])

        return results
    
    def _evaluate(self, params: Dict[str, Any], logger: Logger, evaluator: MetricManager) -> Dict[str, Any]:
        return evaluator.evaluate(params, logger)
    
    def get_optimizations_prop(self) -> List[Tuple[str, str, float]]:
        return self.metric_managers[0].get_optimizations_prop()