from __future__ import annotations
from typing import Any, Dict, List, Tuple

class Evaluator:
    def __init__(self, *args, **kwargs):
        # TODO: Initialize the evaluator with necessary parameters
        pass

    def get_optimizations_prop(self):
        # TODO: Define metric properties for optimization
        ret: List[Tuple[str, str, float]] = []

        ref_point_0: float = 0.0
        ref_point_1: float = 1.0
        ref_point_2: float = 0.0

        ret.append(("metric_0", "max", ref_point_0))
        ret.append(("metric_1", "min", ref_point_1))
        ret.append(("metric_2", "max", ref_point_2))
        return ret
    
    def evaluate(self, *args, **kwargs) -> Dict[str, Tuple[float, float]]:
        # TODO: Implement the evaluation logic
        metric_0_result = {"metric_0": (0.5, 0.1)}
        metric_1_result = {"metric_1": (0.8, 0.05)}
        metric_2_result = {"metric_2": (0.3, 0.02)}
        return {**metric_0_result, **metric_1_result, **metric_2_result}