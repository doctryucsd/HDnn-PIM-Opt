from __future__ import annotations

import os
import sys
from typing import Any

# add the path of acquisition functions to sys.path
sys.path.append(os.getcwd() + "/sim/flow/pesmoc/spearmint/acquisition_functions")


def acqf_factory(acqf: str) -> Any:
    from botorch.acquisition.multi_objective import MOMF  # multi-fidelity
    from botorch.acquisition.multi_objective import (
        MultiObjectiveMCAcquisitionFunction,
        qExpectedHypervolumeImprovement,
        qNoisyExpectedHypervolumeImprovement,
    )
    from botorch.acquisition.preference import (
        AnalyticExpectedUtilityOfBestOption,
    )  # preference based

    if acqf == "qExpectedHypervolumeImprovement":
        botorch_acqf_class = qExpectedHypervolumeImprovement
    elif acqf == "qNoisyExpectedHypervolumeImprovement":
        botorch_acqf_class = qNoisyExpectedHypervolumeImprovement
    elif acqf == "MultiObjectiveMCAcquisitionFunction":
        botorch_acqf_class = MultiObjectiveMCAcquisitionFunction
    elif acqf == "EUBO":
        botorch_acqf_class = AnalyticExpectedUtilityOfBestOption
    elif acqf == "MOMF":
        botorch_acqf_class = MOMF  # multi-fidelity to tryout if we have time in the end

    else:
        raise ValueError(f"Unknown acqf: {acqf}")

    return botorch_acqf_class
