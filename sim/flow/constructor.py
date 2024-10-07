from typing import Any, Dict, List

from botorch.acquisition.input_constructors import acqf_input_constructor


@acqf_input_constructor(ScalarizedUpperConfidenceBound)
def construct_inputs_scalarized_ucb(
    model: Model,
    beta: float,
    weights: List[float],
    posterior_transform: None,
) -> Dict[str, Any]:
    return {
        "model": model,
        "beta": torch.as_tensor(beta, dtype=torch.double),
        "weights": torch.as_tensor(weights, dtype=torch.double),
    }


from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models

gs = GenerationStrategy(
    steps=[
        # Quasi-random initialization step
        GenerationStep(
            model=Models.SOBOL,
            num_trials=5,  # How many trials should be produced from this generation step
            model_kwargs={"seed": 999},  # Any kwargs you want passed into the model
        ),
        # Bayesian optimization step using the custom acquisition function
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,  # No limitation on how many trials should be produced from this step
            # For `BOTORCH_MODULAR`, we pass in kwargs to specify what surrogate or acquisition function to use.
            # `acquisition_options` specifies the set of additional arguments to pass into the input constructor.
            model_kwargs={
                "botorch_acqf_class": ScalarizedUpperConfidenceBound,
                "acquisition_options": {"beta": 0.1, "weights": [1.0, 1.0]},
            },
        ),
    ]
)

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.test_functions import BraninCurrin

# Initialize the client - AxClient offers a convenient API to control the experiment
ax_client = AxClient(generation_strategy=gs)
# Setup the experiment
ax_client.create_experiment(
    name="branincurrin_test_experiment",
    parameters=[
        {
            "name": f"x{i+1}",
            "type": "range",
            # It is crucial to use floats for the bounds, i.e., 0.0 rather than 0.
            # Otherwise, the parameter would
            "bounds": [0.0, 1.0],
        }
        for i in range(2)
    ],
    objectives={
        "branin": ObjectiveProperties(minimize=True),
        "currin": ObjectiveProperties(minimize=True),
    },
)
# Setup a function to evaluate the trials
branincurrin = BraninCurrin()


def evaluate(parameters):
    x = torch.tensor([[parameters.get(f"x{i+1}") for i in range(2)]])
    bc_eval = branincurrin(x).squeeze().tolist()
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {"branin": (bc_eval[0], 0.0), "currin": (bc_eval[1], 0.0)}
