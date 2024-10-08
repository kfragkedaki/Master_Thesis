import torch
import pprint as pp

from src.options import get_options
from src.utils import load_env
from src.agents import Agent
from src.utils.hyperparameter_config import config

import ray
from ray import tune, air
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import session


def run(config: dict()):
    # Pretty print the run args
    config["output_dir"] = "runs"
    args_list = [f"--{k}={v}" for k, v in config.items()]
    args_list.append("--no_tensorboard")
    args_list.append("--no_cuda")
    opts = get_options(args_list)

    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Initialize the Environment
    env = load_env(opts.problem)

    # Train the Agent
    agent = Agent(opts, env, session)
    agent.train()


if __name__ == "__main__":
    N_ITER = 20
    ray.init(num_cpus=3)

    current_best_params = [
        {
            "n_encode_layers": 3,
            "lr_model": 0.0001,
            "batch_size": 512,
            "n_epochs": 50,
            "optimizer_class": "NAdam",
        }
    ]

    searcher = HyperOptSearch(
        space=config,
        metric="loss",
        mode="min",
        n_initial_points=int(N_ITER / 10),
        points_to_evaluate=current_best_params,
    )

    algo = ConcurrencyLimiter(searcher, max_concurrent=3)

    objective = tune.with_resources(
        tune.with_parameters(run), resources={"cpu": 1, "memory": 400 * 1000000}
    )

    tuner = tune.Tuner(
        trainable=objective,
        run_config=air.RunConfig(storage_path="./ray_results"),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=algo,
            num_samples=N_ITER,
        ),
    )
    results = tuner.fit()
    ray.shutdown()
