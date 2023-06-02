#!/usr/bin/env python

import torch
import pprint as pp

from src.options import get_options
from src.utils import load_env
from src.agents import Agent


def run(opts: dict()):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Initialize the Environment
    env = load_env(opts.problem)
    # env = load_env("tsp2")

    Agent(opts, env).train()


if __name__ == "__main__":
    run(get_options())
