#!/usr/bin/env python

import os
import pprint as pp

import torch
import torch.optim as optim

from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, RolloutBaseline
from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_env
from tensorboard_logger import Logger as TbLogger
from torch.utils.tensorboard import SummaryWriter


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard logger and writer
    tb_logger = {
        "logger": None if opts.no_tensorboard else TbLogger(opts.save_dir + "/logs"),
        "writer": None
        if opts.no_tensorboard
        else SummaryWriter(opts.save_dir + "/plots"),
    }

    # Initialize the Environment
    env = load_env(opts.problem)
    # env = load_env("tsp2")

    # Load data from load_path
    assert (
        opts.load_path is None or opts.resume is None
    ), "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    load_data = {} if load_path is None else torch_load_cpu(load_path)

    # baseline_model {
    model = AttentionModel(
        embedding_dim=opts.embedding_dim,
        problem=env,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
    ).to(opts.device)

    if opts.dataParallel:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get("model", {})})

    # Initialize baseline
    if opts.baseline == "exponential":
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == "rollout":
        baseline = RolloutBaseline(model, env, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    # Load baseline from data, make sure script is called with same type of baseline
    if "baseline" in load_data:
        baseline.load_state_dict(load_data["baseline"])

    # }

    # Initialize optimizer
    optimizer = optim.Adam(
        [{"params": model.parameters(), "lr": opts.lr_model}]
        + (
            [{"params": baseline.get_learnable_parameters(), "lr": opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if "optimizer" in load_data:
        optimizer.load_state_dict(load_data["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: opts.lr_decay**epoch
    )

    # Start the actual training loop
    val_dataset = env.make_dataset(
        size=opts.graph_size,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        distribution=opts.data_distribution,
    )

    if opts.resume:
        epoch_resume = int(
            os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1]
        )

        torch.set_rng_state(load_data["rng_state"])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data["cuda_rng_state"])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                env,
                tb_logger,
                opts,
            )


if __name__ == "__main__":
    run(get_options())
