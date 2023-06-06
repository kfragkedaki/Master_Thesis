import os
import csv
import json
import time
import argparse
import torch


def get_options(args=None):

    # Initialize Config
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning"
    )

    # Data
    parser.add_argument(
        "--problem",
        default="evrp",
        help="The problem to solve tsp, cvrp or evrp, default 'tsp'",
    )
    parser.add_argument(
        "--graph_size", type=int, default=10, help="The size of the problem graph"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Number of instances per batch during training",
    )
    parser.add_argument(
        "--epoch_size",
        type=int,
        default=512 * 25,
        help="Number of instances per epoch during training",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=1024 * 2,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default=None,
        help="Dataset file to use for validation",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Dimension of input embedding"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Dimension of hidden layers in Enc/Dec",
    )
    parser.add_argument(
        "--n_encode_layers",
        type=int,
        default=3,
        help="Number of layers in the encoder/critic network",
    )
    parser.add_argument(
        "--tanh_clipping",
        type=float,
        default=10.0,
        help="Clip the parameters to within +- this value using tanh. "
        "Set to 0 to not perform any clipping.",
    )
    parser.add_argument(
        "--normalization",
        default="batch",
        help="Normalization type, 'batch' (default) or 'instance'",
    )

    # Training
    parser.add_argument(
        "--lr_model",
        type=float,
        default=1e-4,
        help="Set the learning rate for the actor network",
    )
    parser.add_argument(
        "--lr_critic",
        type=float,
        default=1e-4,
        help="Set the learning rate for the critic network",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=1.0, help="Learning rate decay per epoch"
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="Set this value to only evaluate model"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=5, help="The number of epochs to train"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed to use")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--exp_beta",
        type=float,
        default=0.8,
        help="Exponential moving average baseline decay (default 0.8)",
    )
    parser.add_argument(
        "--baseline",
        default="rollout",
        help="Baseline to use: 'rollout' or 'exponential'. Defaults to no baseline.",
    )
    parser.add_argument(
        "--bl_alpha",
        type=float,
        default=0.05,
        help="Significance in the t-test for updating rollout baseline",
    )
    parser.add_argument(
        "--bl_warmup_epochs",
        type=int,
        default=None,
        help="Number of epochs to warmup the baseline, default None means 1 for rollout (exponential "
        "used for warmup phase), 0 otherwise. Can only be used with rollout baseline.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1024,
        help="Batch size to use during (baseline) evaluation",
    )
    parser.add_argument(
        "--checkpoint_encoder",
        action="store_true",
        help="Set to decrease memory usage by checkpointing encoder",
    )
    parser.add_argument(
        "--data_distribution",
        type=str,
        default=None,
        help="Data distribution to use during training, defaults and options depend on problem.",
    )
    parser.add_argument(
        "--num_trucks",
        type=int,
        default=2,
        help="The size of the fleet if we have the EVRP problem",
    )
    parser.add_argument(
        "--num_trailers",
        type=int,
        default=3,
        help="The number of the trailers of the EVRP problem.",
    )
    # Misc
    parser.add_argument(
        "--log_step", type=int, default=5, help="Log info every log_step steps"
    )
    parser.add_argument(
        "--run_name", default="rollout", help="Name to identify the run"
    )
    parser.add_argument(
        "--output_dir", default="outputs", help="Directory to write output models to"
    )
    parser.add_argument(
        "--epoch_start",
        type=int,
        default=0,
        help="Start at epoch # (relevant for learning rate decay)",
    )
    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        default=1,
        help="Save checkpoint every n epochs (default 1), 0 to save no checkpoints",
    )
    parser.add_argument(
        "--load_path", help="Path to load model parameters and optimizer state from"
    )
    parser.add_argument("--resume", help="Resume from previous checkpoint file")
    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable logging TensorBoard files",
    )
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )
    parser.add_argument(
        "--truck_names",
        type=str,
        default=None,
        help="Location of the csv file to use for truck names",
    )

    opts = parser.parse_args(args)
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name
    )

    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == "rollout" else 0

    # Configure outputs dir
    os.makedirs(opts.save_dir)

    with open(opts.save_dir + "/results", "w+", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Seed", "Mean Distance"])

    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    use_mps = torch.backends.mps.is_available() and not opts.no_cuda and False

    opts.device = torch.device("cuda:0" if use_cuda else "mps:0" if use_mps else "cpu")
    opts.dataParallel = (
        use_cuda and torch.cuda.device_count() > 1 or (use_mps and torch.has_mps)
    )

    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == "rollout")
    assert (
        opts.epoch_size % opts.batch_size == 0
    ), "Epoch size must be integer multiple of batch size!"

    return opts
