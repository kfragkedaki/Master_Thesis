import logging
import os
import torch

from src.agents.train import train_epoch, validate
from src.nets.attention_model import AttentionModel
from src.utils import torch_load_cpu, get_baseline_model

from tensorboard_logger import Logger as TbLogger
from torch.utils.tensorboard import SummaryWriter


class TSPAgent:
    def __init__(
        self,
        opts,
        env,
    ):
        """
        The TSPAgent is used in companionship with the TSPEnv
        to solve the traveling salesman problem.

        Args:
            opts (int): Configurations
            env (TSPEnv): Environment for the TSP problem
        """
        self.opts = opts
        self.env = env

        # Load data from load_path
        assert (
                opts.load_path is None or opts.resume is None
        ), "Only one of load path and resume can be given"
        load_path = opts.load_path if opts.load_path is not None else opts.resume
        self.load_data = {} if load_path is None else torch_load_cpu(load_path)

        self.model = AttentionModel(
            embedding_dim=opts.embedding_dim,
            problem=env,
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization,
            tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder,
        ).to(opts.device)

        self.baseline_model = get_baseline_model(self.model, self.env, self.opts, self.load_data)

        self.optimizer = torch.optim.Adam(
            [{"params": self.model.parameters(), "lr": opts.lr_model}]
        )

        # Load optimizer state
        if "optimizer" in self.load_data:
            self.optimizer.load_state_dict(self.load_data["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)

        # Initialize learning rate scheduler, decay by lr_decay once per epoch!
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda epoch: opts.lr_decay ** epoch
        )

        # Optionally configure tensorboard logger and writer
        self.tb_logger = {
            "logger": None if opts.no_tensorboard else TbLogger(opts.save_dir + "/logs"),
            "writer": None
            if opts.no_tensorboard
            else SummaryWriter(opts.save_dir + "/plots"),
        }

    def train(
        self,
    ):
        """
        Trains the TSPAgent on an TSPEnvironment.

        Args:
            env: TSPEnv instance to train on
            epochs (int, optional): Amount of epochs to train. Defaults to 100.
            eval_epochs (int, optional): Amount of epochs to evaluate the current
                model against the baseline. Defaults to 1.
            check_point_dir (str, optional): Directiory that the checkpoints will
                be stored in. Defaults to "./check_points/".
        """
        logging.info("Start Training")

        # Start the actual training loop
        val_dataset = self.env.make_dataset(
            size=self.opts.graph_size,
            num_samples=self.opts.val_size,
            filename=self.opts.val_dataset,
            distribution=self.opts.data_distribution,
        )

        if self.opts.resume:
            epoch_resume = int(
                os.path.splitext(os.path.split(self.opts.resume)[-1])[0].split("-")[1]
            )

            torch.set_rng_state(self.load_data["rng_state"])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(self.load_data["cuda_rng_state"])
            # Set the random states
            # Dumping of state was done before epoch callback, so do that now (model is loaded)
            self.baseline_model.epoch_callback(self.model, epoch_resume)
            print("Resuming after {}".format(epoch_resume))
            self.opts.epoch_start = epoch_resume + 1

        if self.opts.eval_only:
            validate(self.model, val_dataset, self.opts)
        else:
            for epoch in range(self.opts.epoch_start, self.opts.epoch_start + self.opts.n_epochs):
                train_epoch(
                    self.model,
                    self.optimizer,
                    self.baseline_model,
                    self.lr_scheduler,
                    epoch,
                    val_dataset,
                    self.env,
                    self.tb_logger,
                    self.opts,
                )
