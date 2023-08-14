import logging
import os
import torch

from src.agents.train import train_epoch, validate
from src.utils import (
    torch_load_cpu,
    get_baseline_model,
    load_attention_model,
    load_optimizers,
    EarlyStopping,
)
from src.utils.hyperparameter_config import config

from tensorboard_logger import Logger as TbLogger
from torch.utils.tensorboard import SummaryWriter
from ray import tune


class Agent:
    def __init__(
        self,
        opts,
        env,
        session=None,
    ):
        """
        The Agent is used in companionship with the problem Environment
        to solve the TSP/VRP/EVRP problem.

        Args:
            opts (int): Configurations
            env (Env): Environment for the problem
        """
        self.opts = opts
        self.env = env
        self.session = session

        # Load data from load_path
        assert (
            opts.load_path is None or opts.resume is None
        ), "Only one of load path and resume can be given"
        load_path = opts.load_path if opts.load_path is not None else opts.resume
        self.load_data = {} if load_path is None else torch_load_cpu(load_path)

        self.model = load_attention_model(env.NAME)(
            embedding_dim=opts.embedding_dim,
            problem=self.env,
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization,
            tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder,
            opts=opts,
        ).to(opts.device)

        self.baseline_model = get_baseline_model(
            self.model, self.env, self.opts, self.load_data
        )

        self.optimizer = load_optimizers(opts.optimizer_class)(
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
            self.optimizer, lambda epoch: opts.lr_decay**epoch
        )

        # Optionally configure tensorboard logger and writer
        self.tb_logger = {
            "logger": None
            if opts.no_tensorboard
            else TbLogger(f"{opts.save_dir}/logs"),
            "writer": None
            if opts.no_tensorboard
            else SummaryWriter(log_dir=f"{opts.save_dir}/plots"),
            "ray": None
            if not opts.hyperparameter_tuning
            else SummaryWriter(log_dir=f"{opts.save_dir}/{session.get_trial_id()}"),
        }

        self.early_stopping = EarlyStopping(
            patience=opts.early_stopping_patience, delta=opts.early_stopping_delta
        )

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
            num_trucks=self.opts.num_trucks,
            num_trailers=self.opts.num_trailers,
            truck_names=self.opts.truck_names,
            display_graphs=self.opts.display_graphs,
            r_threshold=self.opts.battery_limit,
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
            for epoch in range(
                self.opts.epoch_start, self.opts.epoch_start + self.opts.n_epochs
            ):
                loss, model = train_epoch(
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

                if self.opts.hyperparameter_tuning:
                    tune.report(loss=loss.item())

                if self.early_stopping(loss.item()):
                    print(f"Early Stopping, epoch {epoch}")
                    break

            if self.opts.hyperparameter_tuning:
                self.tb_logger["ray"].add_hparams(
                    hparam_dict={
                        k: v for k, v in vars(self.opts).items() if k in config.keys()
                    },
                    metric_dict={"loss": loss.item()},
                    run_name=f"{self.opts.save_dir}/{self.session.get_trial_id()}",
                )
                self.tb_logger["ray"].close()
                torch.save(model, self.session.get_trial_dir() + "/model.pt")

            if not self.opts.no_tensorboard:
                self.tb_logger["writer"].close()
