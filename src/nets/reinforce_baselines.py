import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
import copy
from src.agents.train import rollout
from src.utils import get_inner_model


class Baseline(object):
    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch[0], batch[1], None

    def eval(self, x, c, graph_batch):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class WarmupBaseline(Baseline):
    def __init__(
        self,
        baseline,
        n_epochs=1,
        warmup_exp_beta=0.8,
    ):
        super(Baseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c, graph_batch):
        if self.alpha == 1:
            return self.baseline.eval(x, c, graph_batch)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c, graph_batch)
        v, l = self.baseline.eval(x, c, graph_batch)
        vw, lw = self.warmup_baseline.eval(x, c, graph_batch)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (
            1 - self.alpha * lw
        )

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        self.alpha = (epoch + 1) / float(self.n_epochs)
        if epoch < self.n_epochs:
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class NoBaseline(Baseline):
    def eval(self, x, c, graph_batch):
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):
    def __init__(self, beta):
        super(Baseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, c, graph_batch):
        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1.0 - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        return {"v": self.v}

    def load_state_dict(self, state_dict):
        self.v = state_dict["v"]


class CriticBaseline(Baseline):
    def __init__(self, critic):
        super(Baseline, self).__init__()

        self.critic = critic

    def eval(self, x, c, graph_batch):
        v = self.critic(x)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v, c.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {"critic": self.critic.state_dict()}

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get("critic", {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


class RolloutBaseline(Baseline):
    def __init__(self, model, env, opts, epoch=0):
        super(Baseline, self).__init__()

        self.env = env
        self.opts = opts

        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset
        # assert dataset is not None, "No dataset provided"
        if dataset is not None:
            if len(dataset) != self.opts.val_size:
                print(
                    "Warning: not using saved baseline dataset since val_size does not match"
                )
                dataset = None
            elif (dataset[0] if self.env.NAME == "tsp" else dataset[0]["loc"]).size(
                0
            ) != self.opts.graph_size:
                print(
                    "Warning: not using saved baseline dataset since graph_size does not match"
                )
                dataset = None

        if dataset is None:
            self.dataset = self.env.make_dataset(
                size=self.opts.graph_size,
                num_samples=self.opts.val_size,
                distribution=self.opts.data_distribution,
                num_trucks=self.opts.num_trucks,
                num_trailers=self.opts.num_trailers,
                truck_names=self.opts.truck_names,
                display_graphs=self.opts.display_graphs,
                r_threshold=self.opts.battery_limit,
            )
        else:
            self.dataset = dataset
        print("Evaluating baseline model on evaluation dataset")
        self.bl_vals = (
            rollout(self.model, self.dataset, self.opts, epoch, type="baseline")[0]
            .cpu()
            .numpy()
        )  # save cost = length + penalty - reward
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        return BaselineDataset(
            dataset,
            rollout(self.model, dataset, self.opts, epoch=self.epoch, type="baseline")[
                0
            ].view(-1, 1),
        )

    def unwrap_batch(self, batch):
        return (
            batch[0],
            batch[1],
            batch[2].view(-1),
        )  # Flatten result to undo wrapping as 2D

    def eval(self, x, c, graph_batch):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.model(x, graphs=graph_batch, type="eval")

        # There is no loss
        return v, 0

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        print("Evaluating candidate model on evaluation dataset")
        self.opts.display_graphs = (
            None  # data in graphs have changed from the baseline, we need to reset
        )
        candidate_vals = (
            rollout(model, self.dataset, self.opts, epoch=epoch, type="evaluation")[0]
            .cpu()
            .numpy()
        )

        candidate_mean = candidate_vals.mean()

        print(
            "Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
                epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean
            )
        )
        if candidate_mean - self.mean < 0:
            # Calc p value
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.opts.bl_alpha:
                print("Update baseline")
                self._update_model(model, epoch)

    def state_dict(self):
        return {"model": self.model, "dataset": self.dataset, "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(
            get_inner_model(state_dict["model"]).state_dict()
        )
        self._update_model(load_model, state_dict["epoch"], state_dict["dataset"])


class BaselineDataset(Dataset):
    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.baseline = baseline
        assert len(self.dataset) == len(self.baseline)

    def __getitem__(self, item):
        return {
            "data": self.dataset[item][0],
            "graphs": self.dataset[item][1],
            "baseline": self.baseline[item],
        }

    def __len__(self):
        return len(self.dataset)
