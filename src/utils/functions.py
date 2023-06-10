import torch
import os
import json
import torch.nn.functional as F
from torch.nn import DataParallel
import sys

def load_env(name: str):
    from src.problems import TSP, CVRP, EVRP

    problem = {
        "tsp": TSP,
        "cvrp": CVRP,
        "evrp": EVRP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def load_attention_model(name: str):
    from src.nets import AttentionTSPModel, AttentionVRPModel, AttentionEVRPModel

    model = {
        "tsp": AttentionTSPModel,
        "cvrp": AttentionVRPModel,
        "evrp": AttentionEVRPModel,
    }.get(name, None)
    assert model is not None, "Currently unsupported problem: {}!".format(name)
    return model


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def torch_load_cpu(load_path):
    return torch.load(
        load_path, map_location=lambda storage, loc: storage
    )  # Load on CPU


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print("  [*] Loading model from {}".format(load_path))

    load_data = torch.load(
        os.path.join(os.getcwd(), load_path), map_location=lambda storage, loc: storage
    )

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get("optimizer", None)
        load_model_state_dict = load_data.get("model", load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, "r") as f:
        args = json.load(f)

    # Backwards compatibility
    if "data_distribution" not in args:
        args["data_distribution"] = None
        probl, *dist = args["problem"].split("_")
        if probl == "op":
            args["problem"] = probl
            args["data_distribution"] = dist[0]
    return args


def load_model(path, epoch=None):
    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == ".pt"
            )
        model_filename = os.path.join(path, "epoch-{}.pt".format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, "args.json"))

    problem = load_env(args["problem"])

    model_class = load_attention_model(args["problem"])

    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
        embedding_dim=args["embedding_dim"],
        problem=problem,
        n_encode_layers=args["n_encode_layers"],
        mask_inner=True,
        mask_logits=True,
        normalization=args["normalization"],
        tanh_clipping=args["tanh_clipping"],
        checkpoint_encoder=args.get("checkpoint_encoder", False),
    )
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get("model", {})})

    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis], 1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def get_baseline_model(model, env, opts, load_data):
    from src.nets.reinforce_baselines import NoBaseline, WarmupBaseline, RolloutBaseline

    # Initialize baseline
    if opts.baseline == "rollout":
        baseline_model = RolloutBaseline(model, env, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline_model = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline_model = WarmupBaseline(
            baseline_model, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta
        )

    # Load baseline from data, make sure script is called with same type of baseline
    if "baseline" in load_data:
        baseline_model.load_state_dict(load_data["baseline"])

    return baseline_model
