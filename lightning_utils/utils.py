import torch
import typing as th
import itertools


def args_list_len(*args):
    return max(len(arg) if isinstance(arg, (tuple, list)) else (1 if arg is not None else 0) for arg in args)


def list_args(*args, length: th.Optional[int] = None, return_length: bool = False):
    length = args_list_len(*args) if length is None else length
    if not length:
        results = args if len(args) > 1 else args[0]
        return results if not return_length else (results, length)
    results = [([arg] * length if not isinstance(arg, (tuple, list)) else arg) for arg in args]
    results = results if len(args) > 1 else results[0]
    return results if not return_length else (results, length)


def freeze_params(
    model: th.Optional[torch.nn.Module] = None,
    optimizer: th.Optional[torch.optim.Optimizer] = None,
):
    params = (
        model.parameters()
        if model is not None
        else itertools.chain(
            param for i in range(len(optimizer.param_groups)) for param in optimizer.param_groups[i]["params"]
        )
    )
    old_states = []
    for param in params:
        old_states.append(param.requires_grad)
        param.requires_grad = False
    return old_states


def unfreeze_params(
    model: th.Optional[torch.nn.Module] = None,
    optimizer: th.Optional[torch.optim.Optimizer] = None,
    old_states: th.Optional[th.List[bool]] = None,
):
    params = (
        model.parameters()
        if model is not None
        else itertools.chain(
            param for i in range(len(optimizer.param_groups)) for param in optimizer.param_groups[i]["params"]
        )
    )
    for idx, param in enumerate(params):
        param.requires_grad = True if old_states is None else old_states[idx]
