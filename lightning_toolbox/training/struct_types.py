import typing as th
import torch
import dypy as dy


# types
ResultsDict = th.Dict[str, torch.Tensor]
FactorsDict = th.Dict[str, torch.Tensor]

TermDescriptor = th.Union[str, th.Dict[str, th.Any], dy.FunctionDescriptor]
