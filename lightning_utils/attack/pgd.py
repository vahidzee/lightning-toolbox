import torch
import functools
import typing as th
from torchde.utils import process_function_description, FunctionDescriptor
from .utils import freeze_params, unfreeze_params
from .criterion import Criterion


class PGDAttacker:
    def __init__(
        self,
        criterion: th.Optional[Criterion] = None,
        criterion_roi: th.Union[str, th.List[str]] = "loss",
        num_iters: int = 1,
        alpha: float = 0.01,
        epsilon: th.Optional[float] = None,
        p_norm: str = "inf",
        inputs_clamp: FunctionDescriptor = None,
        random_start: bool = True,
        random_start_generation: FunctionDescriptor = "torch.randn_like",
    ):
        self.criterion = criterion
        self.criterion_roi = criterion_roi
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.alpha = alpha
        self.random_start = random_start
        self.p_norm = p_norm
        self.random_start_generation_function = random_start_generation
        self.inputs_clamp_function = inputs_clamp

    @functools.cached_property
    def generate_random_start(self):
        return process_function_description(self.random_start_generation_function, entry_function="process")

    @functools.cached_property
    def clamp_inputs(self):
        if self.inputs_clamp_function is None:
            return lambda inputs: inputs
        return process_function_description(self.inputs_clamp_function, entry_function="process")

    def objective(
        self,
        *args,
        inputs,
        model: th.Optional[torch.nn.Module] = None,
        training_module: th.Optional["pl.LightningModule"] = None,
        **kwargs,
    ):
        if model is None:
            model = training_module.model if hasattr(training_module, "model") else training_module
        criterion_results = (self.criterion or training_module.criterion)(
            *args, inputs=inputs, **kwargs, training_module=training_module, return_factors=False, model=model
        )
        if isinstance(self.criterion_roi, str):
            return criterion_results[self.criterion_roi].mean()
        return sum(criterion_results[name] for name in self.criterion_roi).mean()

    @classmethod
    def renorm_adversary(
        cls,
        adv_inputs,
        epsilon: th.Optional[float] = None,
        p_norm: th.Optional[th.Union[str, int]] = "inf",
    ):
        if epsilon is None:
            return adv_inputs
        if p_norm == "inf":
            return adv_inputs.clamp(-epsilon, epsilon)
        norms = torch.norm(
            adv_inputs,
            dim=torch.arange(len(adv_inputs.shape))[1:].tolist(),
            p=p_norm,
            keepdim=True,
        )
        return torch.where(norms > epsilon, adv_inputs * epsilon / norms, adv_inputs)

    def __call__(
        self,
        *args,
        inputs,
        return_loss=True,
        force_eval: bool = True,
        model: th.Optional[torch.nn.Module] = None,
        training_module: th.Optional["pl.LightningModule"] = None,
        **kwargs,
    ):
        torch.set_grad_enabled(True)
        if self.random_start:
            delta = self.generate_random_start(inputs)
            delta.requires_grad = True
        else:
            delta = torch.zeros_like(inputs, requires_grad=True)

        # freezing model params
        model = model if model is not None else training_module
        if model is not None:
            params_state = freeze_params(model)
        # force eval
        model_training = model.training
        if force_eval:
            model.eval()

        for i in range(self.num_iters):
            loss = self.objective(
                *args, inputs=inputs.detach() + delta, **kwargs, training_module=training_module, model=model
            )
            if i == 0 and return_loss:
                init_loss = loss.detach()
            loss.backward()
            delta.data.copy_(delta.data + self.alpha * delta.grad.detach().sign())
            delta.data.copy_(self.renorm_adversary(adv_inputs=delta, epsilon=self.epsilon, p_norm=self.p_norm))
            delta.grad.zero_()
        if return_loss:
            final_loss = self.objective(
                *args, inputs=inputs.detach() + delta, **kwargs, training_module=training_module, model=model
            ).detach()

        # unfreezing model
        if model is not None:
            unfreeze_params(model, params_state)
        # unforce eval
        if model_training:
            model.train()
        if return_loss:
            return (
                self.clamp_inputs(inputs + delta.detach()),
                init_loss,
                final_loss,
            )
        return self.clamp_inputs(inputs + delta.detach())
