# Copyright Vahid Zehtab (vahid@zehtab.me) 2021
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import typing as th
import functools
import lightning
import dypy as dy
from dypy.core.functions import call_with_dynamic_args
from lightning_toolbox.objective_function import Objective
import torch
from .datastructures import ArgsListDict


class TrainingModule(lightning.LightningModule):
    def __init__(
        self,
        # model
        model: th.Optional[torch.nn.Module] = None,  # model instance
        model_cls: th.Optional[str] = None,
        model_args: th.Optional[dict] = None,
        # objective
        objective: th.Optional[Objective] = None,  # objective instance
        objective_cls: th.Union[type, str] = "lightning_toolbox.Objective",
        objective_args: th.Optional[dict] = None,
        # optimization configs
        # optimizer name or class
        optimizer: th.Union[str, type, th.List[th.Union[str, type]], None] = None,
        optimizer_frequency: th.Union[int, th.List[th.Optional[int]], None] = None,
        optimizer_is_active: th.Optional[th.Union[dy.FunctionDescriptor, th.List[dy.FunctionDescriptor]]] = None,
        # optimizer parameters (self.<*>)
        optimizer_parameters: th.Optional[th.Union[th.List[str], str]] = None,
        optimizer_args: th.Optional[dict] = None,
        # learning rate
        lr: th.Union[th.List[float], float] = 1e-4,
        # schedulers
        # scheduler name or class
        scheduler: th.Optional[th.Union[str, type, th.List[th.Union[str, type]]]] = None,
        scheduler_name: th.Optional[th.Union[str, th.List[str]]] = None,
        # optimizer index
        scheduler_optimizer: th.Optional[th.Union[int, th.List[int]]] = None,
        scheduler_args: th.Optional[th.Union[dict, th.List[dict]]] = None,
        scheduler_interval: th.Union[str, th.List[str]] = "epoch",
        scheduler_frequency: th.Union[int, th.List[int]] = 1,
        scheduler_monitor: th.Optional[th.Union[str, th.List[str]]] = None,
        scheduler_strict: th.Optional[th.Union[bool, th.List[bool]]] = None,
        # initialization settings
        save_hparams: bool = True,
        initialize_superclass: bool = True,
    ) -> None:
        if initialize_superclass:
            super().__init__()
        if save_hparams:
            self.save_hyperparameters(ignore=["model"])

        # objective
        if objective is not None or objective_cls is not None:
            self.objective = (
                objective if objective is not None else dy.eval(objective_cls)(**(objective_args or dict()))
            )

        # optimizers
        self.optimizers_list = ArgsListDict(
            optimizer=optimizer,
            is_active=optimizer_is_active,
            parameters=optimizer_parameters,
            args=optimizer_args,
            frequency=optimizer_frequency,
        )
        self.optimizers_list = None if self.optimizers_list["optimizer"] is None else self.optimizers_list

        # learning rate
        self.lr = lr  # populate when configure_optimizers is called

        # learning rate schedulers
        self.schedulers_list = ArgsListDict(
            scheduler=scheduler,
            name=scheduler_name,
            optimizer=scheduler_optimizer,
            args=scheduler_args,
            interval=scheduler_interval,
            frequency=scheduler_frequency,
            monitor=scheduler_monitor,
            strict=scheduler_strict,
        )
        self.schedulers_list = None if self.schedulers_list.value(0, "scheduler") is None else self.schedulers_list

        # cross-populate schedulers (if we have schedulers) if scheduler descriptions have no optimizer specified
        if self.schedulers_list is not None and all(sched["optimizer"] is None for sched in self.schedulers_list):
            # cross-populating schedulers repeats the same scheduler description for each optimizer
            schedulers_list = self.schedulers_list.cross_populate(length=self.optimizers_list.args_length)
            # set scheduler optimizers
            schedulers_list["optimizer"] = [
                i for j in range(self.schedulers_list.args_length) for i in range(self.optimizers_list.args_length)
            ]
            self.schedulers_list = schedulers_list

        # process schedulers names
        if self.schedulers_list is not None:
            self.schedulers_list["name"] = (
                self.schedulers_list["name"]
                if isinstance(self.schedulers_list["name"], list)
                else ([self.schedulers_list["name"]] * self.schedulers_list.args_length)
            )
            for idx, sched in enumerate(self.schedulers_list):
                param_name = self.optimizers_list.value(sched["optimizer"], "parameters")
                param_name = (
                    f"/{param_name}" if param_name and isinstance(param_name, str) else f"/{sched['optimizer']}"
                )
                self.schedulers_list["name"][
                    idx
                ] = f"lr_scheduler{param_name}/{sched['name'] if sched['name'] is not None else idx}"

        # initialize the model
        if model is not None or model_args is not None or model_cls is not None:
            self.model = model if model is not None else dy.eval(model_cls)(**(model_args or dict()))

    @functools.cached_property
    def __optimizers_is_active_list(self):
        if self.optimizers_list is None:
            return
        if self.optimizers_list["is_active"] is None:
            return [True for i in range(len(self.optimizers_list["optimizer"]))]
        return [
            dy.eval(optimizer["is_active"], function_of_interest="is_active", dynamic_args=True, strict=False)
            for optimizer in self.optimizers_list
        ]

    def is_optimizer_active(self, optimizer_idx: int, batch_idx: int, epoch: int) -> bool:
        if optimizer_idx is None:
            return True
        is_active = self.__optimizers_is_active_list[optimizer_idx]
        result = is_active is None or (isinstance(is_active, bool) and is_active)
        if callable(is_active):
            result = is_active(training_module=self, optimizer_idx=optimizer_idx, batch_idx=batch_idx, epoch=epoch)
        return result

    def remember(self, **kwargs: th.Dict[str, th.Any]) -> None:
        """
        You can use this function to remember any particular object that you want to
        use later on.

        **Note**: The values that are remembered will be forgotten soon after every batch
        so make sure to save them in another logging mechanism if you want to keep them.
        Typically, this is implemented using a LoggingCallback.
        """
        self.objective.remember(**kwargs)

    def forward(self, *args, **kwargs):
        "Placeholder forward pass for the model"
        if hasattr(self, "model") and self.model is not None:
            # pass the training module to the model so that it can access the objective
            # or any other attribute of the training module that might be needed
            return call_with_dynamic_args(self.model.forward, *args, **kwargs, training_module=self)
        raise NotImplementedError("No model defined")

    def step(
        self,
        batch: th.Optional[th.Any] = None,
        batch_idx: th.Optional[int] = None,
        optimizer_idx: th.Optional[int] = None,
        name: str = "train",
        return_results: bool = False,
        return_factors: bool = False,
        log_results: bool = True,
        **kwargs,  # additional arguments to pass to the objective
    ):
        """Train or evaluate the model with the given batch.

        Args:
            batch: batch of data to train or evaluate with
            batch_idx: index of the batch
            optimizer_idx: index of the optimizer
            name: name of the step ("train" or "val")

        Returns:
            None if the model is in evaluation mode, else a tensor with the training objective
        """
        is_val = name == "val"
        assert hasattr(self, "objective"), "No objective defined"

        if not is_val and not self.is_optimizer_active(
            optimizer_idx=optimizer_idx, batch_idx=batch_idx, epoch=self.current_epoch
        ):
            return None  # this will skip this step, but the optimizer position will be incremented
            # TODO: check if how this affects the schedulers

        results, factors = self.objective(
            batch=batch,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
            training_module=self,
            return_factors=True,
            **kwargs,
        )
        if log_results:
            self.log_step_results(results, factors, name)
        if return_results:
            return (results, factors) if return_factors else results
        return results["loss"] if not is_val else None

    def configure_optimizers(self):
        """Initialize the optimizers and learning rate schedulers."""
        if self.optimizers_list is None:
            # no optimizers specified
            return
        optimizers, schedulers = [], []
        # populate learning rates for each optimizer
        learning_rates = [i["lr"] for i in ArgsListDict(lr=self.lr, length=self.optimizers_list.args_length)]

        # initialize the optimizers
        optimizers_using_frequency = False
        for optimizer, lr in zip(self.optimizers_list, learning_rates):
            opt_args = {"lr": lr, **(optimizer["args"] if optimizer["args"] is not None else {})}
            params = dy.eval(optimizer["parameters"], context=self) if optimizer["parameters"] else self
            if isinstance(params, torch.Tensor):
                params = [params]
            else:
                params = params.parameters() if hasattr(params, "parameters") else params

            opt = dy.eval(optimizer["optimizer"])(params, **opt_args)
            if optimizer["frequency"] is None:
                if optimizers_using_frequency:
                    raise ValueError(
                        "You cannot mix optimizers with and without frequency. "
                        "Either specify frequency for all optimizers or none."
                    )
                optimizers.append(opt)
            else:
                optimizers_using_frequency = True
                optimizers.append(dict(optimizer=opt, frequency=optimizer["frequency"]))

        for sched in self.schedulers_list or []:
            optim = optimizers[sched["optimizer"]]
            scheduler = dy.eval(sched["scheduler"])(
                optim["optimizer"] if isinstance(optim, dict) else optim, **(sched["args"] or dict())
            )
            # DEBUG: wrap scheduler step to check if it is called

            # scheduler.step = decorator(scheduler.step, sched["name"])
            # END DEBUG
            instance = dict()
            for key, value in sched.items():
                if key in ["freqency", "monitor", "strict", "name", "interval"] and value is not None:
                    instance[key] = value
            if optimizers_using_frequency:
                if "lr_scheduler" in optim:
                    raise ValueError(
                        "you cannot have multiple schedulers for the same optimizer, when using optimizer frequency "
                    )  # TODO: use pytorch's ChainedScheduler to chain multiple non-conflicting schedulers
                optim["lr_scheduler"] = dict(scheduler=scheduler, **instance) if instance else scheduler
            else:
                schedulers.append(dict(scheduler=scheduler, **instance) if instance else scheduler)

        if schedulers:
            if len(schedulers) == 1 and len(optimizers) == 1:
                return (
                    dict(optimizer=optimizers[0], lr_scheduler=schedulers[0])
                    if not isinstance(optimizers[0], dict)
                    else dict(**optimizers[0], lr_scheduler=schedulers[0])
                )
            return optimizers, schedulers
        return optimizers

    def log_step_results(self, results, factors, name: str = "train"):
        "Log the results of the step"
        is_val = name == "val"
        # logging results
        for item, value in results.items():
            self.log(
                f"{item}/{name}",
                value.mean() if isinstance(value, torch.Tensor) else value,
                on_step=not is_val,
                on_epoch=is_val,
                logger=True,
                sync_dist=True,
                prog_bar=is_val and name == "loss",
            )
        # validation step only logs
        if not is_val:
            return

        # logging factors
        for item, value in factors.items():
            self.log(
                f"factors/{item}/{name}",
                value.mean() if isinstance(value, torch.Tensor) else value,
                on_step=not is_val,
                on_epoch=is_val,
                logger=True,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx, optimizer_idx=None, **kwargs):
        return self.step(batch, batch_idx, optimizer_idx, name="train", **kwargs)

    def validation_step(self, batch, batch_idx, **kwargs):
        return self.step(batch, batch_idx, name="val", **kwargs)
