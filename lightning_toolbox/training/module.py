import torch
import typing as th
import functools
import lightning
from ..utils import freeze_params, unfreeze_params, list_args
import dycode as dy
from .criterion import Criterion
import torch
import types


class TrainingModule(lightning.LightningModule):
    """
    Generic Lightning Module for training MADE models.

    Attributes:
        model: the model to train
        criterion: the criterion to use for training
        batch_transform: the transform to apply to the inputs before forward pass
    """

    def __init__(
        self,
        # model
        model: th.Optional[torch.nn.Module] = None,
        model_cls: th.Optional[str] = None,
        model_args: th.Optional[dict] = None,
        # criterion
        criterion: th.Union[Criterion, str] = "lightning_toolbox.Criterion",
        criterion_args: th.Optional[dict] = None,
        # input transforms [transform(inputs) -> torch.Tensor]
        batch_transform: th.Optional[dy.FunctionDescriptor] = None,
        # optimization configs [is_active(training_module, optimizer_idx) -> bool]
        optimizer: th.Union[str, th.List[str]] = "torch.optim.Adam",
        optimizer_is_active: th.Optional[th.Union[dy.FunctionDescriptor, th.List[dy.FunctionDescriptor]]] = None,
        optimizer_parameters: th.Optional[th.Union[th.List[str], str]] = None,
        optimizer_args: th.Optional[dict] = None,
        # learning rate
        lr: th.Union[th.List[float], float] = 1e-4,
        # schedulers
        scheduler: th.Optional[th.Union[str, th.List[str]]] = None,
        scheduler_name: th.Optional[th.Union[str, th.List[str]]] = None,
        scheduler_optimizer: th.Optional[th.Union[int, th.List[int]]] = None,
        scheduler_args: th.Optional[th.Union[dict, th.List[dict]]] = None,
        scheduler_interval: th.Union[str, th.List[str]] = "epoch",
        scheduler_frequency: th.Union[int, th.List[int]] = 1,
        scheduler_monitor: th.Optional[th.Union[str, th.List[str]]] = None,
        # initialization settings
        save_hparams: bool = True,
        initialize_superclass: bool = True,
    ) -> None:
        """Initialize the trainer.

        Args:
            model_cls: the class of the model to use (import path)
            model_args: the arguments to pass to the model constructor
            criterion_args: the arguments to pass to the criterion constructor
            attack_args: the arguments to pass to the attacker constructor (PGDAttacker)
            inputs_transform:
                the transform function to apply to the inputs before forward pass, can be used for
                applying dequantizations.

        Returns:
            None
        """
        if initialize_superclass:
            super().__init__()
        if save_hparams:
            self.save_hyperparameters(ignore=["model"])
        # criterion and attacks can be different from the checkpointed model
        criterion = criterion if criterion is not None else self.hparams.criterion
        criterion = dy.eval(criterion) if isinstance(criterion, str) else criterion
        self.criterion = (
            criterion
            if isinstance(criterion, Criterion)
            else criterion(**{**(self.hparams.criterion_args or dict()), **(criterion_args or dict())})
        )

        self.batch_transform = batch_transform if batch_transform is not None else self.hparams.batch_transform

        # optimizers and schedulers
        if (optimizer if optimizer is not None else self.hparams.optimizer) is not None:
            # optimizers
            (
                self.optimizer,
                self.optimizer_is_active_descriptor,
                self.optimizer_parameters,
                self.optimizer_args,
            ), optimizers_count = list_args(
                optimizer if optimizer is not None else self.hparams.optimizer,
                optimizer_is_active if optimizer_is_active is not None else self.hparams.optimizer_is_active,
                optimizer_parameters if optimizer_parameters is not None else self.hparams.optimizer_parameters,
                optimizer_args if optimizer_args is not None else self.hparams.optimizer_args,
                return_length=True,
            )
            self.optimizer_is_active_descriptor = (
                None
                if all(i is None for i in self.optimizer_is_active_descriptor)
                else self.optimizer_is_active_descriptor
            )

            # learning rates
            self.lr = list_args(lr if lr is not None else self.hparams.lr, length=optimizers_count)

            (
                (
                    self.scheduler,
                    self.scheduler_name,
                    self.scheduler_optimizer,
                    self.scheduler_args,
                    self.scheduler_interval,
                    self.scheduler_frequency,
                    self.scheduler_monitor,
                ),
                schedulers_count,
            ) = list_args(
                scheduler if scheduler is not None else self.hparams.scheduler,
                scheduler_name if scheduler_name is not None else self.hparams.scheduler_name,
                scheduler_optimizer if scheduler_optimizer is not None else self.hparams.scheduler_optimizer,
                scheduler_args if scheduler_args is not None else self.hparams.scheduler_args,
                scheduler_interval if scheduler_interval is not None else self.hparams.scheduler_interval,
                scheduler_frequency if scheduler_frequency is not None else self.hparams.scheduler_frequency,
                scheduler_monitor if scheduler_monitor is not None else self.hparams.scheduler_monitor,
                return_length=True,
            )
            schedulers_count = schedulers_count if schedulers_count and self.scheduler[0] is not None else 0
            if not schedulers_count:
                (
                    self.scheduler,
                    self.scheduler_name,
                    self.scheduler_optimizer,
                    self.scheduler_args,
                    self.scheduler_frequency,
                    self.scheduler_interval,
                    self.scheduler_monitor,
                ) = (None, None, None, None, None, None, None)
            if (schedulers_count == 1 and optimizers_count > 1) or (
                schedulers_count > 0 and all(self.scheduler_optimizer[i] is None for i in range(schedulers_count))
            ):
                self.scheduler_optimizer = [i for j in range(schedulers_count) for i in range(optimizers_count)]
                self.scheduler = [j for j in self.scheduler for i in range(optimizers_count)]
                self.scheduler_name = [j for j in self.scheduler_name for i in range(optimizers_count)]
                self.scheduler_args = [j for j in self.scheduler_args for i in range(optimizers_count)]
                self.scheduler_interval = [j for j in self.scheduler_interval for i in range(optimizers_count)]
                self.scheduler_frequency = [j for j in self.scheduler_frequency for i in range(optimizers_count)]
                self.scheduler_monitor = [j for j in self.scheduler_monitor for i in range(optimizers_count)]
            if schedulers_count:
                for idx, name in enumerate(self.scheduler_name):
                    param_name = self.optimizer_parameters[self.scheduler_optimizer[idx]]
                    param_name = (
                        f"/{param_name}"
                        if param_name and isinstance(param_name, str)
                        else f"/{self.scheduler_optimizer[idx]}"
                    )
                    self.scheduler_name[idx] = f"lr_scheduler{param_name}/{name if name is not None else idx}"
                self.__scheduler_step_count = [0 for i in range(len(self.scheduler))]

        # switching between manual and automatic optimization
        if hasattr(self, "optimizer_is_active_descriptor") and self.optimizer_is_active_descriptor is not None:
            self.automatic_optimization = False
            self.training_step = types.MethodType(TrainingModule.training_step_manual, self)
            self.__params_frozen = [False for i in range(optimizers_count)]
            self.__params_state = [None for i in range(optimizers_count)]
        else:
            self.training_step = types.MethodType(TrainingModule.training_step_automatic, self)

        # initialize the model
        if (
            model is not None
            or model_args is not None
            or model_cls is not None
            or (hasattr(self.hparams, "model_cls") and self.hparams.model_cls is not None)
            or (hasattr(self.hparams, "model_args") and self.hparams.model_args is not None)
        ):
            self.model = (
                model if model is not None else dy.eval(self.hparams.model_cls)(**(self.hparams.model_args or dict()))
            )

    @functools.cached_property
    def optimizer_is_active(self):
        if self.optimizer_is_active_descriptor is None:
            return None
        return [
            dy.eval(i, function_of_interest="is_active", dynamic_args=True)
            for i in self.optimizer_is_active_descriptor
        ]

    def forward(self, inputs):
        "Placeholder forward pass for the model"
        if hasattr(self, "model") and self.model is not None:
            return self.model(inputs)
        raise NotImplementedError("No model defined")

    def step(
        self,
        batch: th.Optional[th.Any] = None,
        batch_idx: th.Optional[int] = None,
        optimizer_idx: th.Optional[int] = None,
        name: str = "train",
        transformed_batch: th.Optional[th.Any] = None,
        transform_batch: bool = True,
        return_results: bool = False,
        return_factors: bool = False,
        log_results: bool = True,
        **kwargs,  # additional arguments to pass to the criterion and attacker
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
        transformed_batch = self.process_batch(
            batch, transformed_batch=transformed_batch, transform_batch=transform_batch
        )
        results, factors = self.criterion(batch=transformed_batch, training_module=self, return_factors=True, **kwargs)
        if log_results:
            self.log_step_results(results, factors, name)
        if return_results:
            return (results, factors) if return_factors else results
        return results["loss"] if not is_val else None

    def configure_optimizers(self):
        optimizers = [
            self.__configure_optimizer(
                opt_class=opt_cls, opt_args=opt_args, opt_base_lr=opt_base_lr, opt_parameters=opt_parameters
            )
            for opt_cls, opt_base_lr, opt_parameters, opt_args in zip(
                self.optimizer,
                self.lr,
                self.optimizer_parameters,
                self.optimizer_args,
            )
        ]
        schedulers = (
            [
                self.__configure_scheduler(
                    sched_class=sched_cls,
                    sched_optimizer=optimizers[sched_optimizer if sched_optimizer is not None else 0],
                    sched_args=sched_args,
                    sched_interval=sched_interval,
                    sched_frequency=sched_frequency,
                    sched_monitor=sched_monitor,
                    sched_name=sched_name,
                )
                for sched_cls, sched_optimizer, sched_args, sched_interval, sched_frequency, sched_monitor, sched_name in zip(
                    self.scheduler,
                    self.scheduler_optimizer,
                    self.scheduler_args,
                    self.scheduler_interval,
                    self.scheduler_frequency,
                    self.scheduler_monitor,
                    self.scheduler_name,
                )
            ]
            if self.scheduler
            else None
        )
        if schedulers:
            return (
                dict(optimizer=optimizers[0], scheduler=schedulers[0])
                if len(schedulers) == 1 and len(optimizers) == 1
                else (
                    optimizers,
                    schedulers,
                )
            )
        return optimizers

    def __configure_optimizer(self, opt_class, opt_base_lr, opt_args, opt_parameters):
        opt_class = dy.eval(opt_class)
        opt_args = {"lr": opt_base_lr, **(opt_args if opt_args is not None else {})}
        params = dy.eval(opt_parameters, context=self) if opt_parameters else self
        if isinstance(params, torch.Tensor):
            params = [params]
        else:
            params = params.parameters() if hasattr(params, "parameters") else params

        opt = opt_class(
            params,
            **opt_args,
        )
        return opt

    def __configure_scheduler(
        self, sched_class, sched_optimizer, sched_args, sched_interval, sched_frequency, sched_monitor, sched_name
    ):
        sched_class = dy.eval(sched_class)
        sched_args = sched_args or dict()
        sched = sched_class(sched_optimizer, **sched_args)
        sched_instance_dict = dict(
            scheduler=sched,
            interval=sched_interval,
            frequency=sched_frequency,
            name=sched_name,
            reduce_on_plateau=isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau),
        )
        if sched_monitor is not None:
            sched_instance_dict["monitor"] = sched_monitor
        return sched_instance_dict

    @functools.cached_property
    def batch_transform_function(self):
        """The transform function (callable) to apply to the inputs before forward pass.

        Returns:
            The compiled callable transform function if `self.inputs_transform` is provided else None
        """
        return dy.eval(self.batch_transform, function_of_interest="transform", dynamic_args=True, strict=False)

    def process_batch(
        self,
        batch,
        transformed_batch: th.Optional[th.Any] = None,
        transform_batch: bool = True,
    ):
        "Process the batch before forward pass"
        if transform_batch:
            if self.batch_transform_function is not None:
                transformed_batch = self.batch_transform_function(batch)
            else:
                transformed_batch = batch
        else:
            transformed_batch = batch
        return transformed_batch

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

    def training_step_automatic(self, batch, batch_idx, optimizer_idx=None, **kwargs):
        "Implementation for automatic Pytorch Lightning's training_step function"
        return self.step(batch, batch_idx, optimizer_idx, name="train", **kwargs)

    def manual_lr_schedulers_step(self, scheduler, scheduler_idx, **kwargs):
        "Implementation for manual Pytorch Lightning's lr_step function"
        frequency = self.scheduler_frequency[scheduler_idx]
        if not frequency:
            return
        interval = self.scheduler_interval[scheduler_idx]
        monitor = self.scheduler_monitor[scheduler_idx]

        step = False
        if interval == "batch":
            self.__scheduler_step_count[scheduler_idx] = (1 + self.__scheduler_step_count[scheduler_idx]) % frequency
            step = not self.__scheduler_step_count[scheduler_idx]
        elif interval == "epoch":
            step = self.trainer.is_last_batch and not (self.trainer.current_epoch % frequency)
        if not step:
            return
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if monitor not in self.trainer.callback_metrics:
                return  # no metric to monitor, skip scheduler step until metric is available in next loops
            scheduler.step(self.trainer.callback_metrics[monitor])
        else:
            scheduler.step()

    def training_step_manual(self, batch, batch_idx, **kwargs):
        "Implementation for manual training and optimization"
        optimizers = self.optimizers()
        optimizers = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
        schedulers = self.lr_schedulers()
        schedulers = schedulers if isinstance(schedulers, (list, tuple)) else ([schedulers] if schedulers else [])
        optimizer_is_active = [
            self.optimizer_is_active[i](training_module=self, optimizer_idx=i) for i in range(len(optimizers))
        ]
        # freezing/unfreezing the optimizer parameters
        for optimizer_idx, optimizer in enumerate(optimizers):
            if optimizer_is_active[optimizer_idx] and self.__params_frozen[optimizer_idx]:
                unfreeze_params(optimizer=optimizer, old_states=self.__params_state[optimizer_idx])
                self.__params_frozen[optimizer_idx] = False
            elif not optimizer_is_active[optimizer_idx] and not self.__params_frozen[optimizer_idx]:
                self.__params_state[optimizer_idx] = freeze_params(optimizer=optimizer)
                self.__params_frozen[optimizer_idx] = True
        loss = self.step(batch, batch_idx, None, name="train")
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            self.manual_backward(loss)
        else:
            return loss
        for optimizer_idx, optimizer in enumerate(optimizers):
            if optimizer_is_active[optimizer_idx]:
                optimizer.step()  # todo: add support for LBFGS optimizers via closures
                optimizer.zero_grad()  # todo: move to before the backward call and add support for gradient accumulation
        # following pytorch>=1.1.0 conventions, calling scheduler.step after optimizer.step
        # visit the docs for more details https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        for idx, scheduler in enumerate(schedulers):
            if optimizer_is_active[self.scheduler_optimizer[idx]]:
                self.manual_lr_schedulers_step(scheduler=scheduler, scheduler_idx=idx)
        return loss

    def validation_step(self, batch, batch_idx):
        "Pytorch Lightning's validation_step function"
        return self.step(batch, batch_idx, name="val")
