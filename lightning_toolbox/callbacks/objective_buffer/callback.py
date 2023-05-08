# Copyright Vahid Zehtab (vahid@zehtab.me) 2023
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
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import typing as th

from .buffers import ObjectiveBuffer, ObjectiveExtremesBuffer, ObjectiveRandomBuffer
import dypy as dy
import random


OBJECTIVE_BUFFERS = {
    "extremes": ObjectiveExtremesBuffer,
    "regular": ObjectiveBuffer,
    "random": ObjectiveRandomBuffer,
}


class ObjectiveBufferCallback(pl.Callback):
    """
    This is an abstract class that provides a framework for buffering the values of the objective function terms.
    The values are buffered in a dictionary, and the dictionary is cleared after each epoch (or an arbitrary schedule).

    The values are stored in a dictionary, for each latch in the objective function:
    - inputs_latch: The inputs to the objective function
    - latch: Custom values remembered from previous calls to the objective function
    - results_latch: The results of the terms
    - factors_latch: The factors of the terms

    This could be very useful for debugging, or for times when you are dealing with a small enough dataset that you can
    store the values of an entire epoch in memory.
    """

    def __init__(
        self,
        objective: th.Union["lightning_toolbox.Objective", str] = "objective",  # type: ignore
        phase: th.Union[str, th.List[str]] = "fit",
        # flush options
        flush_on_epoch: bool = True,
        flush_interval: int = 1,  # 0 means no flushing
        # loop options
        buffer_train: bool = True,
        buffer_validation: bool = True,
        buffer_test: bool = False,
        # buffer options
        buffer_type: th.Literal["extremes", "random", "regular"] = "regular",
        buffer_args: th.Optional[dict] = None,  # including size & ...
        buffer_inputs: bool = True,
        buffer_latch: bool = True,
        buffer_results: bool = True,
        buffer_factors: bool = True,
    ) -> None:
        super().__init__()
        # objective of interest is either an attribute of the pl_module, or specified directly by the user
        # in first case, the objective is a string that specifies the name of the attribute, and in the second case
        # the objective is the objective itself
        # if we are given a string, we need to get the objective from the pl_module which is not available at this point
        # so we store the "objective_descriptor" and then get the objective in the setup method (when the pl_module is available)
        self.__objective_descriptor: th.Union["lightning_toobox.Objective", str] = objective  # type: ignore
        self.objective: "lightning_toobox.Objective" = None  # type: ignore

        self.__phase = frozenset(phase) if isinstance(phase, list) else frozenset([phase])

        _latches = dict(inputs=buffer_inputs, latch=buffer_latch, results=buffer_results, factors=buffer_factors)
        self.__latches = frozenset(item for item, do_track in _latches.items() if do_track)

        # buffers
        if buffer_type not in OBJECTIVE_BUFFERS:
            raise ValueError(f"buffer_type must be one of {list(OBJECTIVE_BUFFERS.keys())}")
        buffer_cls: type = OBJECTIVE_BUFFERS[buffer_type]
        buffer_args = buffer_args or dict()
        self.train_buffer = buffer_cls(**buffer_args) if buffer_train else None
        self.validation_buffer = buffer_cls(**buffer_args) if buffer_validation else None
        self.test_buffer = buffer_cls(**buffer_args) if buffer_test else None

        # flush
        self.flush_on_epoch = flush_on_epoch
        self.flush_interval = flush_interval

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if stage in self.__phase:
            self.objective: "lightning_toobox.Objective" = (  # type: ignore
                dy.eval(self.__objective_descriptor, pl_module)
                if isinstance(self.__objective_descriptor, str)
                else self.__objective_descriptor
            )

    def commit_latches(
        self,
        loop: th.Literal["train", "validation", "test"],
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        **details,
    ):
        buffer: th.Optional[ObjectiveBuffer] = getattr(self, f"{loop}_buffer")
        details.update(loop=loop)
        if buffer is None:
            return

        # add shit=random number with probability 0.1 to details
        shit = random.random()
        if shit > 0.95:
            details.update(shit=shit)

        if not self.flush_on_epoch and self.flush_interval and buffer.num_commits == self.flush_interval:
            self.work_on_flush(loop=loop, trainer=trainer, pl_module=pl_module)
            buffer.flush()
        buffer.commit_latches(
            latches={name: latch for name, latch in self.objective.latches.items() if name in self.__latches},
            all_latches=dict(self.objective.latches.items()),
            details=details,
        )

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: th.Any, batch_idx: int
    ) -> None:
        self.commit_latches(
            loop="train",
            trainer=trainer,
            pl_module=pl_module,
            batch_idx=batch_idx,
            step=trainer.global_step,
            epoch=trainer.current_epoch,
        )
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: th.Optional[STEP_OUTPUT],
        batch: th.Any,
        batch_idx: int,
        dataloader_idx: th.Optional[int] = None,
    ) -> None:
        self.commit_latches(
            loop="validation",
            trainer=trainer,
            pl_module=pl_module,
            batch_idx=batch_idx,
            step=trainer.global_step,
            epoch=trainer.current_epoch,
        )
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: th.Optional[STEP_OUTPUT],
        batch: th.Any,
        batch_idx: int,
        dataloader_idx: th.Optional[int] = None,
    ) -> None:
        self.commit_latches(
            loop="test",
            trainer=trainer,
            pl_module=pl_module,
            batch_idx=batch_idx,
            step=trainer.global_step,
            epoch=trainer.current_epoch,
        )
        return super().on_test_batch_end(trainer, pl_module, outputs, batch, dataloader_idx)

    def flush_on_epoch_end(self, loop: str, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        buffer: th.Optional[ObjectiveBuffer] = getattr(self, f"{loop}_buffer")
        if not buffer:
            return
        if self.flush_on_epoch and self.flush_interval and (trainer.current_epoch + 1) % self.flush_interval == 0:
            self.work_on_flush(loop=loop, trainer=trainer, pl_module=pl_module)
            buffer.flush()

    def work_on_flush(self, loop: str, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Method to be implemented by subclasses before flushing the buffer."""
        pass

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.work_on_train_epoch_end(trainer, pl_module)
        self.flush_on_epoch_end(loop="train", trainer=trainer, pl_module=pl_module)
        return super().on_train_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.work_on_validation_epoch_end(trainer, pl_module)
        self.flush_on_epoch_end(loop="validation", trainer=trainer, pl_module=pl_module)
        return super().on_validation_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.work_on_test_epoch_end(trainer, pl_module)
        self.flush_on_epoch_end(loop="test", trainer=trainer, pl_module=pl_module)
        return super().on_test_epoch_end(trainer, pl_module)

    def work_on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Method to be implemented by subclasses to perform work on the train epoch end."""
        pass

    def work_on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Method to be implemented by subclasses to perform work on the validation epoch end."""
        pass

    def work_on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Method to be implemented by subclasses to perform work on the test epoch end."""
        pass
