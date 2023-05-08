# Copyright Vahid Zehtab (vahid@zehtab.me) 2023-
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
from .regular import ObjectiveBuffer
from .markers import _empty
from collections import defaultdict
import typing as th
import bisect
import torch


class ObjectiveExtremesBuffer(ObjectiveBuffer):
    def __init__(
        self,
        size: int = 10,
        monitor_latch: th.Literal["inputs", "latch", "factors", "results"] = "results",
        monitor: str = "loss",
        monitor_mode: th.Literal["min", "max", "both"] = "both",
        # tensor buffering settings
        detach_tensors: bool = True,
        cpu_buffering: bool = True,
    ):
        assert size > 0, "Objective's Extremes buffer size should be greater than zero"
        self.monitor_latch: str = monitor_latch  # the name of the latch to get the monitor value from
        self.monitor: str = monitor  # the name of the value to monitor
        self.monitor_mode: th.Literal["min", "max", "both"] = monitor_mode  # the mode of the monitor
        # keep a sorted datastructure of the best and worst monitor values and their corresponding indices in the buffer
        # we use these datastructures to keep track of the best and worst values committed to the buffer and only keep the
        # top `size` values in the buffer
        # therefore we should be able to add a value to these datastructures preferably in O(log(n)) time or less
        self._min_values, self._max_values = [], []
        super().__init__(size=size, detach_tensors=detach_tensors, cpu_buffering=cpu_buffering)

    def _process_value(self, value: th.Any, batch_size: int, idx: int, is_batch: bool):
        if isinstance(value, tuple):
            return tuple(
                self._process_value(value=v, batch_size=batch_size, idx=idx, is_batch=is_batch) for v in value
            )
        # if is subscriptable, process each element
        if hasattr(value, "__getitem__") and hasattr(value, "__len__"):
            value = (value[idx] if len(value) == batch_size else value) if is_batch else value
        return self._process_tensor(value)

    def _initialize_buffer(self, name: str, kind: th.Literal["min", "max"]):
        name = f"{kind}_{name}"
        if name not in self.buffers and (kind == self.monitor_mode or self.monitor_mode == "both"):
            self.buffers[name] = defaultdict(list)
            setattr(self, name, self.buffers[name])

    def _commit_values(
        self,
        monitor_value: th.Any,
        batch_size: int,
        monitor_idx: int,
        latches: th.Dict[str, th.Dict[th.Any, th.Any]],
        kind: th.Literal["min", "max"],
        is_batch: bool,
    ):
        if not (kind == self.monitor_mode or self.monitor_mode == "both"):
            return
        extreme_values = self._min_values if kind == "min" else self._max_values
        insert_idx = (bisect.bisect_left if kind == "min" else bisect.bisect_right)(extreme_values, monitor_value)
        # update extreme values
        extreme_values.insert(insert_idx, monitor_value)
        clip = False
        if len(extreme_values) > self.size:
            clip = True
            extreme_values.pop(self.size if kind == "min" else 0)
        # commit latches
        for name, latch in latches.items():
            latch_name = f"{kind}_{name}"

            for key, value in latch.items():
                # if the key was missing in previous commits, pad the buffer with empty values
                if key not in self.buffers[latch_name]:
                    self.buffers[latch_name][key] = [_empty] * (self.size if clip else len(extreme_values) - 1)
                self.buffers[latch_name][key].insert(
                    insert_idx,
                    self._process_value(value=value, batch_size=batch_size, idx=monitor_idx, is_batch=is_batch),
                )

            # commit missing values
            for key in [k for k in self.buffers[latch_name].keys() if k not in latch]:
                self.buffers[latch_name][key].insert(insert_idx, _empty)

            # remove the worst/best value if the buffer is full
            if not clip:
                continue
            for key in self.buffers[latch_name].keys():
                self.buffers[latch_name][key].pop(self.size if kind == "min" else 0)

    def commit_latches(
        self,
        latches: th.Dict[str, th.Dict[th.Any, th.Any]],
        all_latches: th.Dict[str, th.Dict[th.Any, th.Any]],
        details: th.Optional[th.Dict[str, th.Any]] = None,
        **kwargs,
    ):
        monitor_values = all_latches[self.monitor_latch][self.monitor]
        if details is not None:
            details["commit_num"] = self.num_commits
        latches.update(details=details)
        # the monitor value is either a scalar (single comparable instance) or a batch of comparable instances (a tensor)
        is_batch: bool = False
        if not isinstance(monitor_values, torch.Tensor) or len(monitor_values.reshape(-1)) != 1:
            monitor_values = monitor_values.reshape(-1)
            batch_size = len(monitor_values)
            is_batch = True
        else:
            monitor_values = [monitor_values.item()]
            batch_size = 1
        for name in latches:
            self._initialize_buffer(name=name, kind="min")
            self._initialize_buffer(name=name, kind="max")

        for i, monitor_value in enumerate(monitor_values):
            monitor_value = monitor_value.item() if isinstance(monitor_value, torch.Tensor) else monitor_value
            self._commit_values(
                monitor_value=monitor_value,
                batch_size=batch_size,
                monitor_idx=i,
                latches=latches,
                kind="min",
                is_batch=is_batch,
            )
            self._commit_values(
                monitor_value=monitor_value,
                batch_size=batch_size,
                monitor_idx=i,
                latches=latches,
                kind="max",
                is_batch=is_batch,
            )
        self.num_commits += 1
