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
import typing as th
import random
import torch


class ObjectiveRandomBuffer(ObjectiveBuffer):
    def __init__(
        self,
        size: int = 10,
        commit_prob: float = 0.5,
        # tensor buffering settings
        detach_tensors: bool = True,
        cpu_buffering: bool = True,
    ):
        assert size > 0, "Objective's Extremes buffer size should be greater than zero"
        assert 0 <= commit_prob <= 1, "Objective's Random buffer commit_prob should be between 0 and 1"
        self.commit_prob = commit_prob
        super().__init__(size=size, detach_tensors=detach_tensors, cpu_buffering=cpu_buffering)

    def commit_latches(
        self,
        latches: th.Dict[str, th.Dict[th.Any, th.Any]],
        details: th.Optional[th.Dict[str, th.Any]] = None,
        **kwargs,
    ):
        """Buffers a list of latches and pad and trim them if necessary."""
        if torch.rand(1) > self.commit_prob:
            self.num_commits += 1
            return
        super().commit_latches(latches=latches, details=details, **kwargs)

    def _trim_buffers(self):
        """Removes the redundant buffered values if necessary."""
        if self.size is not None and len(self) > self.size:
            idx = random.randint(0, self.size)
            for latch_values in [values for buffer in self.buffers.values() for values in buffer.values()]:
                latch_values.pop(idx)
