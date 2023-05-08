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
import typing as th
import torch
from collections import defaultdict
from .markers import _empty
from tabulate import tabulate


class ObjectiveBuffer:
    def __init__(
        self,
        size: th.Optional[int] = None,
        # tensor buffering settings
        detach_tensors: bool = True,
        cpu_buffering: bool = True,
    ):
        self.buffers: th.Dict[str, th.Dict[th.Any, th.List[th.Any]]] = dict()

        self.size: th.Optional[int] = size
        self.cpu_buffering: bool = cpu_buffering
        self.detach_tensors: bool = detach_tensors
        self.num_commits: int = 0

    def flush(self):
        """Flush the buffer"""
        self.buffers = dict()
        self.num_commits = 0

    def commit_latches(
        self,
        latches: th.Dict[str, th.Dict[th.Any, th.Any]],
        details: th.Optional[th.Dict[str, th.Any]] = None,
        **kwargs,
    ):
        """Buffers a list of latches and pad and trim them if necessary."""
        if details is not None:
            details["commit_num"] = self.num_commits
        latches.update(details=details)
        current_length = len(self)
        for name, latch in latches.items():
            # create a new buffer list to track items in the latch
            if name not in self.buffers:
                self.buffers[name] = defaultdict(list)
                # make the the buffer accessible as an attribute
                setattr(self, name, self.buffers[name])

            # commit values in the latch
            for key, value in latch.items():
                if key not in self.buffers[name]:
                    # initialize the buffer with appropriate amount of empty values
                    self.buffers[name][key].extend([_empty] * current_length)
                # append the value to the buffer
                self.buffers[name][key].append(self._process_tensor(value))

            # add empty values to the buffer for missing keys
            for key in [k for k in self.buffers[name] if k not in latch]:
                self.buffers[name][key].append(_empty)
        self._trim_buffers()  # remove redundant items if necessary
        self.num_commits += 1

    def _process_tensor(self, tensor: torch.Tensor):
        """Utility method for processing tensors before commiting them to a buffer

        Default implementation detaches the tensor from its computational graph and
        """
        if isinstance(tensor, (list, tuple)):
            return (list if isinstance(tensor, list) else tuple)(self._process_tensor(t) for t in tensor)
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if self.detach_tensors:
            tensor = tensor.detach()
        if self.cpu_buffering:
            tensor = tensor.cpu()
        return tensor

    def __len__(self):
        """Returns the current legnth of commited latch buffers"""
        if not self.buffers:
            return 0

        return max(
            (max(len(latch_values) for latch_values in latch_buffers.values()) if latch_buffers else 0)
            for latch_buffers in self.buffers.values()
        )

    def _trim_buffers(self):
        """Removes the redundant buffered values if necessary."""
        if self.size is not None and len(self) > self.size:
            for latch_values in [values for buffer in self.buffers.values() for values in buffer.values()]:
                latch_values.pop(0)

    @property
    def dense_buffers(self):
        """Same as buffers but `_empty` values are filtered out for convinience"""
        results: th.Dict[str, th.Dict[th.Any, th.List[th.Any]]] = dict()
        for name, buffer in self.buffers.items():
            results[name]: th.Dict[th.Any, th.List[th.Any]] = defaultdict(list)
            for key, value in buffer.items():
                results[name][key].extend(item for item in value if item != _empty)
        return results

    def __repr__(self) -> str:
        size = f"size={self.size if self.size is not None else 'inf'}"
        cpu_buffering = f"cpu_buffering={self.cpu_buffering}"
        detach_tensors = f"detach_tensors={self.detach_tensors}"
        current_length = f"current_length={len(self)}"
        num_commits = f"num_commits={self.num_commits}"
        return f"{self.__class__.__name__}({', '.join([size, cpu_buffering, detach_tensors, current_length, num_commits])})"

    def summary(self, name: th.Optional[str] = None):
        """Returns a summary of the buffer in a tabular format for display purposes.

        The summary includes the number of items in the buffer, the number of missing values.

        Args:
            name (str, optional): name of the buffer to summarize. If None, all buffers are
                summarized. Defaults to None.
        """
        if name is None:
            return "\n\n".join(self.summary(name=name) for name in self.buffers.keys())
        latch_buffer = self.buffers[name]
        headers = ["key", "length", "missing"]
        table = []
        for key, latch_values in latch_buffer.items():
            length = len(latch_values)
            missing = len([value for value in latch_values if value is _empty])
            table.append([key, length, missing])
        tabular = tabulate(table, headers=headers, tablefmt="grid", stralign="center", numalign="center")
        # center the name of the buffer on top of the table
        table_size = tabular.find("\n")
        name = name.center(table_size - 2)

        return f"{'-'*table_size}\n|{name}|\n{'-'*table_size}\n{tabular}"

    def sparsify(
        self, latch: str, key: str, return_details: bool = False
    ) -> th.Union[th.List[th.Any], th.Tuple[th.List[th.Any], th.List[int]]]:
        """Sparsifies a buffer for a given latch and key.

        Args:
            latch (str): name of the latch
            key (str): name of the key
            return_details (bool, optional): whether to return the details for each commit.
                Defaults to True.

        Returns:
            th.Union[th.List[th.Any], th.Tuple[th.List[th.Any], th.List[int]]]: the sparsified buffer
                and the commit details if `return_details` is True
        """
        latch_buffer = self.buffers[latch]
        if not return_details:
            return [value for value in latch_buffer[key] if value is not _empty]

        details, values = [], []
        for idx, value in enumerate(latch_buffer[key]):
            if value is not _empty:
                values.append(value)
                details.append({k: v[idx] for k, v in self.buffers["details"].items()})
        return values, details

    def retrieve(self, commit_num: int, latch: th.Optional[str] = None):
        """Retrieves the values of the buffer at a given commit number.

        Args:
            commit_num (int): commit number
            latch (str, optional): name of the latch. If None, all latches are retrieved.
                Defaults to None.

        Returns:
            th.Union[th.Dict[str, th.Dict[str, th.Any]], th.Dict[str, th.Any]]: the values of the buffer
                at the given commit number.
        """
        if latch is None:
            return {name: self.retrieve(commit_num=commit_num, latch=name) for name in self.buffers}

        # TODO: find the index of the commit number in the details buffer
        latch_buffer = self.buffers[latch]
        return {key: latch_buffer[key][commit_num] for key in latch_buffer.keys()}
