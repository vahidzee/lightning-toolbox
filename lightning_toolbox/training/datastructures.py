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
import typing as th


class ArgsListDict(dict):
    def __init__(self, length: th.Optional[int] = None, **kwargs):
        super().__init__()
        self.args_length = max(
            len(arg) if isinstance(arg, (tuple, list)) else (1 if arg is not None else 0) for arg in kwargs.values()
        )
        self.args_length = self.args_length if length is None else length
        for name, arg in kwargs.items():
            self[name] = None if isinstance(arg, (tuple, list)) and all(i is None for i in arg) else arg

    def value(self, idx: int, key: str = None) -> dict:
        if key is not None:
            return self[key][idx] if isinstance(self[key], list) else self[key]
        return {name: arg[idx] if isinstance(arg, list) else arg for name, arg in self.items()}

    def __iter__(self):
        self.__iter_idx = -1
        return self

    def __next__(self) -> dict:
        if self.__iter_idx == self.args_length - 1:
            raise StopIteration
        self.__iter_idx += 1
        return {name: arg[self.__iter_idx] if isinstance(arg, list) else arg for name, arg in self.items()}

    def cross_populate(self, length) -> "ArgsListDict":
        results = dict()
        for name, arg in self.items():
            results[name] = [j for j in arg for i in range(length)] if isinstance(arg, list) else arg
        return ArgsListDict(length=self.args_length * length, **results)
