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
from torch.utils.data import Dataset, DataLoader, random_split
import lightning
import dypy as dy


class DataModule(lightning.LightningDataModule):
    def __init__(
        self,
        # dataset
        dataset: th.Union[str, Dataset],
        dataset_args: th.Optional[dict] = None,
        train_dataset: th.Optional[th.Union[str, Dataset, type]] = None,
        train_dataset_args: th.Optional[dict] = None,
        val_size: th.Optional[th.Union[int, float]] = None,
        val_dataset: th.Optional[th.Union[str, Dataset]] = None,
        val_dataset_args: th.Optional[dict] = None,
        test_dataset: th.Optional[th.Union[str, Dataset]] = None,
        test_dataset_args: th.Optional[dict] = None,
        # batch_size
        batch_size: th.Optional[int] = 16,
        train_batch_size: th.Optional[int] = None,
        val_batch_size: th.Optional[int] = None,
        test_batch_size: th.Optional[int] = None,
        # pin memory
        pin_memory: bool = True,
        train_pin_memory: th.Optional[bool] = None,
        val_pin_memory: th.Optional[bool] = None,
        test_pin_memory: th.Optional[bool] = None,
        # shuffle
        train_shuffle: bool = True,
        val_shuffle: bool = False,
        test_shuffle: bool = False,
        # num_workers
        num_workers: int = 0,
        train_num_workers: th.Optional[int] = None,
        val_num_workers: th.Optional[int] = None,
        test_num_workers: th.Optional[int] = None,
        # seed
        seed: int = 0,
        # initialization settings
        save_hparams: bool = True,
        initialize_superclass: bool = True,
        # extra parameters (ignored)
        **kwargs,
    ):
        if initialize_superclass:
            super().__init__()
        if save_hparams:  # Save all the hyper parameters for further reference
            self.save_hyperparameters()

        # seed
        self.seed = seed

        # data
        self.dataset = dataset
        # dataset instances (not classes) are stored in self.*data variables (set during setup)
        self.data, self.train_data, self.val_data, self.test_data = None, None, None, None

        # dataset classes (not instances) are stored in self.*dataset variables
        self.train_dataset = train_dataset if train_dataset is not None else dataset

        self.val_size = val_size if self.val_data is None else 0  # if val_data is provided, val_size is ignored
        assert (
            val_size is None or isinstance(val_size, int) or (isinstance(val_size, float) and (0 <= val_size <= 1))
        ), "invalid validation size is provided (either int, float (between zero and one) or None)"
        self.val_dataset = val_dataset if val_dataset is not None else dataset
        self.test_dataset = test_dataset if test_dataset is not None else dataset
        self.train_dataset_args = {
            **(dataset_args or dict()),
            **(train_dataset_args or dict()),
        }
        self.val_dataset_args = {
            **(dataset_args or dict()),
            **(val_dataset_args or dict()),
        }
        self.test_dataset_args = {
            **(dataset_args or dict()),
            **(test_dataset_args or dict()),
        }

        # batch_size
        self.train_batch_size = train_batch_size if train_batch_size is not None else batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.test_batch_size = test_batch_size if test_batch_size is not None else batch_size
        assert (
            self.train_batch_size or self.val_batch_size or self.test_batch_size
        ), "at least one of batch_sizes should be a positive number"

        # pin memory
        self.train_pin_memory = train_pin_memory if train_pin_memory is not None else pin_memory
        self.val_pin_memory = val_pin_memory if val_pin_memory is not None else pin_memory
        self.test_pin_memory = test_pin_memory if test_pin_memory is not None else pin_memory

        # shuffle
        self.train_shuffle, self.val_shuffle, self.test_shuffle = train_shuffle, val_shuffle, test_shuffle

        # num_workers
        self.train_num_workers = train_num_workers if train_num_workers is not None else num_workers
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.test_num_workers = test_num_workers if test_num_workers is not None else num_workers

    @staticmethod
    def get_dataset(dataset: th.Union[str, Dataset], **params):
        assert isinstance(
            dataset, (type, str, Dataset)
        ), "dataset should either be a dataset instance, a dataset class or an import-paths str"
        if isinstance(dataset, Dataset):
            return dataset
        elif isinstance(dataset, type):
            return dataset(**params)
        else:
            return DataModule.get_dataset(dy.get_value(dataset, strict=True), **params)

    def setup(self, stage: th.Optional[str] = None) -> None:
        if stage == "fit" or stage == "tune":
            self.split_dataset()

        elif stage == "test" and self.test_batch_size:
            test_data = self.get_dataset(
                self.test_dataset,
                **self.test_dataset_args,
            )
            self.test_data = test_data

    def split_dataset(self):
        self.train_data = self.train_data or self.dataset  # in case the train_data is not provided

        dataset = self.get_dataset(
            self.train_dataset,
            **self.train_dataset_args,
        )
        self.data = dataset  # for ease of access (e.g. for plotting)

        if not self.val_size:
            # setup train data (in case validation is not to be a subset of provided dataset with val_size)
            self.train_data = dataset
        if self.val_size and self.val_batch_size:
            # split dataset into train and val in case val_size is provided
            train_len = (
                len(dataset) - self.val_size
                if isinstance(self.val_size, int)
                else int(len(dataset) * (1 - self.val_size))
            )
            prng = torch.Generator()
            prng.manual_seed(self.seed)  # for reproducibility
            train_data, val_data = random_split(dataset, [train_len, len(dataset) - train_len], generator=prng)
            self.train_data, self.val_data = train_data, val_data

        elif self.val_batch_size:
            val_data = self.get_dataset(self.val_dataset, **self.val_dataset_args)
            self.val_data = val_data

    def get_dataloader(self, name, batch_size=None, shuffle=None, num_workers=None, pin_memory=None, **params):
        data = getattr(self, f"{name}_data")
        if data is None:
            return None
        batch_size = batch_size if batch_size is not None else getattr(self, f"{name}_batch_size")
        shuffle = shuffle if shuffle is not None else getattr(self, f"{name}_shuffle")
        num_workers = num_workers if num_workers is not None else getattr(self, f"{name}_num_workers")
        pin_memory = pin_memory if pin_memory is not None else getattr(self, f"{name}_pin_memory")
        return DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **params,
        )

    def train_dataloader(self, **params):
        return self.get_dataloader("train", **params)

    def val_dataloader(self, **params):
        return self.get_dataloader("val", **params)

    def test_dataloader(self, **params):
        return self.get_dataloader("test", **params)
