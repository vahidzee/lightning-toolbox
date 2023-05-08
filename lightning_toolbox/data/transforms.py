# Copyright Vahid Zehtab (vahid@zehtab.me) 2022-2023
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
import dypy as dy
import inspect
import torch
import functools
import torchvision

# for transforms utils
dy.register_context(torchvision)

TransformDescriptor = th.Union[
    dy.FunctionDescriptor, th.Type, str, th.Dict[th.Literal["class_path", "init_args"], th.Any]
]
TransformsDescriptor = th.Union[TransformDescriptor, th.List[TransformDescriptor]]


def _initialize_transforms(transforms: th.Optional[th.Union[list, dict, th.Any]]) -> th.Union[th.Callable, list, None]:
    """Utility function to initialize transforms from a list of transforms, a single transform, or a list of dicts that
    contains class_path and init_args keys.

    This function is used internally by `initialize_transforms` and `initialize_transforms`. It is not recommended to
    use this function directly. Use `initialize_transforms` instead.

    Args:
        transforms: list of transforms, a single transform, or a list of dicts that contains
            class_path and init_args keys.

    """
    if transforms is None or (not inspect.isclass(transforms) and callable(transforms)):
        return transforms

    # list of other transforms
    if isinstance(transforms, list):
        return [_initialize_transforms(i) for i in transforms]

    # either a class and args, or a code block and entry function
    if isinstance(transforms, dict):
        if "class_path" in transforms:
            return dy.get_value(transforms["class_path"])(**transforms.get("init_args", dict()))
        value = dy.eval_function(transforms, "transform")
        return value() if inspect.isclass(value) else value
    if isinstance(transforms, str):
        try:
            return dy.get_value(transforms)()
        except:
            return dy.eval(transforms, function_of_interest="transform")


def initialize_transforms(
    transforms: th.Optional[TransformsDescriptor], force_list: bool = False
) -> th.Optional[TransformsDescriptor]:
    """
    Initialize transforms from a list of transforms, a single transform, or a list of dicts that contains
    class_path and init_args keys.

    Args:
        transforms: list of transforms, a single transform, or a list of dicts that contains
            class_path and init_args keys.
        force_list: if True, the result will be a list even if there is only one transform. Take note that
            if the result is None, it will be returned as None.

    Returns:
        Depending on `force_list` and the number of transforms, it can return a single transform, a list of transforms,
        or None.
    """
    results = _initialize_transforms(transforms)
    if results is None:
        return None
    if force_list and not isinstance(results, list):
        results = [results]
    return results


def _wrap(func, transform: th.Callable):
    """
    Utility function to wrap a getitem function with a single transform. The dataset is modified in place,
    and the original __getitem__ method is wrapped with the transform.

    Args:
        dataset: dataset to wrap
        transform: transform to wrap the dataset with

    Returns:
        None
    """

    def wrapper(*args, **kwargs):
        datam = func(*args, **kwargs)
        return transform(datam)

    return wrapper


def transform_dataset(
    dataset: th.Optional["torch.utils.data.Dataset"],
    transforms: th.Union[th.List[th.Callable], th.Callable, None],
) -> th.Optional["torch.utils.data.Dataset"]:
    """
    Wrap a dataset with (a list of) transform(s). See `_wrap_dataset` for more details.

    Args:
        dataset: dataset to wrap
        transforms: (a list of) transform(s) to wrap the dataset with
    """
    if transforms is None or dataset is None:
        return dataset
    func = dataset.__getitem__
    if isinstance(transforms, list):
        for transform in transforms:
            func = _wrap(func, transform)
    else:
        func = _wrap(func, transforms)

    # dynamically create a new class that inherits from the dataset class, and override __getitem__ method
    # with the wrapped function
    # this is done to avoid modifying the dataset class itself, __getitem__ method and all dunders are bound
    # to the original dataset class and if we modify the dataset class, and it's not possible to wrap the
    # __getitem__ method of a single instance of the dataset class.

    # externally, the wrapped dataset should behave like the original dataset class, and the only difference
    # is that the __getitem__ method is wrapped with the transform(s), for this all the attributes and methods
    # of the original dataset class should be accessible from the wrapped dataset class, and the wrapped
    # dataset class should be an instance of the original dataset class.
    # (isinstance(wrapped_dataset, dataset.__class__) == True)
    WrapDataset = type(
        "WrapDataset",
        (dataset.__class__,),
        dict(
            # __init__ is overridden to bind the original dataset class to the wrapped dataset class
            __init__=lambda self, dataset: setattr(self, "_base_dataset", dataset),
            # dir and getattr are overridden to bind the original dataset class to the wrapped dataset class
            __dir__=lambda self: dir(self._base_dataset),
            __getattr__=(
                lambda self, __name: (getattr(self._base_dataset, __name) if __name != "__getitem__" else func)
            ),
            # __getitem__ is overridden to bind the original dataset class to the wrapped dataset class
            __getitem__=functools.wraps(dataset.__getitem__)(lambda self, index: func(index)),
        ),
    )
    return WrapDataset(dataset)


class TupleDataTransform:
    """
    Utility class to wrap a tuple dataset (any dataset that returns a tuple e.g. (input, target)) with
    a list of transforms for each element of the tuple.

    This class specially useful for dealing with datasets from torchvision, and coupling them with
    torchvision.transforms.

    Example:
        >>> from torchvision.datasets import MNIST
        >>> from torchvision import transforms
        >>> from lightning_toolbox.data import TupleDataTransform
        >>> dataset = MNIST(root=".", download=True, transform=None, target_transform=None)
        >>> transform = TupleDataTransform(transforms.ToTensor(), transforms.ToTensor())
        >>> transform(dataset, transform)

    Attributes:
        transforms: A list of either Nones or transformations to be applied to associated elements of the tuple.
    """

    def __init__(
        self,
        transforms: th.Union[
            th.List[th.Optional[TransformDescriptor]], th.Dict[int, th.Optional[TransformDescriptor]]
        ],
    ):
        """
        Initialize the transform.

        Args:
            transforms: A list of either Nones or transformations to be applied to associated elements of the tuple.
        """
        if isinstance(transforms, dict):
            transforms = [transforms.get(i, None) for i in range(len(transforms))]
        self.transforms = [initialize_transforms(t, force_list=True) if t is not None else None for t in transforms]

    def __call__(self, datam) -> th.Tuple[th.Any, th.Any]:
        """
        Apply the transforms to the input and target.

        Args:
            datam: a tuple of (input, target)

        Returns:
            a tuple of (input, target)
        """
        results = list(datam)
        for i, transforms in enumerate(self.transforms):
            for transform in transforms or []:
                results[i] = transform(results[i])
        return tuple(results)
