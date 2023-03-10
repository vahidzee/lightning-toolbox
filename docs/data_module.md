# `lightning_toolbox.DataModule`
`lightning_toolbox.DataModule` aims to capture the generic task for setting up DataModules in pytorch lightning (which are a way of decoupling data from the model implementation, [see the docs](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html)). It could be very useful if you are dealing with datasets with the following assumptions:

1. You don't want to transform the data on the fly, and your dataset object returns samples which are ready to be fed to your model. (`TODO:` In a future version, we will add support for on-the-fly functional transformations)
2. Creating a `torch.utils.data.DataLoader` from your dataset is as simple as calling `torch.utils.data.DataLoader(dataset, batch_size=...)`, which is the case for most datasets in deep learning.

In this case, you can use `lightning_toolbox.DataModule` to create a `LightningDataModule` from your dataset(s). You can either use it to transform your dataset(s) into a datamodule, or create your own datamodule by inheriting a class from it.

**Table of Contents**:
- [Usage](#usage)
    - [Specifying Datasets](#specifying-datasets)
    - [Validation Split](#validation-split)
    - [Configuring the Dataloaders](#configuring-the-dataloaders)


## Usage
You can use `lightning_toolbox.DataModule` either by instanciating it, or extending it's base functionality through class inheritance. Our `DataModule`'s functionality is two fold:

1. Dealing with the data: The data could be provided to the data module in either of the two following settings:

    - Classes (or string import-path of) datasets to be instantiated along with their instantiation arguments.
    - Dataset instances to be used directly.

    The instantiation process takes place in the `setup` method, ensuring the correct behaviour in multi-device training scenarios. Depending on the provided configuration, the datamodule might split a given `dataset` into training and validation splits based on a `val_size` argument.


2. Creating the dataloaders:
    `train_dataloader`, `val_dataloader`, `test_dataloader` methods are implemented to return the dataloaders based on `train_data`, `val_data`, and `test_data` attributes. The configuration for each dataloader could be individually modified through the `DataModule`'s constructor.



### Specifying Datasets
You can specify your datasets by passing them as arguments to the constructor, or by setting them as attributes of the class. There are four sets of dataset specific arguments in the constructor: 
1. **`dataset` and `dataset_args`**: The dataset class and its arguments, which will be used to instantiate the training and validation datasets based on `val_size`. `val_size` is either an int specifying how many samples from the dataset are to be considered as the validation split or a float number between 0. and 1. specifying the portion of the data that should be used for validation. 

    If you want to use a different dataset for validation, you can specify it in the `val_dataset` and `val_dataset_args` arguments. The `dataset` should either be a dataset instance or a dataset class/import-path. If a dataset instance is provided as `dataset` then `dataset_args` are ignored.
2. **`train_dataset` and `train_dataset_args`**: Similar to `dataset` and `dataset_args` but only used for the training split (overrides `dataset` and `dataset_args`).
3. **`val_dataset` and `val_dataset_args`**: Similar to `train_dataset` and `val_dataset_args` but only used for the validation split.
4. **`test_dataset` and `test_dataset_args`**: Similar to `train_dataset` and `train_dataset_args` but only used for the test split with the exception that they don't override the `dataset*` arguments. This means that if you want a datamodule with a test split dataset and dataloader, you have to explicitly provide  

As a clarification, `dataset*` arguments are only used for the training and validation splits. If you
#### Importing datasets
You can tell `DataModule` to import your intended dataset classes by passing the import path as a string to any of the `*dataset` arguments. For example, if you want to use the `MNIST` dataset from `torchvision.datasets`, you can pass `dataset='torchvision.datasets.MNIST'` to the constructor. The `*dataset_args` arguments should be a dictionary of arguments to be passed to the dataset class. For example, if you want to use the `MNIST` dataset with `root` set to `data/`, you can pass `dataset_args={'root': 'data/'}` to the constructor.



### Validation Split
As mentioned before, the `val_size` argument can be either an int specifying how many samples from the dataset are to be considered as the validation split or a float number between 0. and 1. specifying the portion of the data that should be used for validation. The `seed` argument is used for the random split.

* If you want to use a different dataset for validation, you can specify it in the `val_dataset` and `val_dataset_args` arguments, where `val_dataset` should either be a dataset instance or a dataset class/import-path. If a dataset instance is provided as `val_dataset` then `val_dataset_args` are ignored.
* If You don't want to use a validation split, you can set `val_size=0`, and `val_batch_size=0`.
* When using `val_dataset`, `val_size` is automatically regarded as zero. Meaning that if a `dataset` is also provided, it is considered completely as the training split.

The following example creates a datamodule for the `MNIST` dataset and uses 20% of it as its validation split.
```python
from lightning_toolbox import DataModule

dm = DataModule(
    dataset='torchvision.datasets.MNIST', # You could have also specifed the class itself (torchvision.datasets.MNIST)
    dataset_args={'root': 'data/', 'download': True}, # All arguments to be passed to the dataset class constructor
    val_size=0.2, # 20% of the data will be used for validation, val_size=200 would have used 200 samples for validation
    seed=42, # The seed to be used for the random split
) # This datamodule does not have a test split
```

The following example creates a datamodule for the `MNIST` dataset and uses a different dataset for validation.
```python
from lightning_toolbox import DataModule

dm = DataModule(
    dataset='torchvision.datasets.MNIST', 
    dataset_args={'root': 'data/', 'download': True}, 
    val_dataset='torchvision.datasets.FashionMNIST', 
    val_dataset_args={'root': 'data/', 'download': True}, 
) # val_size is automatically regarded as zero and the dataset is considered completely as the `train_dataset`
```

### Configuring the Dataloaders
Similar to the `*dataset*` arguments, there are four arguments that could be set for the default behaiviour of the dataloaders: 1. `batch_size`, 2. `pin_memory`, 3. `shuffle`, and 4. `num_workers`. You can either set them directly or use a `train_`/`val_`/`test_` prefix to modify individual dataloader behaviours. There are a couple of notes to point out:
* `batch_size` is set to 16 by default.
* `pin_memory` is set to `True` by default.
* `train_shuffle` and `val_shuffle` and `test_shuffle` are by default set to True, False, and False respectively. Given the sensitivity of shuffling in training, there is no base `shuffle` argument and if you want to change the shuffling configuration of any of the dataloaders, you have to individually specify your intended configuration.
* `num_workers` (Number of workers) is set to zero by default. This means that the resulting datamodules are run on the same process/thread by default which could lead to performance bottlenecks. Make sure that you set these arguments suitable for your hardware and data (these articles could be helpful: [1](https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7), [2](https://pytorch-lightning.readthedocs.io/en/stable/guides/speed.html#num-workers), and [3](https://androidkt.com/how-to-assign-num_workers-to-pytorch-dataloader/))
* Needless to say, if a data-split is not present in the datamodule (e.g. `test_dataset` is not provided), the corresponding dataloader will not be available and its arguments will be ignored.

The following datamodule uses a batch size of 32 for the training split and 64 for the validation split, and uses 4 workers for the training split and 8 workers for the validation split.
```python
from lightning_toolbox import DataModule

dm = DataModule(
    ..., # dataset descriptions
    batch_size=32, # This will be used for every split (unless overriden)
    val_batch_size=64, # This will override batch_size for the validation split
    num_workers=4, # This will be used for all the splits (unless overriden)
    val_num_workers=8, # This will override num_workers for the validation split
)
```

## Extending Core functionality through Inheritance
You may wish to extend the base functionality of `lightning_toolbox.DataModule` perhaps to only use the dataloader configurations, or provide your own `perepare_data` method. The only thing to keep in mind is that, if you want to use the dataloader's functionality, your datamodule should have `train_data`, `val_data` and `test_data` specified by the time lightning want's to call the dataloaders functions.

If you wish to use dataset lookup functionality the lookup process is done through [`dypy.eval`](https://github.com/vahidzee/dypy#dynamic-evaluation) and made available through the static method `get_dataset`. In order to use higher-level functionalities such as dataset splitting or dataloaders configurations.

If you wish to setup your own dataset and then split it into validation and train, call `self.split_dataset()`, which by default takes place inside the fit stage of `setup`. You just need to set `self.train_data`, `self.val_data` and `self.dataset` accordingly.

