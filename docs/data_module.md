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
- [Internal behavior](#internal-behavior)
- [Examples](#examples)

## Usage
### Specifying Datasets

### Validation Split

### Configuring the Dataloaders

## Internal behavior

## Examples
