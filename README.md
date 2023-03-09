# Lightning Toolbox: Automization tools for writing Pytorch Lightning experiments

Welcome to lightning-toolbox, a python package that offers a set of automation tools built on top of PyTorch Lightning. As a deep learning researcher, I found that PyTorch Lightning offloads a significant portion of redundant work. However, I still found myself spending a considerable amount of time on writing boilerplate code for datamodules, training/validation steps, logging, and specially for not so complicated costum training loops. This is why I created lightning-toolbox - to make it easier to focus on writing experiment-specific code rather than dealing with tedious setup tasks.

By passing your PyTorch model onto a generic lightning.LitModule (`lightning_toolbox.TrainingModule`), lightning-toolbox automatically populates the objective function, optimizer step, and more. In addition, lightning-toolbox's generic lightning.DataModule (`lightning_toolbox.DataModule`) can turn any PyTorch dataset, into a experiment-ready lightning data module, completing the cycle for writing lightning deep learning experiments.

Most of the functionality provided in this package is based on [dypy](https://github.com/vahidzee/dypy), which enables lazy evaluation of variables and runtime code injections. Although lightning-toolbox is currently in its early stages and mainly serves as a facilitator for my personal research projects, I believe it can be helpful for many others who deal with similar deep learning experiments. Therefore, I decided to open-source this project and continue to add on to it as I move further in my research.

As a disclaimer this package does not intend to solve field-specific problems and provides more generic facilitator codes that the official [lightning-bolts](https://lightning-bolts.readthedocs.io/en/latest/), that you can easily mold into your desired deep learning experiment.


**Table of Contents**:
- [Installation](#installation)
- [Usage](#usage)


## Installation

```bash
pip install lightning-toolbox
```
Lightning toolbox is tested on `lightning==1.9.0`, although there's no version restriction setup for this package, things might break down if the community decides to roll backward incompatible changes to the core Pytorch Lightning API (as they usually do).

## Usage
### TrainingModule & DataModule

```python
from lightning_toolbox import TrainingModule, DataModule
from lightning import Trainer

training_module = TrainingModule(
    # your model
    model = model, # or alternatively provide a model_cls and model_args instead as follows:
    # model_cls = `your_model_cls`, or an import path such as `torchvision.models.resnet18`
    # model_args = `dict` of arguments to pass to your model_cls constructor 
    #              (e.g. `pretrained=True`, for the case of resnet18)

    # objective function
    objective = ... # your objective function, factors, etc. (see the docs for more details)

    # optimizer (or a list of optimizers if you want to use different algorithms for 
    # different parts of your model)
    optimizer = ... # your optimizer (e.g. "torch.optim.AdamW"), you can provide
                    # optimizer_args to pass to the optimizer constructor
    lr = ... # learning rate (e.g. 1e-3)
    optimizer_args = ... # optimizer arguments (e.g. `dict(weight_decay=1e-2)`)
    optimizer_parameters = ... # parameters to optimize (default is "model.parameters()")
    optimizer_frequency = ... # how often to update the optimizer (e.g. 1)
    optimizer_is_active = ... # whether the optimizer is active (e.g. True, or a   
                              #function that returns a bool)
    # lr scheduler (or a list of lr schedulers)
    lr_scheduler = ... # your lr scheduler (e.g. "torch.optim.lr_scheduler.CosineAnnealingLR")
    lr_scheduler_args = ... # lr scheduler arguments (e.g. `dict(T_max=10)`)
    lr_scheduler_interval = ... # step/epoch
    lr_scheduler_monitor = ... # metric to monitor (e.g. "loss/val")
    lr_scheduler_frequency = ... # how often to update the lr (e.g. 1)
    lr_scheduler_optimizer = ... # the index of the optimizer this scheduler is for
    lr_scheduler_name = ... # name of the lr scheduler (e.g. "cosine")

    ... # other arguments (see the docs for more details)
)

data_module = DataModule(
    dataset = ... # your dataset, or an import path 
                  # (e.g. "torchvision.datasets.CIFAR10")
    dataset_args = ... # optionally dataset arguments 
                       # (e.g. `dict(root="data", train=True, download=True)`)

    val_size = ... # validation set size (e.g. 0.2 of the provided dataset, or 1000 samples from it)
    # val_dataset = ... # validation dataset 
                        # (e.g. "torchvision.datasets.MNIST") to use instead of val_size
    # val_dataset_args = ... # validation dataset arguments 
                             # (e.g. `dict(root="data", train=False, download=True)`)

    batch_size = ... # batch size (e.g. 32)
    num_workers = ... # number of workers (e.g. 4)
    pin_memory = ... # whether to use pinned memory (e.g. True)
    train_shuffle = ... # whether to shuffle the training set (e.g. True)
    val_shuffle = ... # whether to shuffle the validation set (e.g. False)
    ... # other arguments (see the docs for more details)
)

trainer = Trainer()
trainer.fit(training_module)
```