
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

    val_size = ... # validation set size 
                   # (e.g. 0.2 of the provided dataset, or 1000 samples from it)
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