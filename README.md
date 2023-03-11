# LightningToolbox: A PyTorch Lightning Facilitator
<p align="center">
  <a href="#installation">Installation</a> •
  <a href="https://github.com/vahidzee/lightning-toolbox/tree/main/docs/README.md">Docs</a> •
  <a href="#license">License</a>
</p>

<p align="center" markdown="1">
    <a href="https://badge.fury.io/py/lightning_toolbox"><img src="https://badge.fury.io/py/lightning_toolbox.svg" alt="PyPI version" height="18"></a>
</p>


Welcome to lightning-toolbox, a python package that offers a set of automation tools built on top of PyTorch Lightning. As a deep learning researcher, I found that PyTorch Lightning offloads a significant portion of redundant work. However, I still found myself spending a considerable amount of time on writing boilerplate code for datamodules, training/validation steps, logging, and specially for not so complicated costum training loops. This is why I created lightning-toolbox - to make it easier to focus on writing experiment-specific code rather than dealing with tedious setup tasks.

By passing your PyTorch model onto a generic `lightning.LightningModule` (`lightning_toolbox.TrainingModule`), lightning-toolbox automatically populates the objective function, optimizer step, and more. In addition, lightning-toolbox's generic `lightning.LightningDataModule` (`lightning_toolbox.DataModule`) can turn any PyTorch dataset, into a experiment-ready lightning data module, completing the cycle for writing lightning deep learning experiments.

Most of the functionality provided in this package is based on [dypy](https://github.com/vahidzee/dypy), which enables lazy evaluation of variables and runtime code injections. Although lightning-toolbox is currently in its early stages and mainly serves as a facilitator for my personal research projects, I believe it can be helpful for many others who deal with similar deep learning experiments. Therefore, I decided to open-source this project and continue to add on to it as I move further in my research.

As a disclaimer this package does not intend to solve field-specific problems and provides more generic facilitator codes that the official [lightning-bolts](https://lightning-bolts.readthedocs.io/en/latest/), that you can easily mold into your desired deep learning experiment.

## Installation

```bash
pip install lightning-toolbox
```
Lightning toolbox is tested on `lightning==1.9.0`, although there's no version restriction setup for this package, things might break down if the community decides to roll backward incompatible changes to the core Pytorch Lightning API (as they usually do).

## License

This project is licensed under the terms of the Apache 2.0 license. See [LICENSE](https://github.com/vahidzee/lightning-toolbox/tree/main/LICENSE) for more details.