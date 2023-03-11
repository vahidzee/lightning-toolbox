# `lightning_toolbox.Objective`
An objective function is a construct of various objective terms, which along an application factor (term factor coefficient) result into a single scalar value. `lightning_toolbox.Objective` is an abstraction for building custom, customizable complex objective functions. Basing its main functionality on lazy instantiations made possible by [dypy](https://github.com/vahidzee/dypy), it is more than capable to provide fully customizable factors and term computations, making it an absolute gem if you intend to keep a minimalistic code base, and have a version control over your experiments (on top of experiment configurations). This is because using `lightning_toolbox.Objective`, you could decouple your objective function from your training procedure and model implementation, and provide it in your experiment configuration.

Let's start with a simple example. Suppose we have a simple linear regression model, and we want to train it using a simple mean squared error loss. We could define our objective function as follows:

```python
from lightning_toolbox.objective import Objective

objective = Objective(
    mse=lambda y, y_hat: (y - y_hat).pow(2).mean(),
)

objective(y=y, y_hat=y_hat) # returns the mean squared error
```

What about if we want to add a regularization term to our objective function? We could do that as follows:

```python
objective = Objective(
    mse=lambda y, y_hat: (y - y_hat).pow(2).mean(),
    reg=lambda w: w.pow(2).mean(),
)

objective(y=y, y_hat=y_hat, w=w) # returns the mean squared error + regularization
```

What if we want to add a term factor coefficient to our regularization term? We could do that as follows:

```python
objective = Objective(
    mse=lambda y, y_hat: (y - y_hat).pow(2).mean(),
    reg={"objective": lambda w: w.pow(2).mean(), "factor": 0.5},
)

objective(y=y, y_hat=y_hat, w=w) # returns the mean squared error + 0.5 * regularization
```

What if we want to add a term factor coefficient to our regularization term, but we want to provide it at runtime? We could do that as follows:

```python
objective = Objective(
    mse=lambda y, y_hat: (y - y_hat).pow(2).mean(),
    reg=lambda w: w.pow(2).mean(),
)

objective["reg"].factor = 0.5
```
