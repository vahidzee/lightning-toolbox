import torch
import functools
import lightning
import dycode as dy
import typing as th
from .struct_types import ResultsDict, FactorsDict, TermDescriptor


class CriterionTerm:
    def __init__(
        self,
        name: str = None,
        factor: th.Optional[th.Union[float, dy.FunctionDescriptor]] = None,
        scale_factor: th.Optional[str] = None,
        term_function: th.Optional[dy.FunctionDescriptor] = None,
        factor_application: str = "multiply",  # multiply or add
        **kwargs,  # function description dictionary
    ) -> None:
        """
        Term is a single value that is used to compute the objective function (e.g. MSE, KL, etc.).
        Terms can be combined into a Criterion, which is a collection of terms that are summed together.

        You can use the Term class in two ways:
        1. You can pass a function that computes the term value, and optionally a factor that scales the term value.
        2. You can inherit from the CriterionTerm class and implement the __call__ method and optionally other methods.

        Args:
            name (str): Name of the term. If not provided, the name will be the name of the function.
            factor (float or FunctionDescriptor): Factor that scales the term value.
                this can be a single float value, or a function that takes the results dictionary (
                dict[str, torch.Tensor] containing the results of other terms in the objective function)
                and the training_module (pytorch_lightning.LightningModule) and returns a float value.

                Example:
                    1. factor annealed with the reciprocal of the number of epochs:
                    >>> factor = "lambda results_dict, training_module: 1/training_module.trainer.current_epoch"

                    2. factor with twice the value of the term "mse":
                    >>> factor = "lambda results_dict, training_module: 2*results_dict['term/mse']"

                Use this option only if you don't want to inherit from the CriterionTerm class. In general, it is
                recommended to use this option since it provides most of the functionality one requires for applying
                a factor to a term.

            scale_factor (str): Name of the term that is used to scale the factor. This is useful when you are combining
                multiple terms of different scales (e.g. MSE and KL) and you want to scale the factor of the KL term so that
                it has the same scale as the MSE term. The factor will be scaled by the ratio of the term value to the
                scale_factor term value.

                Example:
                    >>> scale_factor = "term/mse"
                    This will scale the factor by the ratio of the term value to the MSE term value. Which is:
                        factor = factor * mse_term_value.detach() / term_value.detach()

                If you
            term_function (FunctionDescriptor): Function that computes the term value. This can be a function that takes
                the training_module, batch, [results_dict], and any other argument and returns a torch.Tensor.
                This function will be processed using `dycode.eval`. See `dycode.eval` documentation for more details.

            factor_application (str): How the factor is applied to the term value. Can be "multiply" or "add".

        """
        self.name = name or self.__class__.__name__
        self._factor_application = factor_application
        self._factor_description = factor
        self.initialize_factor_attributes(factor_application, factor)
        self._term_function_description = term_function or kwargs
        self._scale_factor = scale_factor

    def initialize_factor_attributes(
        self,
        factor_application: th.Optional[str] = None,
        factor_description: th.Optional[th.Union[float, dy.FunctionDescriptor]] = None,
    ):
        self._factor_application = factor_application or self._factor_application
        self._factor_description = factor_description or self._factor_description
        if self._factor_application == "multiply" and self._factor_description is None:
            self._factor_description = 1.0
        if self._factor_application == "add" and self._factor_description is None:
            self._factor_description = 0.0

    @functools.cached_property
    def _compiled_factor(self):
        compiled = dy.eval_function(self._factor_description, function_of_interest="factor")
        return dy.dynamic_args_wrapper(compiled) if callable(compiled) else compiled

    def factor_value(self, results_dict: ResultsDict, training_module: lightning.LightningModule) -> torch.Tensor:
        """
        Computes the final factor value to be applied to the term value.
        By default this is a wrapper around the `factor` (function/float) that is passed to the term constructor.

        It does two things:
           1. Gets the factor value (number) from the `factor` (either by calling the function, or by returning the number)
           2. Applies the factor scaling based on other terms in the results_dict (if `scale_factor` is provided)

        If you want to implement a custom factor value, you can override this method in your subclass. If you intent
        to support the `scale_factor` option, you should call `self.apply_scale_factor` in your implementation.

        Args:
            results_dict (ResultsDict): Dictionary containing the results of other terms in the objective function.
                if this dictionary is proccessed by the `Criterion` class it will contain `term/<term_name>` and
                `regularization/<term_name>` entries for each term and regularization in the criterion.

            training_module (lightning.LightningModule): The training module that is being trained.

        Returns:
            number: The factor value to be applied to the term value.
        """
        factor_value = (
            self._compiled_factor(results_dict=results_dict, training_module=training_module)
            if callable(self._compiled_factor)
            else self._compiled_factor
        )
        return self.apply_scale_factor(factor_value, results_dict, training_module)

    def apply_scale_factor(self, term_value, results_dict, training_module=None):
        """
        Applies the scale factor to the term value. This is used to scale the factor value by the ratio of some other term value

        Args:
            term_value (number): The term value to be scaled.
            results_dict (ResultsDict): Dictionary containing the results of other terms in the objective function.
            training_module (lightning.LightningModule): The training module that is being trained.

        Returns:
            number: The scaled term value.
        """
        if self._scale_factor:
            return term_value * results_dict[self._scale_factor].data.clone() / results_dict[self.name].data.clone()
        else:
            return term_value

    def apply_factor(
        self,
        term_value: th.Optional[torch.Tensor] = None,
        factor_value: th.Optional[torch.Tensor] = None,
        term_results: th.Optional[ResultsDict] = None,
        training_module: th.Optional[lightning.LightningModule] = None,
    ) -> torch.Tensor:
        """
        Applies the factor to the term value. Based on the `factor_application` attribute, this can be either
        multiplication or addition of the factor to the term value.

        Override this method if you want to implement a custom factor application.

        Args:
            term_value (torch.Tensor): The term value to be scaled.
                If no term value is provided the term value will be looked up in the `term_results` dictionary.
            factor_value (torch.Tensor): The factor value to be applied to the term value.
                If no factor value is provided the factor value will be computed using `self.factor_value`.
            term_results (ResultsDict): Dictionary containing the results of this term, and other terms in the objective function.
            training_module (lightning.LightningModule): The training module that is being trained.

        Returns:
            torch.Tensor: The scaled term value.
        """
        factor_value = (
            self.factor_value(results_dict=term_results, training_module=training_module)
            if factor_value is None
            else factor_value
        )
        term_value = term_results[self.name] if term_value is None else term_value
        if self._factor_application == "multiply":
            return term_value * factor_value
        elif self._factor_application == "add":
            return term_value + factor_value
        else:
            raise ValueError(f"Unknown factor application {self._factor_application}")

    @functools.cached_property
    def _compiled_term_function(self):
        """
        Compiled version of the term function provided to the constructor in the `term_function` argument.

        The function descriptor is processed using `dycode.eval` and the result is cached using `functools.cached_property`.
        By setting dynamic_args=True when evaluating the function, the function will be wrapped in a `dynamic_args_wrapper`
        that will allow it to be called with any arguments.
        """
        return dy.eval(self._term_function_description, function_of_interest="term", strict=False, dynamic_args=True)

    def __call__(
        self, *args, batch: th.Any = None, training_module: lightning.LightningModule = None, **kwargs
    ) -> torch.Tensor:
        """
        Computes the term value. This is the main method of the `Term` class. It is called by the `Criterion` class
        when computing the objective function value.

        If this method is not overridden, it will call the `term_function` that was provided to the constructor.
        The `term_function` should have the signature `term_function(batch, training_module, *args, **kwargs)`.
        The `batch` and `training_module` arguments are provided by the `Criterion` class, and the `*args` and `**kwargs`
        are provided by the user.

        Args:
            *args: Additional arguments to be passed to the `term_function`.
            batch (th.Any): The batch of data that is being processed.
            training_module (lightning.LightningModule): The training module that is being trained.
            **kwargs: Additional keyword arguments to be passed to the `term_function`.

        Note:
            1. This function is only responsible for computing the term value. The `factor` and `factor_application`
            are applied in the `apply_factor` method.

            2. If you want to implement a custom term value, you can override this method in your subclass. But keep in mind
            that factor application should be decoupled from the term value computation, and is handled by the `apply_factor`
            method.

        Returns:
            torch.Tensor: The term value.
        """
        if not self._term_function_description:
            raise NotImplementedError
        # todo: support for multiple outputs
        return self._compiled_term_function(*args, batch=batch, training_module=training_module, **kwargs).mean()

    @staticmethod
    def from_description(
        description: th.Union["CriterionTerm", TermDescriptor],
        # overwrite attributes of the instance
        name: th.Optional[str] = None,
        factor_application: th.Optional[str] = None,  # multiply or addÍ
    ) -> "CriterionTerm":
        """
        Creates a `CriterionTerm` instance from a `TermDescriptor` object.

        Args:
            description (TermDescriptor): The term descriptor.
            name (str): The name of the term. If not provided, the name from the description will be used.
            factor_application (str): The factor application. If not provided, the factor application from the description will be used.

        Returns:
            CriterionTerm: The criterion term.
        """
        if isinstance(description, CriterionTerm):
            term = description
        elif isinstance(description, str):
            try:
                term = dy.eval(description, dynamic_args=True)()
            except:
                term = CriterionTerm(term_function=description, name=name, factor_application=factor_application)
        # else the description is a dict
        # checking if the description provides a class_path to instantiate a previously defined term
        elif "class_path" in description:
            term = dy.eval(description["class_path"])(**description.get("init_args", dict()))
        # else the description is a dict with required fields to instantiate a new term
        else:
            term = CriterionTerm(**description)
        if name is not None:
            term.name = name
        if factor_application is not None:
            term.initialize_factor_attributes(factor_application=factor_application)
        return term
