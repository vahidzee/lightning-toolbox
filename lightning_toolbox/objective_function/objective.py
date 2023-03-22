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
import torch
import dypy as dy
from .terms import ObjectiveTerm
import warnings

# register torch with dypy
dy.register_context(torch)

# types
ResultsDict = th.Dict[str, torch.Tensor]
FactorsDict = th.Dict[str, torch.Tensor]

TermDescriptor = th.Union[str, th.Dict[str, th.Any], dy.FunctionDescriptor]


class Objective:
    """Generic training objective function abstraction.

    ObjectiveFunction is a collection of terms that are combined together to form the overall loss/objective function.

    The overall loss/objective value is computed as follows:
    - terms are processed and their values are stored in the results dict
    - results of the terms are combined together to form the

    The abstraction for terms are ObjectiveTerm objects. Each term is a function that by default has a freeform signature
    `*args, **kwargs`, which takes on all the arguments passed on to the objective function. The term abstraction allows
    for the specification of the arguments that are passed to the term function. The term abstraction also allows for
    the specification of the name of the term. The term abstraction also allows for the computation of factors that are
    applied to the term values.

    The combination of the terms and the factors are controlled by the term abstraction itself, but the overall reduction
    of the terms is controlled by the objective, which by default is the sum of the term values.

    Attributes:
        terms: List of terms to use in the objective function.
        latch: A dictionary that can be used to store values for later use. The values are cleared after each call to
            __call__.
        results_latch: A dictionary that stores the results of the terms. The values are cleared after each call to
            __call__.
        factors_latch: A dictionary that stores the factors of the terms. The values are cleared after each call to
            __call__. The factors are computed by the terms themselves, and just stored here for convenience.
    """

    latch: th.Dict[th.Any, th.Any]

    def __init__(
        self,
        *term_args: TermDescriptor,
        **term_kwargs: TermDescriptor,
    ) -> None:
        term_args = [ObjectiveTerm.from_description(term, objective=self) for term in term_args]
        term_kwargs = [
            ObjectiveTerm.from_description(term, objective=self, name=name) for name, term in term_kwargs.items()
        ]
        self.terms: th.List[ObjectiveTerm] = term_args + term_kwargs
        # to make sure all terms have unique names
        self.__rename_terms(terms=self.terms)
        # initialize the latches
        self.inputs_latch, self.latch, self.results_latch, self.factors_latch = {}, {}, {}, {}

    @property
    def results(self) -> ResultsDict:
        return self.results_latch

    @property
    def factors(self) -> FactorsDict:
        return self.factors_latch

    def remember(self, **kwargs) -> None:
        """Keep some values for later use. Get's cleared after each call to __call__"""
        self.latch.update(kwargs)

    def _forget(self) -> None:
        """Forget all remembered values and clear all latches."""
        self.latch.clear()
        self.results_latch.clear()
        self.factors_latch.clear()

    def __getitem__(self, key: th.Union[str, int]) -> ObjectiveTerm:
        if isinstance(key, int):
            return self.terms[key]
        for term in self.terms:  # TODO: use a dict for faster lookup
            if term.name == key:
                return term
        raise KeyError(f"Term with name {key} not found.")

    @property
    def terms_names(self) -> th.List[str]:
        return [term.name for term in self.terms]

    @staticmethod
    def __rename_terms(terms: th.List[ObjectiveTerm], prefix: str = "") -> None:
        names_count = {term.name: 0 for term in terms}
        for term in terms:
            names_count[term.name] += 1
        for name in names_count:
            names_count[name] = names_count[name] if names_count[name] > 1 else -1
        for term in terms[::-1]:
            names_count[term.name] -= 1
            term.name = (
                f"{prefix}{term.name}"
                if names_count[term.name] < 0
                else f"{prefix}{term.name}/{names_count[term.name]}"
            )
            term.name = term.name.replace("__", "/")

    def process_terms_results(self, *args, **kwargs) -> None:
        """
        Call all the terms and store their results in the results latch.

        Terms are called in the order they are provided to the objective. When a term is called, everything in provided
        to the objective is also passed to it. The results of the term are either a single value or a dict of values.
        If the results are a dict, and the term is to contribute to the overall loss, it has to contain a key "loss".

        All of the returned keys are stored in the results latch, but only the terms that provide a "loss" key (or
        have a single scalar value returned) are used to compute the overall loss. Other returned values can be used
        for logging or other purposes.

        Args:
            *args, **kwargs: the arguments passed to the objective are directly passed to the terms.
        """
        if len(self.terms) == 0:
            warnings.warn("No terms provided to the objective. Undefined behaviour might occur", RuntimeWarning)

        for term in self.terms:
            term_results = term(*args, **kwargs)
            if isinstance(term_results, dict):
                for name in term_results:
                    # for loss, use the term name since it is the overall term value and should be reduced together with other terms
                    self.results_latch[f"{term.name}/{name}" if name != "loss" else term.name] = term_results[name]
            else:
                self.results_latch[term.name] = term_results

    def reduce(self) -> torch.Tensor:
        """
        Reduce the term results to a single value by applying the factors and summing them together.
        Override this method to change the reduction function.

        This method assumes that the term results are processed and ready in `self.results_latch`, and the factors
        are ready in `self.factors_latch`.

        Returns:
            The reduced loss/objective value.
        """
        factors_applied_values = [
            term.apply_factor(term_value=self.results_latch[term.name], factor_value=self.factors_latch[term.name])
            for term in self.terms
            # only reduce terms that contribute to the loss
            if term.name in self.results_latch
        ]
        return sum(factors_applied_values)

    def __call__(
        self, *args, return_factors: bool = False, **kwargs
    ) -> th.Union[ResultsDict, th.Tuple[ResultsDict, FactorsDict]]:
        self._forget()  # clear the latches
        self.inputs_latch.update(kwargs)  # save all the input arguments in the latch
        self.process_terms_results(*args, **kwargs)  # process the terms
        self.factors_latch = {
            term.name: term.factor(*args, **kwargs) for term in self.terms if term.name in self.results_latch
        }  # compute the factor values for the terms that contribute to the loss
        self.results_latch["loss"] = self.reduce()  # reduce the term results with the factors
        return self.results_latch if not return_factors else (self.results_latch, self.factors_latch)
