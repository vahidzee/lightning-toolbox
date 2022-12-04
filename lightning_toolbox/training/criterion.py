import typing as th
import torch
import lightning
import functools
from .terms import CriterionTerm
from .struct_types import ResultsDict, FactorsDict, TermDescriptor


class Criterion:
    """Generic training objective abstraction for PyTorch Lightning.

    Criterion is a list of terms that are combined together to form the overall loss.
    Terms are either regularizations or objectives. The only difference between the two is that regularizations
    are processed after the original objective terms, and returned as a separate set of values in the results dict.

    The overall loss is computed as follows:
    - terms are processed and their values are stored in the results dict
    - regularizations are processed and their values are stored in the results dict
    - the overall loss is computed as the sum of the terms and regularizations, or the product of the terms and regularizations

    The abstraction for terms are CriterionTerm objects. Each term is a function that takes the following arguments:
    - *args, **kwargs: the arguments passed to the criterion
    - batch: the batch of data
    - training_module: the training module

    The term abstraction also allows for the computation of factors that are applied to the term values.

    The combination of the terms and the factors are controlled by the term abstraction itself, but the overall reduction
    of the terms is controlled by the criterion. The criterion can reduce the terms by summing them or multiplying them.

    Attributes:
        terms: List of terms to use in the training objective.
        regularizations: List of regularizations to use in the training objective.
        terms_reduction: Reduction method to use for terms.
        regularizations_reduction: Reduction method to use for regularizations.
        overall_reduction: Reduction method to use for overall loss.
    """

    def __init__(
        self,
        terms: th.Optional[th.Union[th.List[TermDescriptor], th.Tuple[TermDescriptor]]] = None,
        regularizations: th.Optional[th.Union[th.List[TermDescriptor], th.Tuple[TermDescriptor]]] = None,
        terms_reduction: str = "sum",  # sum or multiply
        regularizations_reduction: str = "sum",  # sum or multiply
        overall_reduction: str = "sum",  # sum or multiply
    ):
        self.terms = [
            CriterionTerm.from_description(
                term, factor_application="add" if terms_reduction == "multiply" else "multiply"
            )
            for term in terms
        ]
        self.regularizations = [
            CriterionTerm.from_description(
                term, factor_application="add" if regularizations_reduction == "multiply" else "multiply"
            )
            for term in (regularizations or [])
        ]
        self.rename_terms(terms=self.terms, prefix="term/")
        self.rename_terms(terms=self.regularizations, prefix="regularization/")
        self.terms_reduction = terms_reduction
        self.regularizations_reduction = regularizations_reduction
        self.overall_reduction = overall_reduction

    @staticmethod
    def rename_terms(terms: th.List[CriterionTerm], prefix: str = "") -> None:
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

    def process_term_results(
        self,
        *args,
        batch: th.Any = None,
        training_module: lightning.LightningModule = None,
        terms: th.Union[str, th.List[CriterionTerm]] = "terms",
        **kwargs,
    ):
        results = {}
        for term in getattr(self, terms) if isinstance(terms, str) else terms:
            term_results = term(*args, batch=batch, training_module=training_module, **kwargs)
            if isinstance(term_results, dict):
                for name in term_results:
                    # for loss, use the term name since it is the overall term value and should be reduced together with other terms
                    results[f"{term.name}/{name}" if name != "loss" else term.name] = term_results[name]
            else:
                results[term.name] = term_results
        return results

    def reduce(self, term_results: ResultsDict, factors_dict: FactorsDict, terms_name: str = "terms") -> torch.Tensor:
        reduction = getattr(self, f"{terms_name}_reduction")
        factors_applied_values = [
            term.apply_factor(term_value=term_results[term.name], factor_value=factors_dict[term.name])
            for term in getattr(self, terms_name)
        ]
        if reduction == "sum":
            return sum(factors_applied_values)
        elif reduction == "multiply":
            return functools.reduce(lambda x, y: x * y, factors_applied_values)

    @property
    def terms_names(self) -> th.List[str]:
        return [term.name for term in self.terms + self.regularizations]

    def __call__(
        self,
        *args,
        batch: th.Any = None,
        training_module: lightning.LightningModule = None,
        return_factors: bool = True,
        **kwargs,
    ) -> th.Union[ResultsDict, th.Tuple[ResultsDict, FactorsDict]]:
        results = self.process_term_results(
            *args, batch=batch, training_module=training_module, **kwargs, terms=self.terms + self.regularizations
        )
        factors = {
            term.name: term.factor_value(results_dict=results, training_module=training_module) for term in self.terms
        }
        results["loss"] = self.reduce(term_results=results, factors_dict=factors, terms_name="terms")
        factors = {
            **factors,
            **{
                term.name: term.factor_value(results_dict=results, training_module=training_module)
                for term in self.regularizations
            },
        }
        regularizations_reduced = self.reduce(term_results=results, factors_dict=factors, terms_name="regularizations")
        results["loss"] = (
            (results["loss"] + regularizations_reduced)
            if self.overall_reduction == "sum"
            else (results["loss"] * regularizations_reduced)
        )
        return results if not return_factors else (results, factors)
