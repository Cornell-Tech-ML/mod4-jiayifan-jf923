from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    # Calculate the central difference for the given argument
    l1 = list(vals)
    l1[arg] += epsilon
    forward = f(*l1)
    l2 = list(vals)
    l2[arg] -= epsilon
    backward = f(*l2)
    return (forward - backward) / (2.0 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative from the given variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Return a unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf node in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Check if the variable is constant (not changing)."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the parents of the variable in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute gradients."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    L = []
    marked = set()

    def visit(curr_node: Variable) -> None:
        if curr_node.unique_id in marked or curr_node.is_constant():
            return
        for parent_node in curr_node.parents:
            visit(parent_node)
        marked.add(curr_node.unique_id)
        L.insert(0, curr_node)

    visit(variable)
    return L


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes. No return.
    Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    Args:
    ----
        variable: The variable for which to compute the derivative.
        deriv: The derivative to propagate back through the graph.

    """
    # TODO: Implement for Task 1.4.
    order = topological_sort(variable)
    id_list = [i.unique_id for i in order]
    d_dict = dict.fromkeys(id_list, 0.0)
    d_dict[variable.unique_id] = deriv

    for curr_node in order:
        d = d_dict[curr_node.unique_id]
        if curr_node.is_leaf():
            curr_node.accumulate_derivative(d)
        else:
            for var, d in curr_node.chain_rule(d):
                d_dict[var.unique_id] = d_dict.get(var.unique_id, 0.0) + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors."""
        return self.saved_values
