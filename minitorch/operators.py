"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Return the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Check if one number is less than another."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if two numbers are equal."""
    return abs(x - y) < 1e-2


def max(x: float, y: float) -> float:
    """Return the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close in value."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Apply the ReLU activation function."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Calculate the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculate the exponential function."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the derivative of log times a second arg."""
    return d / x


def inv(x: float) -> float:
    """Calculate the reciprocal."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of reciprocal times a second arg."""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of ReLU times a second arg."""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable."""
    return lambda a: [fn(x) for x in a]


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Apply a function to two lists element-wise."""
    return lambda a, b: [fn(x, y) for x, y in zip(a, b)]


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function."""

    def reducer(ls: Iterable[float]) -> float:
        result = start
        for item in ls:
            result = fn(result, item)
        return result

    return reducer


def negList(a: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map."""
    return map(neg)(a)


def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith."""
    return zipWith(add)(a, b)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list using reduce."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce."""
    return reduce(mul, 1.0)(ls)
