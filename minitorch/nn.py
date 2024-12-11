from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw

    reshaped_tensor = input.contiguous().view(
        batch, channel, new_height, kh, new_width, kw
    )

    rearranged_tensor = reshaped_tensor.permute(0, 1, 2, 4, 3, 5)

    final_tensor = rearranged_tensor.contiguous().view(
        batch, channel, new_height, new_width, kh * kw
    )
    return final_tensor, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform average pooling with a specified kernel size.

    Args:
    ----
        input: Tensor with dimensions batch x channel x height x width.
        kernel: Tuple indicating the height and width of the pooling kernel.

    Returns:
    -------
        Tensor of dimensions batch x channel x new_height x new_width.

    """
    tiled_input, new_height, new_width = tile(input, kernel)
    pooled_tensor = tiled_input.sum(dim=-1) / (kernel[0] * kernel[1])

    return pooled_tensor.view(input.shape[0], input.shape[1], new_height, new_width)


# Task 4.4: Softmax and Dropout
max_reduce = FastOps.reduce(operators.max, float("-inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax along a specified dimension as a 1-hot tensor.

    Args:
    ----
        input: Tensor to compute argmax over.
        dim: Dimension to apply argmax.

    Returns:
    -------
        Tensor with the same shape as input, with 1 at the max value's position along dim, and 0 elsewhere.

    """
    max_vals = max_reduce(input, dim)
    return input == max_vals


class Max(Function):
    """Max operator for forward and backward passes."""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass: compute max values along dim."""
        dim_value = int(dim.item())
        max_vals = max_reduce(input, dim_value)
        ctx.save_for_backward(input, max_vals, dim_value)
        return max_vals

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass: compute gradients for max."""
        input, max_vals, dim_value = ctx.saved_values
        mask = input == max_vals
        return grad_output * mask, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the maximum values along a specified dimension."""
    return Max.apply(input, tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute softmax over a specified dimension.

    Args:
    ----
        input: Tensor to apply softmax.
        dim: Dimension to normalize.

    Returns:
    -------
        Tensor with softmax applied along dim.

    """
    exp_vals = (input - max_reduce(input, dim)).exp()
    return exp_vals / exp_vals.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute log-softmax over a specified dimension.

    Args:
    ----
        input: Tensor to apply logsoftmax.
        dim: Dimension to normalize.

    Returns:
    -------
        Tensor with log-softmax applied along dim.

    """
    max_vals = max_reduce(input, dim)
    shifted = input - max_vals
    log_sum_exp = shifted.exp().sum(dim).log()
    return shifted - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D max pooling with the specified kernel size.

    Args:
    ----
        input: Tensor with shape (batch, channel, height, width).
        kernel: Tuple specifying height and width of pooling kernel.

    Returns:
    -------
        Tensor of pooled output with reduced height and width.

    """
    tiled, new_height, new_width = tile(input, kernel)
    max_vals = max_reduce(tiled, 4)
    return max_vals.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor.

    Args:
    ----
        input: Tensor to apply dropout.
        rate: Dropout probability [0, 1).
        ignore: If True, no dropout is applied.

    Returns:
    -------
        Tensor with elements randomly zeroed out based on rate.

    """
    if ignore:
        return input

    if rate >= 1.0:
        return input.zeros()

    mask = rand(input.shape) > rate
    scale = 1.0 / (1.0 - rate)
    return input * mask * scale
