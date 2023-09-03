# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import operator
import functools
import typing
import jax
import jax.numpy as jnp


C = typing.TypeVar("C", bound=typing.Callable[..., typing.Any])
A = typing.TypeVar("A", bound=tuple[object, ...])
K = typing.TypeVar("K", bound=dict[str, object])


def determine_splits(
    args: tuple[typing.Any, ...], kwargs: dict[str, typing.Any], chunk_size: int
) -> tuple[int, int]:
    # Determine leading axis size
    lead_sizes: set[int] = set()
    for leaf in jax.tree_util.tree_leaves((args, list(kwargs.values()))):
        if jnp.ndim(leaf) < 1:
            raise ValueError("attempted to vmap over array with zero dimensions")
        lead_sizes.add(jnp.shape(leaf)[0])
    if not lead_sizes:
        raise ValueError("vmap over empty arguments")
    elif len(lead_sizes) > 1:
        lead_size_str = ", ".join(map(str, lead_sizes))
        raise ValueError(f"inconsistent leading dimensions for vmap: {lead_size_str}")
    size = next(iter(lead_sizes))
    num_chunks, remainder = divmod(size, operator.index(chunk_size))
    return num_chunks, remainder


def split_args(
    args: A, kwargs: K, chunk_size: int, num_chunks: int, remainder: int
) -> tuple[tuple[A, K], typing.Union[tuple[A, K], tuple[None, None]]]:
    main_size = num_chunks * chunk_size
    leading_args, leading_kwargs = jax.tree_util.tree_map(
        lambda arr: arr[:main_size].reshape((num_chunks, chunk_size) + arr.shape[1:]),
        (args, kwargs),
    )
    # Remainder
    if remainder < 1:
        rem_args = None
        rem_kwargs = None
    else:
        rem_args, rem_kwargs = jax.tree_util.tree_map(
            operator.itemgetter(slice(-remainder, None)), (args, kwargs)
        )
    return (leading_args, leading_kwargs), (rem_args, rem_kwargs)


def chunked_vmap(fun: C, chunk_size: int) -> C:
    """Like :func:`jax.vmap` but limited to batches of `chunk_size` steps.

    This function behaves like :func:`jax.vmap` in that it vectorizes
    a function to apply over batches of inputs. However, unlike vmap
    which carries out all calculations in parallel, this version will
    perform at most `chunk_size` steps at once and uses
    :func:`jax.lax.scan` to loop over chunks.

    This is useful in cases where the calculations in `fun` involve
    large intermediate values which can exhaust available memory. With
    `chunked_vmap` it is possible to place an upper bound on peak
    memory use from the intermediate results while preserving some of
    the performance benefits of vmap, particularly on GPUs.
    """
    chunk_size = operator.index(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive (got {chunk_size})")

    def outer_scan(_, x):
        args, kwargs = x
        ret = jax.vmap(fun)(*args, **kwargs)
        return None, ret

    @functools.wraps(fun)
    def wrapped_fun(*args, **kwargs):
        num_chunks, remainder = determine_splits(args, kwargs, chunk_size)
        if num_chunks < 1 or (num_chunks == 1 and remainder == 0):
            # Trivial case: only one vmap needed
            return jax.vmap(fun)(*args, **kwargs)
        # Divide arguments into part for a scan, and a remainder chunk
        (leading_args, leading_kwargs), (rem_args, rem_kwargs) = split_args(
            args, kwargs, chunk_size, num_chunks, remainder
        )
        # Perform a scan for the main chunks
        _, ret = jax.lax.scan(
            outer_scan, None, (leading_args, leading_kwargs), length=num_chunks
        )
        ret = jax.tree_util.tree_map(
            lambda arr: arr.reshape((-1,) + arr.shape[2:]), ret
        )
        # Next, the remainder (if needed)
        if rem_args is not None and rem_kwargs is not None:
            ret = jax.tree_util.tree_map(
                lambda a, b: jnp.concatenate([a, b], axis=0),
                ret,
                jax.vmap(fun)(*rem_args, **rem_kwargs),
            )
        return ret

    return typing.cast(C, wrapped_fun)
