# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import operator
import functools
import typing
import jax
import jax.numpy as jnp


C = typing.TypeVar("C")
X = typing.TypeVar("X")
Y = typing.TypeVar("Y")


def get_leaf_length(leaf: jax.Array) -> int:
    if jnp.ndim(leaf) < 1:
        raise ValueError("pytree array leaf is zero-dimensional, cannot scan")
    return operator.index(jnp.shape(leaf)[0])


def get_target_length(xs: object, length: typing.Optional[int]) -> int:
    leaf_lengths = {get_leaf_length(x) for x in jax.tree_util.tree_leaves(xs)}
    if len(leaf_lengths) > 1:
        size_str = ", ".join(map(str, leaf_lengths))
        raise ValueError(f"inconsistent lengths for tree input: {size_str}")
    if length is not None:
        length = operator.index(length)
        if length < 0:
            raise ValueError(f"invalid length {length}, must not be negative")
        leaf_lengths.add(length)
    if not leaf_lengths:
        raise ValueError("no input values provided and no specified length")
    elif len(leaf_lengths) > 1:
        size_str = ", ".join(map(str, leaf_lengths))
        raise ValueError(f"ambiguous lengths for scan: {size_str}")
    return leaf_lengths.pop()


def compute_slices(
    start: int, step: int, num_ys: int, target_length: int, reverse: bool
) -> tuple[slice, typing.Optional[int], slice, int, slice]:
    start_index = start
    end_index = start + (step * (num_ys - 1))
    if step > 0:
        start_skip = start_index
        end_skip = target_length - end_index - 1
    else:
        start_skip = end_index
        end_skip = target_length - start_index - 1
    if reverse:
        start_skip, end_skip = end_skip, start_skip
    if num_ys == 1:
        # Special case for 1 output (do the core without a scan)
        pre_step = True
        outer_steps = 0
    elif start_skip < abs(step) - 1:
        # We have to separate out the first step
        pre_step = True
        outer_steps = num_ys - 1
    else:
        # We can merge the first step with part of the start skip
        start_skip -= abs(step) - 1
        pre_step = False
        outer_steps = num_ys
    # Compute slices
    if reverse:
        start_slice = slice(target_length - start_skip, None)
        pre_slice = target_length - start_skip - 1 if pre_step else None
        core_slice = slice(
            end_skip, target_length - start_skip - (1 if pre_step else 0)
        )
        end_slice = slice(None, end_skip)
    else:
        start_slice = slice(None, start_skip)
        pre_slice = start_skip if pre_step else None
        core_slice = slice(
            start_skip + (1 if pre_step else 0), target_length - end_skip
        )
        end_slice = slice(target_length - end_skip, None)
    return start_slice, pre_slice, core_slice, outer_steps, end_slice


def sliced_scan(
    f: typing.Callable[[C, X], tuple[C, Y]],
    init: C,
    xs: X,
    length: typing.Optional[int] = None,
    reverse: bool = False,
    unroll: int = 1,
    *,
    start: typing.Optional[int] = None,
    stop: typing.Optional[int] = None,
    step: typing.Optional[int] = None,
) -> tuple[C, Y]:
    def skip_step(carry, x):
        new_carry, _ = f(carry, x)
        return new_carry, None

    target_length = get_target_length(xs, length)
    unroll = min(max(1, operator.index(unroll)), target_length)
    start, stop, step = slice(start, stop, step).indices(target_length)
    num_ys = len(range(start, stop, step))

    if num_ys == target_length:
        # Normal scan
        carry, ys = jax.lax.scan(
            f,
            init,
            xs,
            length=target_length,
            reverse=reverse,
            unroll=unroll,
        )
        if step < 0:
            ys = jax.tree_util.tree_map(functools.partial(jnp.flip, axis=0), ys)
        return carry, ys
    elif num_ys == 0:
        # No core iterations. Loop for the carry and produce empty ys
        carry, _ = jax.lax.scan(
            skip_step,
            init,
            xs,
            length=target_length,
            reverse=reverse,
            unroll=unroll,
        )

        def dummy_scan(init, xs):
            _, ys = jax.lax.scan(
                f, init, xs, length=length, reverse=reverse, unroll=unroll
            )
            return ys

        ys = jax.tree_util.tree_map(
            lambda sd: jnp.zeros((0,) + sd.shape[1:], dtype=sd.dtype),
            jax.eval_shape(dummy_scan, init, xs),
        )
        return carry, ys

    def scan_y_to_carry(carry, x):
        old_carry, _ = carry
        new_carry, new_y = f(old_carry, x)
        return (new_carry, new_y), None

    def inner_scan(carry, xs):
        dummy_y = jax.tree_map(
            lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype),
            jax.eval_shape(
                lambda carry, x: f(carry, x)[1],
                carry,
                jax.tree_util.tree_map(operator.itemgetter(0), xs),
            ),
        )
        (carry, y), _ = jax.lax.scan(
            scan_y_to_carry,
            (carry, dummy_y),
            xs,
            length=abs(step),
            reverse=reverse,
            unroll=min(unroll, abs(step)),
        )
        return carry, y

    start_slice, pre_slice, core_slice, outer_steps, end_slice = compute_slices(
        start=start,
        step=step,
        num_ys=num_ys,
        target_length=target_length,
        reverse=reverse,
    )

    def do_skip_steps(carry: C, slicer: slice) -> C:
        length = len(range(*slicer.indices(target_length)))
        if length <= 0:
            new_carry = carry
            return carry
        elif length == 1:
            # Call function directly
            new_carry, _ = jax.jit(skip_step)(
                carry, jax.tree_util.tree_map(operator.itemgetter(slicer.start), xs)
            )
        else:
            new_carry, _ = jax.lax.scan(
                skip_step,
                carry,
                jax.tree_util.tree_map(operator.itemgetter(slicer), xs),
                length=length,
                reverse=reverse,
                unroll=min(unroll, length),
            )
        return new_carry

    carry = do_skip_steps(init, start_slice)
    if pre_slice is not None:
        carry, pre_y = jax.jit(f)(
            carry, jax.tree_util.tree_map(operator.itemgetter(pre_slice), xs)
        )
        pre_y = jax.tree_util.tree_map(
            functools.partial(jnp.expand_dims, axis=0), pre_y
        )
    else:
        pre_y = None
    # Do core steps
    if outer_steps > 0:
        if abs(step) > 1:
            # Requires a nested scan
            def leaf_reshape(leaf):
                return leaf[core_slice].reshape(
                    (outer_steps, abs(step)) + leaf.shape[1:]
                )

            carry, ys = jax.lax.scan(
                inner_scan,
                carry,
                jax.tree_util.tree_map(leaf_reshape, xs),
                length=outer_steps,
                reverse=reverse,
                unroll=min(outer_steps, max(1, unroll // abs(step))),
            )
        else:
            # No nested scan required
            carry, ys = jax.lax.scan(
                f,
                carry,
                jax.tree_util.tree_map(operator.itemgetter(core_slice), xs),
                length=outer_steps,
                reverse=reverse,
                unroll=min(outer_steps, max(1, unroll)),
            )
        # Concatenate if necessary
        if pre_y is not None:
            ys = jax.tree_util.tree_map(
                lambda a, b: jnp.concatenate([a, b] if not reverse else [b, a], axis=0),
                pre_y,
                ys,
            )
    else:
        ys = typing.cast(Y, pre_y)
    # Do end steps
    carry = do_skip_steps(carry, end_slice)
    if step < 0:
        ys = jax.tree_util.tree_map(functools.partial(jnp.flip, axis=0), ys)
    return carry, ys
