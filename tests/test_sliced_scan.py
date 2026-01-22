# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import operator
import jax
import jax.numpy as jnp
import powerpax as ppx
from hypothesis import given, settings, strategies as st


@st.composite
def length_slice_unroll(draw):
    length = draw(st.integers(min_value=0, max_value=15))
    sl = draw(st.slices(length).filter(lambda sl: abs(sl.step or 1) < 2**16))
    unroll = draw(st.integers(min_value=1, max_value=length + 1))
    return length, sl, unroll


@given(
    len_sl_unr=length_slice_unroll(),
    reverse=st.booleans(),
    use_xs_length=st.tuples(st.booleans(), st.booleans()).filter(any),
)
@settings(deadline=None)
def test_matches_scan(len_sl_unr, reverse, use_xs_length):
    length, sl, unroll = len_sl_unr
    use_xs, use_length = use_xs_length

    def scan_fn(carry, x):
        return carry + 1, {"carry": carry, "x": x}

    if use_xs:
        xs = {
            "val": jnp.arange(length, dtype=jnp.int32),
            "val2": jnp.arange(6 * length, dtype=jnp.uint16).reshape((length, 2, 3)),
        }
    else:
        xs = None
    extra_args = {"length": length} if use_length else {}
    jax_carry, jax_ys = jax.lax.scan(
        scan_fn, 0, xs, reverse=reverse, unroll=unroll, **extra_args
    )
    jax_ys = jax.tree_util.tree_map(operator.itemgetter(sl), jax_ys)
    ppx_carry, ppx_ys = jax.jit(
        lambda init, xs: ppx.sliced_scan(
            scan_fn,
            init,
            xs,
            reverse=reverse,
            unroll=unroll,
            start=sl.start,
            stop=sl.stop,
            step=sl.step,
            **extra_args,
        )
    )(0, xs)
    assert jax.tree_util.tree_structure(
        (jax_carry, jax_ys)
    ) == jax.tree_util.tree_structure((ppx_carry, ppx_ys))
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda a, b: jnp.array_equal(a, b) and a.dtype == b.dtype,
            (jax_carry, jax_ys),
            (ppx_carry, ppx_ys),
        )
    )
