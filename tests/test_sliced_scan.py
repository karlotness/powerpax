# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import operator
import jax
import jax.numpy as jnp
import powerpax as ppx
from hypothesis import given, strategies as st


@given(
    length=st.integers(min_value=0, max_value=15),
    start=(st.integers(min_value=-31, max_value=31) | st.none()),
    stop=(st.integers(min_value=-31, max_value=31) | st.none()),
    step=(st.integers(min_value=-31, max_value=31).filter(bool) | st.none()),
    reverse=st.booleans(),
    use_xs_length=st.tuples(st.booleans(), st.booleans()).filter(any),
    unroll=st.integers(min_value=1, max_value=17),
)
def test_matches_scan(length, start, stop, step, reverse, use_xs_length, unroll):
    use_xs, use_length = use_xs_length

    def scan_fn(carry, x):
        return carry + 1, (carry, x)

    xs = jnp.arange(length) if use_xs else None
    extra_args = {"length": length} if use_length else {}
    jax_carry, jax_ys = jax.lax.scan(
        scan_fn, 0, xs, reverse=reverse, unroll=unroll, **extra_args
    )
    jax_ys = jax.tree_util.tree_map(
        operator.itemgetter(slice(start, stop, step)), jax_ys
    )
    ppx_carry, ppx_ys = jax.jit(
        lambda init, xs: ppx.sliced_scan(
            scan_fn,
            init,
            xs,
            reverse=reverse,
            unroll=unroll,
            start=start,
            stop=stop,
            step=step,
            **extra_args,
        )
    )(0, xs)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda la, lb: jnp.all(la == lb), jax_ys, ppx_ys)
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda la, lb: jnp.all(la == lb), jax_carry, ppx_carry)
    )
