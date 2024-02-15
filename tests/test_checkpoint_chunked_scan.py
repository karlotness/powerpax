# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import jax
import jax.numpy as jnp
import powerpax as ppx
import pytest


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 5, 7, 15, 100])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize(
    "use_xs,use_length", [(True, True), (True, False), (False, True)]
)
def test_matches_scan(chunk_size, reverse, use_xs, use_length):
    def scan_fn(carry, x):
        return carry + 1, (carry, x)

    length = 15
    xs = jnp.arange(length) if use_xs else None
    extra_args = {"length": length} if use_length else {}
    jax_carry, jax_ys = jax.lax.scan(scan_fn, 0, xs, reverse=reverse, **extra_args)
    ppx_carry, ppx_ys = jax.jit(
        lambda init, xs: ppx.checkpoint_chunked_scan(
            scan_fn, init, xs, reverse=reverse, chunk_size=chunk_size, **extra_args
        )
    )(0, xs)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda la, lb: jnp.all(la == lb), jax_ys, ppx_ys)
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda la, lb: jnp.all(la == lb), jax_carry, ppx_carry)
    )
