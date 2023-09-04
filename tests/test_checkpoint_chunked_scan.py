# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import jax
import jax.numpy as jnp
import powerpax as ppx
import pytest


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 5, 7, 15, 100])
@pytest.mark.parametrize("reverse", [False, True])
def test_matches_scan(chunk_size, reverse):
    def scan_fn(carry, x):
        return carry + 1, (carry, x)

    xs = jnp.arange(15)
    jax_carry, jax_ys = jax.lax.scan(scan_fn, 0, xs, reverse=reverse)
    ppx_carry, ppx_ys = jax.jit(
        lambda init, xs: ppx.checkpoint_chunked_scan(
            scan_fn, init, xs, reverse=reverse, chunk_size=chunk_size
        )
    )(0, xs)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda la, lb: jnp.all(la == lb), jax_ys, ppx_ys)
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda la, lb: jnp.all(la == lb), jax_carry, ppx_carry)
    )
