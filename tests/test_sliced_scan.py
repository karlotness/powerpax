# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import operator
import jax
import jax.numpy as jnp
import powerpax as ppx
import pytest


@pytest.mark.parametrize(
    "start,stop,step",
    [
        (None, None, None),
        (None, None, -1),
        (5, None, None),
        (None, -3, None),
        (-1, 0, 1),
        (0, -1, -1),
        (None, None, 2),
        (1, None, 2),
        (2, None, 2),
        (1, -3, 3),
        (-7, -2, 2),
        (2, 3, 4),
        (3, 2, -4),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_matches_scan(start, stop, step, reverse):
    def scan_fn(carry, x):
        return carry + 1, (carry, x)

    xs = jnp.arange(15)
    jax_carry, jax_ys = jax.lax.scan(scan_fn, 0, xs, reverse=reverse)
    jax_ys = jax.tree_util.tree_map(
        operator.itemgetter(slice(start, stop, step)), jax_ys
    )
    ppx_carry, ppx_ys = jax.jit(
        lambda init, xs: ppx.sliced_scan(
            scan_fn, init, xs, reverse=reverse, start=start, stop=stop, step=step
        )
    )(0, xs)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda la, lb: jnp.all(la == lb), jax_ys, ppx_ys)
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda la, lb: jnp.all(la == lb), jax_carry, ppx_carry)
    )
