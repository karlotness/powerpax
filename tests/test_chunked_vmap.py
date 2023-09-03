# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import jax
import jax.numpy as jnp
import powerpax as ppx
import pytest


@pytest.mark.parametrize("chunk_size", [1, 3, 5, 9, 10, 11])
def test_matches_vmap(chunk_size):
    def fun(arg):
        return arg * 2

    arr = jnp.arange(10)
    jax_res = jax.vmap(fun)(arr)
    chunked_res = jax.jit(ppx.chunked_vmap(fun, chunk_size=chunk_size))(arr)
    assert jax_res.dtype == chunked_res.dtype
    assert jax_res.shape == chunked_res.shape
    assert jnp.all(jax_res == chunked_res)


@pytest.mark.parametrize("chunk_size", [-1, 0])
def test_error_invalid_chunk_size(chunk_size):
    def fun(arg):
        return arg * 2

    with pytest.raises(ValueError):
        _ = ppx.chunked_vmap(fun, chunk_size=chunk_size)
