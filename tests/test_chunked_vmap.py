# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import jax
import jax.numpy as jnp
import powerpax as ppx
import pytest
from hypothesis import given, strategies as st


@st.composite
def length_chunk_size(draw):
    length = draw(st.integers(min_value=0, max_value=15))
    chunk_size = draw(st.integers(min_value=1, max_value=length + 1))
    return length, chunk_size


@given(
    len_cs=length_chunk_size(),
    use_args_kwargs=st.tuples(st.booleans(), st.booleans()).filter(any),
)
def test_matches_vmap(len_cs, use_args_kwargs):
    length, chunk_size = len_cs
    use_args, use_kwargs = use_args_kwargs

    def fun(arg=1, arg2=1):
        return arg * arg2

    arr = jnp.arange(length)
    args = (arr,) if use_args else ()
    kwargs = {"arg2": arr} if use_kwargs else {}
    jax_res = jax.vmap(fun)(*args, **kwargs)
    chunked_res = jax.jit(ppx.chunked_vmap(fun, chunk_size=chunk_size))(*args, **kwargs)
    assert jax_res.dtype == chunked_res.dtype
    assert jax_res.shape == chunked_res.shape
    assert jnp.all(jax_res == chunked_res)


@pytest.mark.parametrize("chunk_size", [-1, 0])
def test_error_invalid_chunk_size(chunk_size):
    def fun(arg):
        return arg * 2

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        _ = ppx.chunked_vmap(fun, chunk_size=chunk_size)
