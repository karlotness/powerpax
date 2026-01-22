# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import powerpax as ppx
import jax


def test_roundtrip_static():
    obj = ppx.Static(object())
    flat, tree = jax.tree_util.tree_flatten(obj)
    obj2 = jax.tree_util.tree_unflatten(tree, flat)
    assert obj == obj2
    assert hash(obj) == hash(obj2)


def test_static_under_jit():
    @jax.jit
    def f(arg1, arg2):
        return arg1 * ord(arg2.value)

    assert f(2, ppx.Static("a")) == 2 * ord("a")
    assert f(3, ppx.Static("b")) == 3 * ord("b")


def test_no_leaves():
    obj = ppx.Static(1.5)
    leaves = jax.tree_util.tree_leaves(obj)
    assert len(leaves) == 0
