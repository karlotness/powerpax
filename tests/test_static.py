# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import powerpax as ppx
import jax


def test_roundtrip_static():
    obj = ppx.Static(1.5)
    flat, tree = jax.tree_util.tree_flatten(obj)
    obj2 = jax.tree_util.tree_unflatten(tree, flat)
    assert obj == obj2
    assert hash(obj) == hash(obj2)


def test_no_leaves():
    obj = ppx.Static(1.5)
    leaves = jax.tree_util.tree_leaves(obj)
    assert not leaves
