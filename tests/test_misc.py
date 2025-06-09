# Copyright 2025 Karl Otness
# SPDX-License-Identifier: MIT


from powerpax._loop import clip
from hypothesis import given, strategies as st


@given(
    val=st.integers(),
    min_val=st.integers(),
    max_val=st.integers(),
)
def test_clip(val, min_val, max_val):
    clipped = clip(val, min_val, max_val)
    assert clipped >= min_val
    assert clipped <= max(min_val, max_val)
