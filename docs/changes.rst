Changelog
=========

This document provides a brief summary of changes in each released
version of `powerpax`. More information and release builds are also
available on the `GitHub releases page
<https://github.com/karlotness/powerpax/releases>`__.

v0.2.0
------

* Require Python 3.10 or later.
* Fix bug in :func:`~powerpax.sliced_scan` and
  :func:`~powerpax.checkpoint_chunked_scan` for zero length inputs

v0.1.2
------
Fix deprecation warning from stray use of `jax.tree_map`.

v0.1.1
------
Fix bug in :func:`~powerpax.checkpoint_chunked_scan`. The issue caused
incorrect results in cases where `xs` is :pycode:`None` and
`chunk_size` does not evenly divide `length`.

v0.1.0
------
Initial release, includes:

* :func:`~powerpax.chunked_vmap`
* :func:`~powerpax.sliced_scan`
* :func:`~powerpax.checkpoint_chunked_scan`
* :func:`~powerpax.Static`
