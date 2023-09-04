Powerpax
========

A small collection of utility functions for JAX.

See the :doc:`reference section <reference>` for complete
documentation. These utilities build on functionality in JAX and
include:

:func:`chunked_vmap <powerpax.chunked_vmap>`
   Limit the number of vectorized steps evaluated in parallel which
   can reduce peak memory consumption with :func:`vmap <jax.vmap>`.

:func:`sliced_scan <powerpax.sliced_scan>`
   Keep a subset of iterations from a :func:`scan <jax.lax.scan>`
   without first storing all intermediate steps.

:class:`Static <powerpax.Static>`
   Wrap a value to treat it as a static member of a PyTree without
   defining a custom class.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   install
   reference
   license
