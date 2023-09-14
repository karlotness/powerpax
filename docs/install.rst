Installation
============

.. highlight:: shell-session

This package depends on `JAX <https://github.com/google/jax>`__ which
requires special attention to enable support for GPUs, etc. Before
installing this package, follow the `JAX installation instructions
<https://github.com/google/jax#installation>`__.

PyPI
----

This package is available as "`powerpax
<https://pypi.org/project/powerpax/>`__" on PyPI. You can install it
using pip::

  $ python -m pip install powerpax

or on Windows:

.. code-block:: ps1con

   PS> py -m pip install powerpax

For more details on pip, consult its `user guide
<https://pip.pypa.io/en/stable/user_guide/>`__. Before installing it
is best to ensure your package manager is up to date (`upgrade pip
<https://pip.pypa.io/en/stable/installation/#upgrading-pip>`__).

Conda-Forge
-----------

This package can also be installed into a `Conda environment
<https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html>`__
from the `conda-forge <https://conda-forge.org/>`__ channel::

  $ conda install -c conda-forge powerpax

For more information, consult the `conda-forge documentation
<https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge>`__.
Installing JAX, particularly with GPU support may require additional
attention, consult the `JAX Conda install instructions
<https://github.com/google/jax#conda-installation>`__.

Source
------

The package can also be installed from source code, which is available
from our `GitHub repository <https://github.com/karlotness/powerpax>`__.

To install the package from source, navigate to the directory
containing your copy of the source code and run (note the trailing
``.``)::

  $ python -m pip install .

Running Tests
~~~~~~~~~~~~~

Once you have an installed version of the package, the test suite can
help confirm whether the routines are operating correctly. We use
`pytest <https://pytest.org/>`__ as a test runner. Once the package is
installed, navigate to the root directory of the repository and run::

  $ python -m pip install -r tests/requirements.txt
  $ python -m pytest
