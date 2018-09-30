*******
Install
*******

Conda
^^^^^

TODO

Manual
^^^^^^

Limix-lmm requires scipy, numpy, pandas, limix-core, geno-sugar among other Python packages.
We will show here step-by-step  how to install the dependencies and limix-lmm.

* Create a new environment in conda_::

    conda create -n limix-lmm python=3.6
    source activate limix-lmm

* Install limix and R dependencies::

    conda install -c conda-forge limix-core numpy scipy pandas sphinx sphinx_rtd_theme

* Install limix-lmm::

    git clone https://github.com/limix/limix-lmm.git
    cd limix-lmm
    python setup.py install

* Build the documentation::

    cd doc
    make html

The documentation is in HTML and will be available at
``_build/html/index.html``.

.. _conda: https://conda.io
