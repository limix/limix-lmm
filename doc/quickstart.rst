.. _python:

*********************
Quick Start in Python
*********************

GWAS with Linear Mixed Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We here show how to run structLMM and alternative linear
mixed models implementations in Python.

Before getting started, let's get some data::

    wget http://www.ebi.ac.uk/~casale/data_structlmm.zip
    unzip data_structlmm.zip

Now we are ready to go.

.. literalinclude:: examples/lmm_example.py
   :encoding: latin-1

The following script can be downloader :download:`here <lmm_example.py>`.


Multi-trait Linear Mixed Model (MTLMM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: examples/mtlmm_example.py
   :encoding: latin-1

The following script can be downloader :download:`here <mtlmm_example.py>`.


A full description of all methods can be found in :ref:`public`.

