r"""
*********
Limix-LMM
*********

Flexible Linear mixed model Toolbox for Genome-wide association studies

"""

from __future__ import absolute_import as _absolute_import
from . import plot
from .lmm_core import LMM
from ._testit import test

__version__ = "0.0.1"

__all__ = ["LMM", "plot", "__version__", "test"]
