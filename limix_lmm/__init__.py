r"""
*********
Limix-LMM
*********

Flexible Linear mixed model Toolbox for Genome-wide association studies

"""

from __future__ import absolute_import as _

from . import plot
from .lmm import LMM
from ._testit import test
from .lmm_core import LMMCore
from .mtlmm import MTLMM


__version__ = "0.0.1"

__all__ = ["LMM", "LMMCore", "MTLMM", "plot", "__version__", "test"]
