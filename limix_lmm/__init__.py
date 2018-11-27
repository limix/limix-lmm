r"""
*********
Limix-LMM
*********

Flexible Linear mixed model Toolbox for Genome-wide association studies

"""

from __future__ import absolute_import as _

from . import plot
from .lmm import LMM
from .lmm_core import LMMCore
from .mtlmm import MTLMM
from ._testit import test
from .sh import download, unzip


__version__ = "0.1.2"

__all__ = [
    "LMM",
    "LMMCore",
    "MTLMM",
    "plot",
    "download",
    "unzip",
    "__version__",
    "test",
]
