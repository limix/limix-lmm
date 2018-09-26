import time
import sys
import os
import numpy as np
import pandas as pd
import scipy as sp
import h5py
import dask.dataframe as dd
import scipy.linalg as la
from geno_reader import BedReader
from geno_reader import build_geno_query
from geno_reader import GIter
from optparse import OptionParser
from genolmm.lmm import LMM
#from genolmm.utils.sugar_utils import *
import warnings

def run_lmm(reader,
            gp,
            batch_size=1000,
            filt_opt=None):
    """
    Utility function to run StructLMM

    Parameters
    ----------
    reader : :class:`limix.data.BedReader`
        limix bed reader instance.
    gp : :class:`limix_core.gp.GP`
        limix GP instance. The gp must be optimized on the null model.
    batch_size : int
        to minimize memory usage the analysis is run in batches.
        The number of variants loaded in a batch
        (loaded into memory at the same time).
    filt_opt : dict
        option used for the getGenotype method of :class:`limix.data.BedReader`

    Returns
    -------
    res : *:class:`pandas.DataFrame`*
        contains pv, effect size, standard error on effect size,
        and test statistcs as well as variant info.
    """
    # set more options in filt
    filt_opt['return_snpinfo'] = True
    filt_opt['impute'] = True

    # define lmm
    lmm = LMM(gp.mean.y, gp.mean.W, gp.covar)

    n_batches = reader.getSnpInfo().shape[0]/batch_size

    t0 = time.time()

    res = []
    for i, gr in enumerate(GIter(reader, batch_size=batch_size)):
        print('.. batch %d/%d' % (i, n_batches))

        X, _res = gr.getGenotypes(**filt_opt)
        if X.shape[1]==0:
            print('No variant survived the filters')
            continue

        # run lmm
        lmm.process(X)
        pv = lmm.getPv()
        beta = lmm.getBetaSNP()
        beta_ste = lmm.getBetaSNPste()
        lrt = lmm.getLRT()

        # add pvalues, beta, etc to res
        _res = _res.assign(pv=pd.Series(pv, index=_res.index))
        _res = _res.assign(beta=pd.Series(beta, index=_res.index))
        _res = _res.assign(beta_ste=pd.Series(beta_ste, index=_res.index))
        _res = _res.assign(lrt=pd.Series(lrt, index=_res.index))
        res.append(_res)

    res = pd.concat(res)
    res.reset_index(inplace=True, drop=True)

    t = time.time() - t0
    print('%.2f s elapsed' % t)

    return res
