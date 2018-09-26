import time
import sys
import os
import numpy as np
import pandas as pd
import scipy as sp
import h5py
import dask.dataframe as dd
from geno_reader import BedReader
from geno_reader import build_geno_query
from optparse import OptionParser
from genolmm import run_lmm_int
from genolmm import run_lmm

if __name__=='__main__':

    # define bed, phenotype and environment files
    bedfile = 'data_genolmm/chrom22_subsample20_maf0.10'
    phenofile = 'data_genolmm/expr.csv'
    envfile = 'data_genolmm/env.txt'

    # download data
    # cmd = 'wget http://www.ebi.ac.uk/~casale/data_structlmm.zip'
    # cmd = 'unzip data_structlmm.zip'

    # import geno and subset to first 1000 variants
    reader = BedReader(bedfile)
    query = build_geno_query(idx_start=0, idx_end=1000)
    reader.subset_snps(query, inplace=True)

    # pheno
    y = pd.DataFrame.from_csv(phenofile, index_col=0).T
    y = y['gene1'].values[:, sp.newaxis]

    # mean as fixed effect
    covs = sp.ones((y.shape[0], 1))
    E = sp.randn(y.shape[0], 2)

    # run analysis with standard lmm
    # pure environment is modelled as random effects
    res_lmm = run_lmm(reader, y, W=None,
                      covs=covs,
                      batch_size=100,
                      unique_variants=True)

    # run analysis with fixed-effect lmm
    # envs are modelled as random effects
    res_int = run_lmm_int(reader, y, E,
                          W=E,
                          covs=covs,
                          batch_size=100,
                          unique_variants=True)

    # export
    print 'Export'
    if not os.path.exists('out'):
        os.makedirs('out')
    res_int.to_csv('out/res_int.csv', index=False)
    res_lmm.to_csv('out/res_lmm.csv', index=False)
