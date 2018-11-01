if __name__ == "__main__":
    import time
    import sys
    import os
    import numpy as np
    import pandas as pd
    import scipy as sp
    from optparse import OptionParser
    from limix_core.util.preprocess import gaussianize
    from limix_core.gp import GP2KronSumLR
    from limix_core.covar import FreeFormCov
    from limix_lmm.lmm_core import LMM
    from pandas_plink import read_plink
    import limix_core as lxc
    import geno_sugar as gs
    from sklearn.preprocessing import Imputer

    # import genotype file
    bedfile = "data_structlmm/chrom22_subsample20_maf0.10"
    (bim, fam, G) = read_plink(bedfile)

    # subsample snps
    Isnp = gs.is_in(bim, ("22", 17500000, 18000000))
    G, bim = gs.snp_query(G, bim, Isnp)

    # load phenotype file
    phenofile = "data_structlmm/expr.csv"
    dfp = pd.read_csv(phenofile, index_col=0)
    pheno = gaussianize(dfp.loc["gene1"].values[:, None])

    # mean as fixed effect
    covs = sp.ones((pheno.shape[0], 1))

    # fit null model
    wfile = "data_structlmm/env.txt"
    W = sp.loadtxt(wfile)
    W = W[:, W.std(0) > 0]
    W -= W.mean(0)
    W /= W.std(0)
    W /= sp.sqrt(W.shape[1])

    # larn a covariance on the null model
    gp = GP2KronSumLR(Y=pheno, Cn=FreeFormCov(1), G=W, F=covs, A=sp.ones((1, 1)))
    gp.covar.Cr.setCovariance(0.5 * sp.ones((1, 1)))
    gp.covar.Cn.setCovariance(0.5 * sp.ones((1, 1)))
    info_opt = gp.optimize(verbose=False)

    # define lmm
    lmm = LMM(pheno, covs, gp.covar.solve)

    # define geno preprocessing function
    impute = gs.preprocess.impute(
        Imputer(missing_values=np.nan, strategy="mean", axis=1)
    )
    standardize = gs.preprocess.standardize()
    preprocess = gs.preprocess.compose([impute, standardize])

    # loop on geno
    res = []
    n_analyzed = 0
    queue = gs.GenoQueue(G, bim, batch_size=200, preprocess=preprocess)
    for _G, _bim in queue:
        pv, beta = lmm.process(_G.T)
        _bim = _bim.assign(lmm_pv=pd.Series(pv, index=_bim.index))
        _bim = _bim.assign(lmm_beta=pd.Series(beta, index=_bim.index))
        res.append(_bim)
        n_analyzed += _G.shape[0]
        print(".. analysed %d/%d variants" % (n_analyzed, G.shape[0]))

    res = pd.concat(res)
    res.reset_index(inplace=True, drop=True)

    # export
    print("Export")
    if not os.path.exists("out"):
        os.makedirs("out")
    res.to_csv("out/res_lmm.csv", index=False)
