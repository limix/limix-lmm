.. _python:

*********************
Quick Start in Python
*********************

GWAS with Linear Mixed Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We here show how to run structLMM and alternative linear
mixed models implementations in Python.

.. testcode::

    import os
    import numpy as np
    import pandas as pd
    import scipy as sp
    from limix_core.util.preprocess import gaussianize
    from limix_core.gp import GP2KronSumLR
    from limix_core.covar import FreeFormCov
    from limix_lmm import LMM
    from limix_lmm import download, unzip
    from pandas_plink import read_plink
    from sklearn.impute import SimpleImputer
    import geno_sugar as gs
    import geno_sugar.preprocess as prep

    # Download and unzip the dataset files
    download("http://www.ebi.ac.uk/~casale/data_structlmm.zip")
    unzip("data_structlmm.zip")

    # import genotype file
    bedfile = "data_structlmm/chrom22_subsample20_maf0.10"
    (bim, fam, G) = read_plink(bedfile, verbose=False)

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
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    preprocess = prep.compose(
        [
            prep.filter_by_missing(max_miss=0.10),
            prep.impute(imputer),
            prep.filter_by_maf(min_maf=0.10),
            prep.standardize(),
        ]
    )

    # loop on geno
    res = []
    queue = gs.GenoQueue(G, bim, batch_size=200, preprocess=preprocess)
    for _G, _bim in queue:
        lmm.process(_G)
        pv = lmm.getPv()
        beta = lmm.getBetaSNP()
        _bim = _bim.assign(lmm_pv=pd.Series(pv, index=_bim.index))
        _bim = _bim.assign(lmm_beta=pd.Series(beta, index=_bim.index))
        res.append(_bim)

    res = pd.concat(res)
    res.reset_index(inplace=True, drop=True)

    # export
    print("Exporting to out/")
    if not os.path.exists("out"):
        os.makedirs("out")
    res.to_csv("out/res_lmm.csv", index=False)

.. testoutput::

    .. read 200 / 994 variants (20.12%)
    .. read 400 / 994 variants (40.24%)
    .. read 600 / 994 variants (60.36%)
    .. read 800 / 994 variants (80.48%)
    .. read 994 / 994 variants (100.00%)
    Exporting to out/



Multi-trait Linear Mixed Model (MTLMM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. testcode::

    import scipy as sp
    import scipy.linalg as la
    from limix_core.gp import GP2KronSum
    from limix_core.covar import FreeFormCov

    def generate_data(N, P, K, S):
        import scipy as sp

        # fixed eff
        F = sp.randn(N, K)
        B0 = 2 * (sp.arange(0, K * P) % 2) - 1.0
        B0 = sp.reshape(B0, (K, P))
        FB = sp.dot(F, B0)

        # gerenrate phenos
        h2 = sp.linspace(0.1, 0.5, P)

        # generate data
        G = 1.0 * (sp.rand(N, S) < 0.2)
        G -= G.mean(0)
        G /= G.std(0)
        G /= sp.sqrt(G.shape[1])
        Wg = sp.randn(P, P)
        Wn = sp.randn(P, P)

        B = sp.randn(G.shape[1], Wg.shape[1])
        Yg = G.dot(B).dot(Wg.T)
        Yg -= Yg.mean(0)
        Yg *= sp.sqrt(h2 / Yg.var(0))
        Cg0 = sp.cov(Yg.T)

        B = sp.randn(G.shape[0], Wg.shape[1])
        Yn = B.dot(Wn.T)
        Yn -= Yn.mean(0)
        Yn *= sp.sqrt((1 - h2) / Yn.var(0))
        Cn0 = sp.cov(Yn.T)

        Y = FB + Yg + Yn

        return Y, F, G, B0, Cg0, Cn0

    N = 1000
    P = 4
    K = 2
    S = 500
    Y, F, G, B0, Cg0, Cn0 = generate_data(N, P, K, S)

    # compute eigenvalue decomp of RRM
    R = sp.dot(G, G.T)
    R /= R.diagonal().mean()
    R += 1e-4 * sp.eye(R.shape[0])
    Sr, Ur = la.eigh(R)

    # fit null model
    Cg = FreeFormCov(Y.shape[1])
    Cn = FreeFormCov(Y.shape[1])
    gp = GP2KronSum(Y=Y, S_R=Sr, U_R=Ur, Cg=Cg, Cn=Cn, F=F, A=sp.eye(P))
    gp.covar.Cg.setCovariance(0.5 * sp.cov(Y.T))
    gp.covar.Cn.setCovariance(0.5 * sp.cov(Y.T))
    gp.optimize(factr=10)

    # run MTLMM
    from limix_lmm import MTLMM
    mtlmm = MTLMM(Y, F=F, A=sp.eye(P), Asnp=sp.eye(P), covar=gp.covar)
    mtlmm.process(G)
    pv = mtlmm.getPv()
    B = mtlmm.getBetaSNP()
    print(pv)
    print(B)

.. testoutput::

    asd

A full description of all methods can be found in :ref:`public`.

