import os
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg as la
from limix_core.util.preprocess import gaussianize
from limix_lmm import GWAS_LMM, GWAS_MTLMM
from limix_lmm import download, unzip
from pandas_plink import read_plink
from sklearn.impute import SimpleImputer
import geno_sugar as gs
import geno_sugar.preprocess as prep


if __name__ == "__main__":

    # download data
    download("http://www.ebi.ac.uk/~casale/data_structlmm.zip")
    unzip("data_structlmm.zip")

    # import snp data
    bedfile = "data_structlmm/chrom22_subsample20_maf0.10"
    (bim, fam, G) = read_plink(bedfile, verbose=False)

    # consider the first 100 snps
    snps = G[:100].compute().T

    # define genetic relatedness matrix
    W_R = sp.randn(fam.shape[0], 20)
    R = sp.dot(W_R, W_R.T)
    R /= R.diagonal().mean()
    S_R, U_R = la.eigh(R)

    # load phenotype data
    phenofile = "data_structlmm/expr.csv"
    dfp = pd.read_csv(phenofile, index_col=0)
    pheno = gaussianize(dfp.loc["gene1"].values[:, None])

    # define covs
    covs = sp.ones([pheno.shape[0], 1])

    # linear model test
    lm = GWAS_LMM(pheno, covs=covs, verbose=True)
    res = lm.process(snps)
    print(res.head())

    # linear mixed model
    lmm = GWAS_LMM(pheno, covs=covs, eigh_R=(S_R, U_R), verbose=True)
    res = lmm.process(snps)
    print(res.head())

    # low-rank linear mixed model (low-rank)
    lrlmm = GWAS_LMM(pheno, covs=covs, W_R=W_R, verbose=True)
    res = lrlmm.process(snps)
    print(res.head())

    # generate interacting variables to test
    inter = sp.randn(pheno.shape[0], 1)

    # interaction test
    lmi = GWAS_LMM(pheno, covs=covs, inter=inter, verbose=True)
    res = lmi.process(snps)
    print(res.head())

    # generate interacting variables to test
    inter0 = sp.randn(pheno.shape[0], 1)

    # interaction test
    lmi = GWAS_LMM(pheno, covs=covs, inter=inter, inter0=inter0, verbose=True)
    res = lmi.process(snps)
    print(res.head())

    # test multi-trait association
    P = 4
    phenos = sp.randn(pheno.shape[0], P)
    Asnps = sp.eye(P)
    mtlmm = GWAS_MTLMM(phenos, covs=covs, Asnps=Asnps, eigh_R=(S_R, U_R), verbose=True)
    res = mtlmm.process(snps)
    print(res.head())

    # common effect test
    Asnps0 = sp.ones([P, 1])
    mtlmm = GWAS_MTLMM(
        phenos, covs=covs, Asnps=Asnps, Asnps0=Asnps0, eigh_R=(S_R, U_R), verbose=True
    )
    res = mtlmm.process(snps)
    print(res.head())

    # slice of genome to analyze
    Isnp = gs.is_in(bim, ("22", 17500000, 18000000))
    G, bim = gs.snp_query(G, bim, Isnp)

    # define geno preprocessing function for geno-wide analysis
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    preprocess = prep.compose(
        [
            prep.filter_by_missing(max_miss=0.10),
            prep.impute(imputer),
            prep.filter_by_maf(min_maf=0.10),
            prep.standardize(),
        ]
    )

    from limix_lmm.util import append_res

    # slide large genetic region using batches of 200 variants
    res = []
    queue = gs.GenoQueue(G, bim, batch_size=200, preprocess=preprocess)
    for _G, _bim in queue:

        _res = {}
        _res["lm"] = lm.process(_G)
        _res["lmm"] = lmm.process(_G)
        _res["lrlmm"] = lrlmm.process(_G)
        _res = append_res(_bim, _res)
        _res.append(_res)

    # export
    print("Exporting to out/")
    if not os.path.exists("out"):
        os.makedirs("out")
    res = pd.concat(res)
    res.reset_index(inplace=True, drop=True)
    res.to_csv("out/res_lmm.csv", index=False)
