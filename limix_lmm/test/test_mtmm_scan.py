from numpy.testing import assert_allclose


def _generate_data(N, P, K, S):
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


def test_mtmm_scan_pv_beta():
    import scipy as sp
    import scipy.linalg as la
    from limix_core.gp import GP2KronSum
    from limix_core.covar import FreeFormCov

    N = 200
    P = 4
    M = 2
    K = 2
    S = 10
    Y, F, G, B0, Cg0, Cn0 = _generate_data(N, P, K, S)
    A = sp.eye(P)
    Asnp = sp.rand(P, M)

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
    from limix_lmm.lmm_core import MTLMM

    mtlmm = MTLMM(Y, F=F, A=A, Asnp=Asnp, covar=gp.covar)
    pv, B = mtlmm.process(G)

    # run standard LMMcore
    from limix_lmm.lmm_core import LMMCore

    y = sp.reshape(Y, [Y.size, 1], order="F")
    covs = sp.kron(A, F)
    Aext = sp.kron(Asnp, sp.ones((G.shape[0], 1)))
    Gext = sp.kron(sp.ones((Asnp.shape[0], 1)), G)
    Wext = sp.einsum("ip,in->inp", Aext, Gext).reshape(Aext.shape[0], -1)
    stlmm = LMMCore(y, covs, Ki_dot=gp.covar.solve)
    stlmm.process(Wext, step=Asnp.shape[1])
    pv0 = stlmm.getPv()
    B0 = stlmm.getBetaSNP()

    assert_allclose(pv0, pv, atol=1e-9)
    assert_allclose(B0, B, atol=1e-9)
