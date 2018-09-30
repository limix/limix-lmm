import scipy as sp
import scipy.linalg as la
from limix_core.gp import GP2KronSum
from limix_core.covar import FreeFormCov
from limix_lmm.lmm_core import MTLMM


def generate_data(N, P, K, S):

    # fixed eff
    F = sp.randn(N, K)
    B0 = 2 * (sp.arange(0, K*P) % 2) - 1.
    B0 = sp.reshape(B0, (K, P))
    FB = sp.dot(F, B0)

    # gerenrate phenos
    h2 = sp.linspace(0.1, 0.5, P)

    # generate data
    G = 1. * (sp.rand(N, S)<0.2)
    G-= G.mean(0); G/= G.std(0); G/= sp.sqrt(G.shape[1])
    Wg = sp.randn(P, P)
    Wn = sp.randn(P, P)

    B = sp.randn(G.shape[1], Wg.shape[1])
    Yg = G.dot(B).dot(Wg.T)
    Yg-= Yg.mean(0)
    Yg*= sp.sqrt(h2 / Yg.var(0))
    Cg0 = sp.cov(Yg.T)

    B = sp.randn(G.shape[0], Wg.shape[1])
    Yn = B.dot(Wn.T)
    Yn-= Yn.mean(0)
    Yn*= sp.sqrt((1-h2) / Yn.var(0))
    Cn0 = sp.cov(Yn.T)

    Y = FB + Yg + Yn

    return Y, F, G, B0, Cg0, Cn0


if __name__=='__main__':

    N = 1000
    P = 4
    K = 2
    S = 500
    Y, F, G, B0, Cg0, Cn0 = generate_data(N, P, K, S)

    # compute eigenvalue decomp of RRM
    R = sp.dot(G, G.T)
    R/= R.diagonal().mean()
    R+= 1e-4 * sp.eye(R.shape[0])
    Sr, Ur = la.eigh(R)

    # fit null model
    Cg = FreeFormCov(Y.shape[1])
    Cn = FreeFormCov(Y.shape[1])
    gp = GP2KronSum(Y=Y, S_R=Sr, U_R=Ur, Cg=Cg, Cn=Cn, F=F, A=sp.eye(P))
    gp.covar.Cg.setCovariance(0.5 * sp.cov(Y.T))
    gp.covar.Cn.setCovariance(0.5 * sp.cov(Y.T))
    gp.optimize(factr=10)

    # run MTLMM
    mtlmm = MTLMM(Y, F=F, Ki_dot=gp.covar.solve)
    pv, B = mtlmm.process(G)
