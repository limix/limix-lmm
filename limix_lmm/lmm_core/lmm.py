from .lmm_core import LMMCore
from time import time


class LMM(LMMCore):
    r"""
    Standard LMM with general bg covariance

    The LMM model is

    .. math::
        \mathbf{y}=\sim\mathcal{N}(
        \underbrace{\mathbf{F}\mathbf{b}}_{\text{covariates}}+
        \underbrace{\mathbf{x}\beta}_{\text{genetics}},
        \underbrace{\mathbf{K}_{\boldsymbol{\theta}}}_{\text{covariance}})

    The test :math:`\beta\neq{0}` is done for all provided variants
    one-by-one.

    Parameters
    ----------
    y : (`N`, 1) ndarray
        phenotype vector
    F : (`N`, L) ndarray
        fixed effect design for covariates.
    Ki_dot : function
        method that takes an array and returns the dot product of
        the inverse of the covariance and the input array.

    Examples
    --------

    .. doctest::

        >>> from numpy.random import RandomState
        >>> import scipy as sp
        >>> from limix_lmm import LMM
        >>> from limix_core.gp import GP2KronSumLR
        >>> from limix_core.covar import FreeFormCov
        >>> random = RandomState(1)
        >>> from numpy import set_printoptions
        >>> set_printoptions(4)
        >>>
        >>> # generate data
        >>> N = 100
        >>> k = 1
        >>> S = 1000
        >>> y = random.randn(N, 1)
        >>> E = random.randn(N, k)
        >>> G = 1.*(random.rand(N, S) < 0.2)
        >>> F = random.rand(N, 1)
        >>> F = sp.concatenate([sp.ones((N, 1)), F], 1)
        >>>
        >>> # larn a covariance on the null model
        >>> gp = GP2KronSumLR(Y=y, Cn=FreeFormCov(1), G=E, F=F, A=sp.ones((1,1)))
        >>> gp.covar.Cr.setCovariance(0.5*sp.ones((1,1)))
        >>> gp.covar.Cn.setCovariance(0.5*sp.ones((1,1)))
        >>> info_opt = gp.optimize(verbose=False)
        >>>
        >>> lmm = LMM(y, F, gp.covar.solve)
        >>> lmm.process(G)
        >>> pv = lmm.getPv()
        >>> beta = lmm.getBetaSNP()
        >>> beta_ste = lmm.getBetaSNPste()
        >>> lrt = lmm.getLRT()
        >>>
        >>> print(pv[:4])
        [0.8335 0.1669 0.9179 0.279 ]
        >>> print(beta[:4])
        [-0.0479  0.3145  0.0235 -0.2465]
        >>> print(beta_ste[:4])
        [0.2279 0.2275 0.2283 0.2276]
        >>> print(lrt[:4])
        [0.0442 1.9108 0.0106 1.1721]
    """

    def __init__(self, y, F, Ki_dot=None):
        import scipy as sp

        if F is None:
            F = sp.ones((y.shape[0], 1))
        self.y = y
        self.F = F
        self.Ki_dot = Ki_dot
        self.df = y.shape[0] - F.shape[1]
        self._fit_null()

    def _fit_null(self):
        """ Internal functon. Fits the null model """
        import scipy as sp
        import scipy.linalg as la

        if self.Ki_dot is None:
            self.Kiy = self.y
            self.KiF = self.F
        else:
            self.Kiy = self.Ki_dot(self.y)
            self.KiF = self.Ki_dot(self.F)
        self.FKiy = sp.dot(self.F.T, self.Kiy)
        self.FKiF = sp.dot(self.F.T, self.KiF)
        self.yKiy = sp.dot(self.y[:, 0], self.Kiy[:, 0])
        # calc beta_F0 and s20
        self.A0i = la.inv(self.FKiF)
        self.beta_F0 = sp.dot(self.A0i, self.FKiy)
        self.s20 = (self.yKiy - sp.dot(self.FKiy[:, 0], self.beta_F0[:, 0])) / self.df

    def process(self, G, verbose=False):
        r"""
        Fit genotypes one-by-one.

        Parameters
        ----------
        G : (`N`, `S`) ndarray
            genotype vector for `N` individuals and `S` variants.
        verbose : bool
            verbose flag.

        Returns
        -------
        pv : ndarray
            P values
        beta : ndarray
            variant effect szies
        """
        import scipy as sp
        import scipy.stats as st

        t0 = time()
        # precompute some stuff
        if self.Ki_dot is None:
            KiG = G
        else:
            KiG = self.Ki_dot(G)
        GKiy = sp.dot(G.T, self.Kiy[:, 0])
        GKiG = sp.einsum("ij,ij->j", G, KiG)
        FKiG = sp.dot(self.F.T, KiG)

        # Let us denote the inverse of Areml as
        # Ainv = [[A0i + m mt / n, m], [mT, n]]
        A0iFKiG = sp.dot(self.A0i, FKiG)
        n = 1.0 / (GKiG - sp.einsum("ij,ij->j", FKiG, A0iFKiG))
        M = -n * A0iFKiG
        self.beta_F = self.beta_F0 + M * sp.dot(M.T, self.FKiy[:, 0]) / n
        self.beta_F += M * GKiy
        self.beta_g = sp.einsum("is,i->s", M, self.FKiy[:, 0])
        self.beta_g += n * GKiy

        # sigma
        s2 = self.yKiy - sp.einsum("i,is->s", self.FKiy[:, 0], self.beta_F)
        s2 -= GKiy * self.beta_g
        s2 /= self.df

        # dlml and pvs
        self.lrt = -self.df * sp.log(s2 / self.s20)
        self.pv = st.chi2(1).sf(self.lrt)

        t1 = time()
        if verbose:
            print("Tested for %d variants in %.2f s" % (G.shape[1], t1 - t0))

        return self.pv, self.beta_g
