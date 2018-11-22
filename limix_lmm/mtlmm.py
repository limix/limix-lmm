from time import time

import scipy as sp
import scipy.stats as st

from .lmm_core import LMMCore
from .util import calc_Ai_beta_s2
from .mtlmm_helper import define_helper


class MTLMM(LMMCore):
    r"""
    Multi-trait LMM for single-variant association testing.

    The model is

    .. math::
        \text{vec}\left(\mathbf{Y}\right)\sim\mathcal{N}(
        \underbrace{(\mathbf{A}\otimes\mathbf{F})\text{vec}\left(\mathbf{B}\right)}_{\text{covariates}}+
        \underbrace{(\mathbf{A}_{\text{snp}\otimes\mathbf{g})\mathbf{b}}_{\text{genetics}}+
        \sigma^2\underbrace{
        \mathbf{C}^{\text{(g)}}_0\otimes\mathbf{R} +
        \mathbf{C}^{\text{(n)}}_0\otimes\mathbf{I}
        }_{\text{covariance}})

    where :math:`\otimes` is Kronecker product.

    Importantly, :math:`\mathbf{K}_{\boldsymbol{\theta}_0}`
    is not re-estimated under the laternative model,
    i.e. only :math:`\sigma^2` is learnt.
    The covariance parameters :math:`\boldsymbol{\theta}_0`
    need to be learnt using another module, e.g. limix-core.
    Note that as a consequence of this,
    such an implementation of single-variant analysis does not require the
    specification of the whole covariance but just of a method that
    specifies the product of its inverse by a vector.

    The test :math:`\boldsymbol{\beta}\neq{0}` is done
    in bocks of ``step`` variants,
    where ``step`` can be specifed by the user.

    Parameters
    ----------
    Y : (`N`, `P`) ndarray
        phenotype matrix
    F : (`N`, `L`) ndarray
        row fixed-effect design for covariates.
        (default is an intercept)
    A : (`P`, `W`) ndarray
        col fixed-effect design for covariates (default is eye)
    Asnp : (`P`, `Wg`) ndarray
        col fixed-effect design for genetic variants (default is eye)
    Ki_dot : function
        method that takes an array and returns the dot product of
        the inverse of the covariance and the input array.
    """

    def __init__(self, Y, F=None, A=None, Asnp=None, covar=None):

        if F is None:
            F = sp.ones((Y.shape[0], 1))
        if A is None:
            A = sp.eye(Y.shape[1])
        if Asnp is None:
            Asnp = sp.eye(Y.shape[1])

        # the helper implements methods that are specific to the covariance
        self.helper = define_helper(Y, F, A, Asnp, covar)

        # store useful stuff
        self.Y = Y
        self.F = F
        self.A = A
        self.Asnp = Asnp
        self.df = Y.size - A.shape[1] * F.shape[1]

        # fit null
        self._fit_null()

    def _fit_null(self):
        """ Internal functon. Fits the null model """
        self.WKiy = self.helper.get_WKiy()
        self.yKiy = self.helper.get_yKiy()
        self.WKiW = self.helper.get_WKiW()
        # calc beta_F0 and s20
        self.A0i, self.beta_F0, self.s20 = calc_Ai_beta_s2(
            self.yKiy, self.WKiW, self.WKiy, self.df
        )

    def process(self, G, verbose=False):
        r"""
        Fit genotypes one-by-one.

        Parameters
        ----------
        G : (`N`, `S`) ndarray
        verbose : bool
            verbose flag.
        """
        t0 = time()
        k = self.A.shape[1] * self.F.shape[1]
        m = self.Asnp.shape[1]
        W1KiW1 = sp.zeros((k + m, k + m))
        W1KiW1[:k, :k] = self.WKiW
        W1Kiy = sp.zeros((k + m, 1))
        W1Kiy[:k, 0] = self.WKiy[:, 0]
        s2 = sp.zeros(G.shape[1])
        self.beta_g = sp.zeros([m, G.shape[1]])
        for s in range(G.shape[1]):
            self.helper.set_g(G[:, [s]])
            W1Kiy[k:, 0] = self.helper.get_W1Kiy_2()[:, 0]
            W1KiW1[:k, k:] = self.helper.get_W1KiW1_12()
            W1KiW1[k:, :k] = W1KiW1[:k, k:].T
            W1KiW1[k:, k:] = self.helper.get_W1KiW1_22()
            # the following can be sped up by using block matrix inversion, etc
            _, beta, s2[s] = calc_Ai_beta_s2(self.yKiy, W1KiW1, W1Kiy, self.df)
            self.beta_g[:, s] = beta[k:, 0]
        # dlml and pvs
        self.lrt = -self.df * sp.log(s2 / self.s20)
        self.pv = st.chi2(m).sf(self.lrt)

        t1 = time()
        if verbose:
            print("Tested for %d variants in %.2f s" % (G.shape[1], t1 - t0))

        return self.pv, self.beta_g
