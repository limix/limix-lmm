from .lmm_core import LMMCore


class MTLMM(LMMCore):
    r"""
    Multi-trait LMM for single-variant association testing.

    The model is

    .. math::
        \text{vec}\left(\mathbf{Y}\right)\sim\mathcal{N}(
        \underbrace{(\mathbf{A}\otimes\mathbf{F})\text{vec}\left(\mathbf{B}\right)}_{\text{covariates}}+
        \underbrace{(\mathbf{A}\otimes\mathbf{g})\mathbf{b}}_{\text{genetics}}+
        \sigma^2\underbrace{\mathbf{K}_{
        \boldsymbol{\theta}_0}}_{\text{covariance}})

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

    def __init__(self, Y, F=None, A=None, Asnp=None, Ki_dot=None):
        import scipy as sp

        if F is None:
            F = sp.ones((Y.shape[0], 1))
        if A is None:
            A = sp.eye(Y.shape[1])
        if Asnp is None:
            Asnp = sp.eye(Y.shape[1])

        # store useful stuff
        self.Asnp = Asnp

        # convert to a univariate problem
        y = sp.reshape(Y, (Y.size, 1), order="F")
        W = sp.kron(A, F)
        super(MTLMM, self).__init__(y, W, Ki_dot=Ki_dot)
        self._fit_null()

    def process(self, G, verbose=False):
        r"""
        Fit genotypes one-by-one.

        Parameters
        ----------
        G : (`N`, `S`) ndarray
        Inter : (`N`, `M`) ndarray
            Matrix of `M` factors for `N` inds with which
            each variant interact
            By default, Inter is set to a matrix of ones.
        step : int
            Number of consecutive variants that should be
            tested jointly.
        verbose : bool
            verbose flag.
        """
        import scipy as sp

        Aext = sp.kron(self.Asnp, sp.ones((G.shape[0], 1)))
        Gext = sp.kron(sp.ones((self.Asnp.shape[0], 1)), G)
        Wext = sp.einsum("ip,in->inp", Aext, Gext).reshape(Aext.shape[0], -1)
        return super(MTLMM, self).process(
            Wext, step=self.Asnp.shape[0], verbose=verbose
        )
