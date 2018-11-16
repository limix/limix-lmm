import scipy as sp

from .util import christof_trick


def define_helper(Y, F, A, Asnp, covar=None):
    import limix_core

    if covar is None:
        return MTLMMHelperBase(Y, F, A, Asnp)
    else:
        # at the moment only Cov2KronSum is supported but
        # more covariances can be added within the same framework
        covtypes = [limix_core.covar.Cov2KronSum]
        assert type(covar) in covtypes, "This covariance type is not supported"
        if type(covar == limix_core.covar.Cov2KronSum):
            return MTLMMHelper2KronSum(Y, F, A, Asnp, covar)


class MTLMMHelperBase:
    r"""
    Internal helper class for MTLMM
    """

    def __init__(self, Y, F, A, Asnp):
        self._Y = Y
        self._F = F
        self._A = A
        self._Asnp = Asnp
        self._YAsnp = sp.dot(self._Y, self._Asnp)

    def set_g(self, _g):
        self._g = _g

    def get_WKiy(self):
        RV = sp.dot(self._F.T, self._Y.dot(self._A))
        RV = sp.reshape(RV, [RV.size, 1], order="F")
        return RV

    def get_yKiy(self):
        return sp.einsum("ip,ip->", self.LY, self._Y)

    def get_WKiW(self):
        return christof_trick(self._A, self._F, sp.ones(self.Y.size))

    def get_W1Kiy_2(self):
        return sp.dot(self._YAsnp.T, self._g)

    def get_W1KiW1_12(self):
        return christof_trick(
            self._A, self._F, sp.ones(self.Y.size), A2=self._Asnp, F2=self._g
        )

    def get_W1KiW1_22(self):
        return christof_trick(self._Asnp, self._g, sp.ones(self.Y.size))


class MTLMMHelper2KronSum(MTLMMHelperBase):
    r"""
    Internal helper class for MTLMM
    """

    def __init__(self, Y, F, A, Asnp, covar):
        self.Y = Y
        self.F = F
        self.A = A
        self.Asnp = Asnp
        self.covar = covar

        # compute stuff from covariance
        self.LY = covar.Lr().dot(self.Y.dot(covar.Lc().T))
        self._Y = covar.D() * self.LY
        self._A = covar.Lc().dot(self.A)
        self._Asnp = covar.Lc().dot(self.Asnp)
        self._F = covar.Lr().dot(self.F)
        self._D = self.covar.D()
        self._YAsnp = sp.dot(self._Y, self._Asnp)

    def set_g(self, g):
        self._g = self.covar.Lr().dot(g)

    def get_WKiy(self):
        RV = sp.dot(self._F.T, self._Y.dot(self._A))
        RV = sp.reshape(RV, [RV.size, 1], order="F")
        return RV

    def get_yKiy(self):
        return sp.einsum("ip,ip->", self.LY, self._Y)

    def get_WKiW(self):
        return christof_trick(self._A, self._F, self.covar.D())

    def get_W1Kiy_2(self):
        return sp.dot(self._YAsnp.T, self._g)

    def get_W1KiW1_12(self):
        return christof_trick(
            self._A, self._F, self.covar.D(), A2=self._Asnp, F2=self._g
        )

    def get_W1KiW1_22(self):
        return christof_trick(self._Asnp, self._g, self.covar.D())
