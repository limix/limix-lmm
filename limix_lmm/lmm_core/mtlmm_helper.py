import scipy as sp


def christof_trick(A1, F1, D, A2=None, F2=None):
    if A2 is None:  A2 = A1
    if F2 is None:  F2 = F1
    out = sp.zeros([A1.shape[1] * F1.shape[1], A2.shape[1] * F2.shape[1]])
    for c in range(A1.shape[0]):
        Oc = sp.dot(A1[[c],:].T, A2[[c],:])
        Or = sp.dot(F1.T, D[:,[c]] * F2)
        out += sp.kron(Oc, Or)
    return out


class MTLMMHelper():
    r"""
    Internal helper class for MTLMM
    """

    def __init__(self, Y, F, A, Asnp, covar=None):
        self.Y = Y
        self.F = F
        self.A = A
        self.Asnp = Asnp
        self.covar = covar

        # compute stuff from covariance
        if covar is not None:
            self.LY = covar.Lr().dot(self.Y.dot(covar.Lc().T))
            self._Y = covar.D() * self.LY
            self._A = covar.Lc().dot(self.A)
            self._Asnp = covar.Lc().dot(self.Asnp)
            self._F = covar.Lr().dot(self.F)
            self._D = self.covar.D()
        else:
            self.LY = self.Y
            self._Y = self.Y
            self._A = self.A
            self._Asnp = self.Asnp
            self._F = self.F
            self._D = sp.ones(self.Y.shape)

        # useful to cache this for the alternative model
        self._YAsnp = sp.dot(self._Y, self._Asnp)

    def set_g(self, _g):
        if self.covar is not None:
            self._g = self.covar.Lr().dot(_g)
        else:
            self._g = _g

    def get_WKiy(self):
        RV = sp.dot(self._F.T, self._Y.dot(self._A))
        RV = sp.reshape(RV, [RV.size, 1], order='F')
        return RV

    def get_yKiy(self):
        return sp.einsum('ip,ip->', self.LY, self._Y)

    def get_WKiW(self):
        return christof_trick(self._A, self._F, self.covar.D())

    def get_W1Kiy_2(self):
        return sp.dot(self._YAsnp.T, self._g)

    def get_W1KiW1_12(self):
        return christof_trick(self._A,
                              self._F,
                              self.covar.D(),
                              A2 = self._Asnp,
                              F2 = self._g)

    def get_W1KiW1_22(self):
        return christof_trick(self._Asnp, self._g, self.covar.D())
