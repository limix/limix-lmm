import scipy as sp
import scipy.linalg as la


def calc_Ai_beta_s2(yKiy, FKiF, FKiy, df):
    import scipy as sp
    import scipy.linalg as la

    Ai = la.pinv(FKiF)
    beta = sp.dot(Ai, FKiy)
    s2 = (yKiy - sp.dot(FKiy[:, 0], beta[:, 0])) / df
    return Ai, beta, s2


def hatodot(A, B):
    """ should be implemented in C """
    import scipy as sp

    A1 = sp.kron(A, sp.ones((1, B.shape[1])))
    B1 = sp.kron(sp.ones((1, A.shape[1])), B)
    return A1 * B1


def christof_trick(A1, F1, D, A2=None, F2=None):
    if A2 is None:  A2 = A1
    if F2 is None:  F2 = F1
    out = sp.zeros([A1.shape[1] * F1.shape[1], A2.shape[1] * F2.shape[1]])
    for c in range(A1.shape[0]):
        Oc = sp.dot(A1[[c],:].T, A2[[c],:])
        Or = sp.dot(F1.T, D[:,[c]] * F2)
        out += sp.kron(Oc, Or)
    return out
