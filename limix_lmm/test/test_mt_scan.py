from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy import dot, eye, add, kron, concatenate, array

from numpy_sugar.linalg import rsolve, economic_qs

from limix_lmm._blk_diag import BlockDiag

import numpy as np

np.set_printoptions(precision=20)


def fit_beta(Y, A, M, C0, C1, QS, G):
    n, p = Y.shape
    betas = []
    for i in range(G.shape[1]):
        MG = concatenate([M, G[:, [i]]], axis=1)

        Di = [D.inv() for D in D(C0, C1, QS)]

        AQtM = [kron(A, dot(Q.T, MG)) for Q in QS[0]]
        DiAQtM = [Di.dot(AQtM) for Di, AQtM in zip(Di, AQtM)]
        QtY = [dot(Q.T, Y) for Q in QS[0] if Q.size > 0]
        DiQtY = [Di.dot_vec(QtY) for Di, QtY in zip(Di, QtY)]

        nominator = []

        denominator = [dot(i.T, j) for i, j in zip(DiAQtM, AQtM)]
        nominator = [dot(i.T, j) for i, j in zip(AQtM, DiQtY)]

        denominator = add.reduce(denominator)
        nominator = add.reduce(nominator)

        beta = rsolve(denominator, nominator)
        beta.reshape((MG.shape[1], p), order="F")
        betas.append(beta)
    return betas


def D(C0, C1, QS):
    C0 = C0
    C1 = C1
    S0 = QS[1]

    p = C0.shape[0]
    n = QS[0][0].shape[0]
    r = QS[0][0].shape[1]

    siz = [s for s in [r, n - r] if s > 0]
    D = [BlockDiag(p, s) for s in siz]

    for i in range(C0.shape[0]):
        for j in range(C0.shape[1]):
            D[0].set_block(i, j, C0[i, j] * S0 + C1[i, j])
            if len(D) > 1:
                D[1].set_block(i, j, full(n - r, C1[i, j]))

    return D


def test_mt_scan():

    random = RandomState(0)
    # samples
    n = 5
    # traits
    p = 2
    # covariates
    d = 3

    Y = random.randn(n, p)
    A = random.randn(p, p)
    A = dot(A, A.T)
    M = random.randn(n, d)
    K = random.randn(n, n)
    K = (K - K.mean(0)) / K.std(0)
    K = K.dot(K.T) + eye(n) + 1e-3
    QS = economic_qs(K)

    C0 = random.randn(p, p)
    C0 = dot(C0, C0.T)
    C1 = random.randn(p, p)
    C1 = dot(C1, C1.T)
    G = random.randn(n, 3)

    betas = fit_beta(Y, A, M, C0, C1, QS, G)

    assert_allclose(
        betas,
        [
            array(
                [
                    [0.1171543072226424],
                    [0.2922669722595269],
                    [-0.02153087832329973],
                    [-0.6785191889622902],
                    [1.2163628766377277],
                    [-0.1328747439139128],
                    [-0.7187298358085206],
                    [-1.3501558521634132],
                ]
            ),
            array(
                [
                    [-0.38239934605314946],
                    [0.24597204056173463],
                    [0.010946258320120424],
                    [-0.04119008869431426],
                    [0.1474223136659856],
                    [-0.3345533712484771],
                    [-1.4415249194182163],
                    [-1.490028121254687],
                ]
            ),
            array(
                [
                    [0.22472471155023616],
                    [0.7345724052293824],
                    [0.18207580059536876],
                    [0.5916437252056872],
                    [1.2864372666081683],
                    [0.5670883175815873],
                    [-0.3512789451485551],
                    [0.9050459221116203],
                ]
            ),
        ],
    )
