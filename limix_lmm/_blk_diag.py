from __future__ import division

from numpy import zeros, concatenate, add, kron, stack, sum, log, block, asarray, dot

from numpy_sugar.linalg import ddot


class BlockDiag(object):
    r""" Implements a special kind of block diagonal matrix.

    Implements::

        D = |      ...     |
            | ... D_ij ... |
            |      ...     |

    for which::

        D_ij = Diag([d_ij1  d_ij2  …  d_ijn]) ∈ ℝ^{n×n}

             and i, j ∈ {1, 2, …, n}

    Parameters
    ----------
    d : int
        Number of blocks.
    n : int
        Block size.
    data : array_like, optional
        Data from another block diagonal. Defaults to ``None``.

    Notes
    -----
    The data is compactly store in an n×(nd) array, each row containing the diagonals
    of the D_{i:} blocks::

        data_{i,j*n:j*n+n} = [d_ij1  d_ij2  …  d_ijn].
    """

    def __init__(self, d, n, data=None):
        self._d = d
        self._n = n
        if data is None:
            self._data = zeros((d, n * d))
        else:
            self._data = data

    def matrix(self):
        n = self._n
        d = self._d
        D = zeros((d * n, d * n))
        for i in range(d):
            row_offset = i * n
            for j in range(d):
                col_offset = j * n
                for k in range(n):
                    D[row_offset + k, col_offset + k] = self._data[i, col_offset + k]
        return D

    def set_block(self, i, j, v):
        r""" Set the block ``(i, j)``.

        Set D_{i,j} with a given vector.

        Parameters
        ----------
        i : int
            Row ``i``.
        j : int
            Row ``j``.
        v : array_like
            Vector of size ``n``.
        """
        self._data[i, j * self._n : j * self._n + self._n] = asarray(v, float)

    def get_block(self, i, j):
        r""" Get the block ``(i, j)``.

        Get D_{i,j}.

        Returns
        -------
        :class:`numpy.darray`
            Diagonal of D_{i,j}.
        """
        return _get_block(self._data, i, j)

    def inv(self):
        r""" Matrix inverse.

        Returns
        -------
        :class:`BlockDiag`
            Inverse of this matrix.
        """
        return BlockDiag(self._d, self._n, _rinv(self._data))

    @property
    def T(self):
        r""" Transpose. """
        d = self._d
        n = self._n
        Dt = BlockDiag(d, n)
        for i in range(d):
            for j in range(d):
                Dt._data[j, i * n : i * n + n] = self._data[i, j * n : j * n + n]
        return Dt

    def logdet(self):
        r""" Log of the matrix determinant. """
        return _logdet(self._data, self._n)

    def dot(self, B):
        r""" Product of this with another matrix. """
        return _bd_dot_fd(self._data, B)

    def dotd(self, B):
        r""" Product of this with another matrix. """
        return _bd_dotd(self, B)

    def dot_vec(self, B):
        return self.dot(_vec(B))

    def concat(self, B):
        r""" Concatenate two block diagonals on the column axis.

        This is equivalent to

        .. python::

            np.concatenate([self.matrix(), B.matrix()], axis=1)

        and then transforming the result into :class:`BlockDiag`.
        """
        A = self
        assert A._d == B._d
        C = BlockDiag(A._d, A._n + B._n)
        for i in range(C._d):
            for j in range(C._d):
                v = concatenate([A.get_block(i, j), B.get_block(i, j)])
                C.set_block(i, j, v)
        return C


def _bd_dot(A, D):
    if isinstance(D, BlockDiag):
        return _bd_dot_fbd(A, D)
    return D.T.dot(A.T).T


def _bd_dotd(A, B):
    arr = []
    for i in range(A._d):
        acc = []
        for j in range(B._d):
            acc.append(A.get_block(i, j) * B.get_block(j, i))
        arr.append(add.reduce(acc))
    return concatenate(arr)


def _bd_kron_dot(A, B, D):
    rows = []
    for i in range(D._d):
        AiB = kron(A[i, :], B)
        rows.append(_bd_dot(AiB, D))
    return concatenate(rows, axis=0)


def _bd_dot_fbd(A, D):
    return D.T.dot(A.T).T


def _bd_dot_fd(D, A):
    assert not isinstance(D, BlockDiag)
    d = D.shape[0]
    rows = []
    for i in range(d):
        if A.ndim == 1:
            rows.append(_bd_dot_fv(D, i, A))
        else:
            cols = []
            for j in range(A.shape[1]):
                cols.append(_bd_dot_fv(D, i, A[:, j]))
            rows.append(stack(cols, axis=1))
    return concatenate(rows, axis=0)


def _bd_dot_fv(D, i, a):
    assert not isinstance(D, BlockDiag)
    d = D.shape[0]
    n = D.shape[1] // d
    acc = []
    for k in range(d):
        ak = a[k * n : k * n + n]
        acc.append(ddot(_get_block(D, i, k), ak, left=True))
    return add.reduce(acc)


def _bd_dot_dd(A, B):

    m = A.shape[1] // A.shape[0]

    C = []
    for i in range(A.shape[0]):

        row = []
        for j in range(B.shape[1] // m):
            n = B.shape[0]
            r = sum(A[i].reshape((n, m)) * B[:, j * m : (j * m + m)], axis=0)
            row.append(r)
        row = concatenate(row)
        C.append(row)

    return stack(C, axis=0)


def _get_block(D, i, j):
    n = D.shape[1] // D.shape[0]
    return D[i, j * n : j * n + n]


def _rinv(K):
    m = K.shape[1] // K.shape[0]

    if K.shape[0] == 1:
        return 1 / K

    A = K[:-1][:, :-m]
    B = K[:-1][:, -m:]
    C = K[-1:][:, :-m]
    D = K[-1:][:, -m:]

    do = _bd_dot_dd
    Ai = _rinv(A)
    D_ = _rinv(D - do(do(C, Ai), B))

    A_ = Ai + do(do(do(do(Ai, B), D_), C), Ai)
    B_ = -do(Ai, do(B, D_))
    C_ = -do(D_, do(C, Ai))

    return block([[A_, B_], [C_, D_]])


def _logdet(K, m):
    if K.shape[0] == 1:
        return sum(log(K))

    A = K[:-1][:, :-m]
    B = K[:-1][:, -m:]
    C = K[-1:][:, :-m]
    D = K[-1:][:, -m:]

    do = _bd_dot_dd
    Ai = _rinv(A)

    v0 = _logdet(D - do(do(C, Ai), B), m)
    v1 = _logdet(A, m)
    return v0 + v1


def _vec(A):
    return A.reshape((-1, 1), order="F")


def dot_vec(A, B):
    r""" Implements ``dot(A, vec(B))``. """
    return dot(A, B.reshape((-1, 1), order="F"))
