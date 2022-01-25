import numpy as np
import numba as nb

from numpy.lib.twodim_base import triu


def qr_complete_fn(type):
    numba_ez_geqrf = nb.types.ExternalFunction("numba_ez_geqrf", nb.types.intc(
        nb.types.char,  # kind
        nb.types.intp,  # m
        nb.types.intp,  # n
        nb.types.CPointer(type),  # a
        nb.types.intp,  # lda
        nb.types.CPointer(type),  # tau
    ))

    numba_ez_xxgqr = nb.types.ExternalFunction("numba_ez_xxgqr", nb.types.intc(
        nb.types.char,  # kind
        nb.types.intp,  # m
        nb.types.intp,  # n
        nb.types.intp,  # k
        nb.types.CPointer(type),  # a
        nb.types.intp,  # lda
        nb.types.CPointer(type),  # tau
    ))

    errs = {
        1: 'Array must be two-dimensional',
        2: 'numba_ez_geqrf returned error',
        3: 'numba_ez_xxgqr returned error',
    }

    @nb.njit(nogil=True, parallel=False, cache=True)
    def _qr_complete(a):
        """
        Based on numpy.linalg.qr(a, mode='complete'). This doesn't support complex numbers or multiple arrays.
        Had to make own version as numba does not support the 'complete' argument.
        """
        if a.ndim != 2:
            return 1, None, None
        m, n = a.shape
        mn = min(m, n)

        dtype = a.dtype
        kind = ord('d' if dtype is nb.float64 else 's')

        # NOTE: byteorder attribute missing from numba dtype
        # if dtype.byteorder not in ('=', '|'):
        #     a = np.asarray(a, dtype=a.dtype.newbyteorder('='))
        a = a.T.copy()
        tau = np.empty(mn, dtype=dtype)

        # calculate optimal size of work data 'work'
        res = numba_ez_geqrf(kind, m, n, a.ctypes, max(1, m), tau.ctypes)
        if res != 0:
            return 2, None, None

        #  generate q from a
        if m > n:
            mc = m
            q = np.empty((m, m), dtype)
        else:
            mc = mn
            q = np.empty((n, m), dtype)
        q[:n] = a

        # compute q
        res = numba_ez_xxgqr(kind, m, mc, mn, q.ctypes, max(1, m), tau.ctypes)
        if res != 0:
            return 3, None, None

        q = q[:mc].T
        r = a[:, :mc].T
        return 0, q, triu(r)

    return _qr_complete
