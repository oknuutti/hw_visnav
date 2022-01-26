import numpy as np
import numba as nb
from numba import extending as nb_extending

from numpy.lib.twodim_base import triu


@nb_extending.overload(np.linalg.qr)
def qr_impl(a, mode):
    if nb.literally(mode).literal_value != 'complete':
        return None

    numba_ez_geqrf = nb.types.ExternalFunction("numba_ez_geqrf", nb.types.intc(
        nb.types.char,  # kind
        nb.types.intp,  # m
        nb.types.intp,  # n
        nb.types.CPointer(a.dtype),  # a
        nb.types.intp,  # lda
        nb.types.CPointer(a.dtype),  # tau
    ))

    numba_ez_xxgqr = nb.types.ExternalFunction("numba_ez_xxgqr", nb.types.intc(
        nb.types.char,  # kind
        nb.types.intp,  # m
        nb.types.intp,  # n
        nb.types.intp,  # k
        nb.types.CPointer(a.dtype),  # a
        nb.types.intp,  # lda
        nb.types.CPointer(a.dtype),  # tau
    ))

    def _qr_complete(a, mode):
        """
        Based on numpy.linalg.qr(a, mode='complete'). This doesn't support complex numbers or multiple arrays.
        Had to make own version as numba does not support the 'complete' argument.
        """
        if a.ndim != 2:
            assert False, 'fail'  # TODO: exceptions
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
            assert False, 'fail'    # TODO: exceptions

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
            assert False, 'fail'    # TODO: exceptions

        q = q[:mc].T
        r = triu(a[:, :mc].T)
        return q, r

    return _qr_complete


def is_own_sp_mx(arr):
    return arr is not None and arr.__class__.__name__ == 'DictArray2DClass'


def DictArray2D(shape, dtype):
    DictArray2DType = nb.types.deferred_type()

    nb_type, idx_type, col_dict_type = {
        np.float32: [nb.float32, nb.int32, nb.types.DictType(nb.int32, nb.float32)],
        np.float64: [nb.float64, nb.int64, nb.types.DictType(nb.int64, nb.float64)],
    }[dtype]

    @nb.experimental.jitclass([
        ('shape', nb.types.UniTuple(idx_type, 2)),
        ('data', nb.types.DictType(idx_type, col_dict_type)),
    ])
    class DictArray2DClass:
        def __init__(self, shape):
            self.shape = shape
            self.data = nb.typed.Dict.empty(idx_type, col_dict_type)

        def __setitem__(self, idx, val):
            pass

        def __getitem__(self, idx):
            major, minor = idx
            val = self.data.get(major, nb.typed.Dict.empty(idx_type, nb_type)).get(minor, dtype(0.0))
            return val

        def copyto(self, idx, trg):
            pass

        def mult_with_arr(self, other):
            if other.size == 1:
                for row in self.data.values():
                    for key in row.keys():
                        row[key] *= other[0, 0]

            elif self.shape[0] == other.shape[0]:
                for i, row in self.data.items():
                    for j in row.keys():
                        row[j] *= other[i, 0]

            elif self.shape[1] == other.shape[1]:
                for row in self.data.values():
                    for j in row.keys():
                        row[j] *= other[0, j]

            else:
                assert False, 'fail!'    # should do the exceptions better

        def isfinite(self):
            for row in self.data.values():
                for cell in row.values():
                    if not np.isfinite(cell):
                        return False
            return True

    DictArray2DType.define(DictArray2DClass.class_type.instance_type)


    @nb_extending.overload_method(nb.types.misc.ClassInstanceType, 'copyto')
    #     , jitoptions=dict(signature=[
    #     nb.void(DictArray2DType, nb.types.UniTuple(nb.int32, 2), nb.float32[:]),
    #     nb.void(DictArray2DType, nb.types.Tuple((nb.int32[:], nb.int32)), nb.float32[:]),
    # ]))
    def copyto_fn(obj, idx, trg):
        major, minor = idx

        def isarray(x):
            return isinstance(x, nb.types.IterableType)

        if not isarray(major) and not isarray(minor):
            def _copyto(obj, idx, trg):
                major, minor = idx
                trg[0] = obj.data.get(major, nb.typed.Dict.empty(idx_type, nb_type)).get(minor, dtype(0.0))
        elif isarray(major) and not isarray(minor):
            def _copyto(obj, idx, trg):
                major, minor = idx
                for k, i in enumerate(major):
                    trg[k] = obj.data.get(i, nb.typed.Dict.empty(idx_type, nb_type)).get(minor, dtype(0.0))
        elif not isarray(major) and isarray(minor):
            def _copyto(obj, idx, trg):
                major, minor = idx
                for k, j in enumerate(minor):
                    trg[k] = obj.data.get(major, nb.typed.Dict.empty(idx_type, nb_type)).get(j, dtype(0.0))
        elif isarray(major) and isarray(minor):
            def _copyto(obj, idx, trg):
                major, minor = idx
                for k, (i, j) in enumerate(zip(major, minor)):
                    trg[k] = obj.data.get(i, nb.typed.Dict.empty(idx_type, nb_type)).get(j, dtype(0.0))
        else:
            assert False, 'wrong types'

        return _copyto


    @nb_extending.overload_method(nb.types.misc.ClassInstanceType, '__setitem__')
    def setitem_fn(obj, idx, val):
        major, minor = idx

        def isarray(x):
            return isinstance(x, nb.types.IterableType)

        if not isarray(major) and not isarray(minor):
            def _setitem(obj, idx, val):
                major, minor = idx
                obj.data.setdefault(major, nb.typed.Dict.empty(idx_type, nb_type))[minor] = val

        elif isarray(major) and not isarray(minor):
            if isarray(val):
                def _setitem(obj, idx, val):
                    major, minor = idx
                    for k, i in enumerate(major):
                        obj.data.setdefault(i, nb.typed.Dict.empty(idx_type, nb_type))[minor] = val[k]
            else:
                def _setitem(obj, idx, val):
                    major, minor = idx
                    for k, i in enumerate(major):
                        obj.data.setdefault(i, nb.typed.Dict.empty(idx_type, nb_type))[minor] = val

        elif not isarray(major) and isarray(minor):
            if isarray(val):
                def _setitem(obj, idx, val):
                    major, minor = idx
                    for k, j in enumerate(minor):
                        obj.data.setdefault(major, nb.typed.Dict.empty(idx_type, nb_type))[j] = val[k]
            else:
                def _setitem(obj, idx, val):
                    major, minor = idx
                    for k, j in enumerate(minor):
                        obj.data.setdefault(major, nb.typed.Dict.empty(idx_type, nb_type))[j] = val

        elif isarray(major) and isarray(minor):
            if isarray(val):
                def _setitem(obj, idx, val):
                    major, minor = idx
                    for k, (i, j) in enumerate(zip(major, minor)):
                        obj.data.setdefault(i, nb.typed.Dict.empty(idx_type, nb_type))[j] = val[k]
            else:
                def _setitem(obj, idx, val):
                    major, minor = idx
                    for k, (i, j) in enumerate(zip(major, minor)):
                        obj.data.setdefault(i, nb.typed.Dict.empty(idx_type, nb_type))[j] = val
        else:
            assert False, 'wrong types'

        return _setitem

    return DictArray2DClass(shape)
