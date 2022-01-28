import ctypes

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


def own_sp_mx_to_coo(arr):
    assert is_own_sp_mx(arr), 'only works for instances of class DictArray2DClass'
    from scipy import sparse as sp
    coo = sp.coo_matrix(arr.shape, arr.dtype)

    n = arr.nnz()
    idx_dtype = {'float32': np.int32, 'float64': np.int64}[arr.dtype.name]
    coo.row, coo.col, coo.data = map(lambda dtype: np.empty((n,), dtype), (idx_dtype, idx_dtype, arr.dtype))

    arr.to_coo(coo.row, coo.col, coo.data)

    return coo


def DictArray2D(shape, dtype):
#    DictArray2DType = nb.types.deferred_type()
    dtype = np.dtype(dtype)
    np_idx_type = {
        'float32': np.dtype(np.int32),
        'float64': np.dtype(np.int64),
    }[dtype.name]

    nb_type, idx_type, didx_type = {
        'float32': [nb.float32, nb.int32, nb.types.UniTuple(nb.int32, 2)],
        'float64': [nb.float64, nb.int64, nb.types.UniTuple(nb.int64, 2)],
    }[dtype.name]
    dict_type = nb.types.DictType(didx_type, nb_type)

    @nb.experimental.jitclass([
        ('shape', nb.types.UniTuple(idx_type, 2)),
        ('dtype', nb.typeof(dtype)),
        ('idx_type', nb.typeof(np_idx_type)),
        ('data', dict_type),
    ])
    class DictArray2DClass:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self.idx_type = np_idx_type
            self.data = nb.typed.Dict.empty(didx_type, nb_type)

        def __setitem__(self, idx, val):
            # array indices handled with overloaded functions  (NOTE: overloading doesnt seem to work though)

            # NOTE: overloading doesnt seem to work when calling from nopython,
            #       uncommenting the following assert enables overloading when calling from normal python,
            #       however, __setitem__ then fails even for the simple indexing case when called from nopython
            # assert False, 'overloaded'

            i, j = idx
            idx = (self.idx_type.type(i), self.idx_type.type(j))
            self.data[idx] = val

        def __getitem__(self, idx):
            # array indices handled with copyto as had problems with returning arrays
            i, j = idx
            idx = (self.idx_type.type(i), self.idx_type.type(j))
            return self.data.get(idx, self.dtype.type(0.0))

        def copyto(self, idx, trg):
            assert False, 'overloaded'

        def to_coo(self, rows, cols, data):
            for k, ((i, j), cell) in enumerate(self.data.items()):
                rows[k] = i
                cols[k] = j
                data[k] = cell

        def mult_with_arr(self, other):
            if other.size == 1:
                for idx, cell in self.data.items():
                    self.data[idx] *= other[0, 0]

            elif self.shape[0] == other.shape[0]:
                for (i, j), cell in self.data.items():
                    self.data[(i, j)] *= other[i, 0]

            elif self.shape[1] == other.shape[1]:
                for (i, j), cell in self.data.items():
                    self.data[(i, j)] *= other[j, 0]

            else:
                assert False, 'fail!'               #       should do the exceptions better

        def isfinite(self):
            for cell in self.data.values():
                if not np.isfinite(cell):
                    return False
            return True

        def nnz(self):
            return len(self.data)

    # DictArray2DType.define(DictArray2DClass.class_type.instance_type)

    # or could be DictArray2DClass.class_type.instance_type instead of nb.types.misc.ClassInstanceType?
    @nb_extending.overload_method(nb.types.misc.ClassInstanceType, 'copyto')
    #     , jitoptions=dict(signature=[
    #     nb.void(DictArray2DType, nb.types.UniTuple(nb.int32, 2), nb.float32[:]),
    #     nb.void(DictArray2DType, nb.types.Tuple((nb.int32[:], nb.int32)), nb.float32[:]),
    # ]))
    def copyto_impl(obj, idx, trg):
        major, minor = idx

        def isarray(x):
            return isinstance(x, nb.types.IterableType)

        if not isarray(major) and not isarray(minor):
            def _copyto(obj, idx, trg):
                major, minor = idx
                idx = (obj.idx_type.type(major), obj.idx_type.type(minor))
                trg[0] = obj.data.get(idx, obj.dtype.type(0.0))
        elif isarray(major) and not isarray(minor):
            def _copyto(obj, idx, trg):
                major, minor = idx
                minor = obj.idx_type.type(minor)
                for k, i in enumerate(major):
                    i = obj.idx_type.type(i)
                    trg[k] = obj.data.get((i, minor), obj.dtype.type(0.0))
        elif not isarray(major) and isarray(minor):
            def _copyto(obj, idx, trg):
                major, minor = idx
                major = obj.idx_type.type(major)
                for k, j in enumerate(minor):
                    j = obj.idx_type.type(j)
                    trg[k] = obj.data.get((major, j), obj.dtype.type(0.0))
        elif isarray(major) and isarray(minor):
            def _copyto(obj, idx, trg):
                major, minor = idx
                for k, (i, j) in enumerate(zip(major, minor)):
                    i = obj.idx_type.type(i)
                    j = obj.idx_type.type(j)
                    trg[k] = obj.data.get((i, j), obj.dtype.type(0.0))
        else:
            assert False, 'wrong types'

        return _copyto

    # or could be DictArray2DClass.class_type.instance_type instead of nb.types.misc.ClassInstanceType?
    @nb_extending.overload_method(nb.types.misc.ClassInstanceType, '__setitem__')
    def setitem_impl(obj, idx, val):
        major, minor = idx

        def isarray(x):
            return isinstance(x, nb.types.IterableType)

        if not isarray(major) and not isarray(minor):
            def _setitem(obj, idx, val):
                major, minor = idx
                idx = (obj.idx_type.type(major), obj.idx_type.type(minor))
                obj.data[idx] = val

        elif isarray(major) and not isarray(minor):
            if isarray(val):
                def _setitem(obj, idx, val):
                    major, minor = idx
                    minor = obj.idx_type.type(minor)
                    for k, i in enumerate(major):
                        i = obj.idx_type.type(i)
                        obj.data[(i, minor)] = val[k]
            else:
                def _setitem(obj, idx, val):
                    major, minor = idx
                    minor = obj.idx_type.type(minor)
                    for k, i in enumerate(major):
                        i = obj.idx_type.type(i)
                        obj.data[(i, minor)] = val

        elif not isarray(major) and isarray(minor):
            if isarray(val):
                def _setitem(obj, idx, val):
                    major, minor = idx
                    major = obj.idx_type.type(major)
                    for k, j in enumerate(minor):
                        j = obj.idx_type.type(j)
                        obj.data[(major, j)] = val[k]
            else:
                def _setitem(obj, idx, val):
                    major, minor = idx
                    major = obj.idx_type.type(major)
                    for k, j in enumerate(minor):
                        j = obj.idx_type.type(j)
                        obj.data[(major, j)] = val

        elif isarray(major) and isarray(minor):
            if isarray(val):
                def _setitem(obj, idx, val):
                    major, minor = idx
                    for k, (i, j) in enumerate(zip(major, minor)):
                        i = obj.idx_type.type(i)
                        j = obj.idx_type.type(j)
                        obj.data[(i, j)] = val[k]
            else:
                def _setitem(obj, idx, val):
                    major, minor = idx
                    for k, (i, j) in enumerate(zip(major, minor)):
                        i = obj.idx_type.type(i)
                        j = obj.idx_type.type(j)
                        obj.data[(i, j)] = val
        else:
            assert False, 'wrong types'

        return _setitem

    return DictArray2DClass(shape, dtype)
