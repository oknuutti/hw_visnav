import logging
import warnings

import numpy as np
from scipy import sparse as sp
import numba as nb
#from memory_profiler import profile

from visnav.algo import tools
from visnav.algo.linalg import is_own_sp_mx, DictArray2D, own_sp_mx_to_coo
from visnav.algo.tools import maybe_decorate

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NUMBA_LEVEL = 3
#logging.getLogger('numba').setLevel(logging.DEBUG)


class InnerLinearizerQR:

    (
        STATE_INITIALIZED,
        STATE_LINEARIZED,
        STATE_MARGINALIZED,
        STATE_NUMERICAL_FAILURE,
    ) = range(4)

    def __init__(self, problem, jacobi_scaling_eps=1e-5, huber_coefs=None, use_weighted_residuals=False,
                 use_own_sp_mx=NUMBA_LEVEL >= 4):
        self.problem = problem
        self.dtype = self.problem.dtype
        self.idx_dtype = problem.IDX_DTYPE
        self.jacobi_scaling_eps = jacobi_scaling_eps

        huber_coefs = [None]*3 if huber_coefs is None else huber_coefs
        self.huber_coef_repr = huber_coefs[0]
        self.huber_coef_loc = huber_coefs[1]
        self.huber_coef_ori = huber_coefs[2]
        self.use_weighted_residuals = use_weighted_residuals
        self.use_own_sp_mx = use_own_sp_mx

        self.use_valid_projections_only = True
        self._pose_damping = None

        self._blocks = None
        self._x_block = None
        self._a_block = None
        self._pose_size = self.problem.pose_size
        self._pose_n = len(self.problem.poses)
        self.nb = len(self.problem.xb)
        self.np = self._pose_size * self._pose_n
        self.nl = 3
        # self.mr = 2 * len(self.problem.pts2d)
        self.mm = ((3 if self.problem.meas_r is not None and len(self.problem.meas_r) > 0 else 0)
                   + (3 if self.problem.meas_aa is not None and len(self.problem.meas_aa) > 0 else 0)
                  ) * (len(self.problem.meas_idxs) if self.problem.meas_idxs is not None else 0)

        self._total_error = None

        self._state = self.STATE_INITIALIZED

    @property
    def non_lm_blocks(self):
        return ([self._x_block] if self._x_block is not None else []) \
             + ([self._a_block] if self._a_block is not None else [])

    @property
    def all_blocks(self):
        def generator():
            for blk in self._blocks:
                yield blk
            for blk in self.non_lm_blocks:
                yield blk
        return generator()

    def linearize(self):
        mr, mx, ma = self.problem.pts2d.size, self.problem.meas_r.size, self.problem.meas_aa.size
        m = mr + mx + ma

        rr, *rxra = self.problem.residual(parts=True)
        spf = 'own' if self.use_own_sp_mx else 'csr'
        (Jrb, Jrp, Jrl), *JxJa = self.problem.jacobian(parts=True, r_fmt=('dense', spf, spf))
        rx, ra, Jxb, Jxp, Jab, Jap = [None] * 6

        if self.use_valid_projections_only:
            if self.use_own_sp_mx:
                numerically_valid = Jrp.isfinite() and Jrl.isfinite()
            else:
                numerically_valid = np.all(np.isfinite(Jrp.data if sp.issparse(Jrp) else Jrp)) \
                                    and np.all(np.isfinite(Jrl.data if sp.issparse(Jrl) else Jrl))
            numerically_valid = numerically_valid \
                                and (Jrb is None or np.all(np.isfinite(Jrb.data if sp.issparse(Jrb) else Jrb))) \
                                and np.all(np.isfinite(rr))
            if not numerically_valid:
                self._state = self.STATE_NUMERICAL_FAILURE
                return

        rr, (Jrb, Jrp, Jrl), err = self._apply_huber(rr, (Jrb, Jrp, Jrl), m, self.huber_coef_repr)
        self._total_error = err

        if mx > 0:
            Jxb, Jxp, _ = JxJa[0]
            rx = rxra[0]
            rx, (Jxb, Jxp), err = self._apply_huber(rx, (Jxb, Jxp), m, self.huber_coef_loc)
            self._x_block = NonLMResidualBlock(self.problem.meas_idxs, rx, Jxb, Jxp, psize=self._pose_size)
            self._total_error += err

        if ma > 0:
            Jab, Jap, _ = JxJa[0 if mx == 0 else 1]
            ra = rxra[0 if mx == 0 else 1]
            ra, (Jab, Jap), err = self._apply_huber(ra, (Jab, Jap), m, self.huber_coef_ori)
            self._a_block = NonLMResidualBlock( self.problem.meas_idxs, ra, Jab, Jap, psize=self._pose_size)
            self._total_error += err

        if self._blocks is None:
            if NUMBA_LEVEL >= 3:
                self._blocks = nb.typed.List.empty_list(ResidualBlockType, len(self.problem.valid_pts3d))
            else:
                self._blocks = []

        if NUMBA_LEVEL >= 4:
            self._bld_lm_blks(self._blocks, self.problem.valid_pts3d, self.problem.pt3d_idxs, self.problem.pose_idxs,
                              rr, Jrb, Jrp, Jrl, self.idx_dtype, self._pose_size)
        else:
            self.build_landmark_blocks(rr, Jrb, Jrp, Jrl)

        self._state = self.STATE_LINEARIZED

    @staticmethod
    @maybe_decorate(nb.njit(nogil=True, parallel=False, cache=True), NUMBA_LEVEL >= 4)
    def _bld_lm_blks(blocks, valid_pts3d, pt3d_idxs, pose_idxs, rr, Jrb, Jrp, Jrl, idx_dtype, pose_size):
        # TODO: debug, some problem with logic present, results are poorer than previous method
        #       - However, not sure if worthwhile to fix, seems execution times are very similar
        #       - NUMBA_LEVEL == 2-3, seems best execution time -wise, not certain about memory use

        blk_i = 0
        for pt_i in np.arange(len(valid_pts3d), dtype=idx_dtype):
            r_idxs = np.where(pt3d_idxs == pt_i)[0].astype(idx_dtype)
            blk_pose_idxs = pose_idxs[r_idxs]
            blk_np = len(r_idxs)
            m, n_b, n_p, n_l = blk_np * 2, Jrb.shape[1], blk_np * pose_size, 3

            if blk_np < 2:
                # too few observations of this 3d-point
                valid_pts3d[pt_i] = False
                continue

            if len(blocks) <= blk_i:
                blocks.append(ResidualBlock(r_idxs, blk_pose_idxs, m, n_b, n_p, n_l, pose_size))

            blk = blocks[blk_i]
            blk.reset()
            blk_i += 1

            for i, ri in enumerate(r_idxs):
                i = idx_dtype(i)
                for j in range(n_b):
                    j = idx_dtype(j)
                    blk.S[i*2,     j] = Jrb[ri*2,     j]
                    blk.S[i*2 + 1, j] = Jrb[ri*2 + 1, j]

                for j in range(pose_size):
                    j = idx_dtype(j)
                    blk.S[i*2,     n_b + i * pose_size + j] = Jrp[ri*2,     blk_pose_idxs[i] + j]
                    blk.S[i*2 + 1, n_b + i * pose_size + j] = Jrp[ri*2 + 1, blk_pose_idxs[i] + j]

                for j in range(n_l):
                    j = idx_dtype(j)
                    blk.S[i*2,     n_b + n_p + j] = Jrl[ri*2,     pt_i * 3 + j]
                    blk.S[i*2 + 1, n_b + n_p + j] = Jrl[ri*2 + 1, pt_i * 3 + j]

                blk.S[i * 2,     -1] = rr[ri * 2,     idx_dtype(0)]
                blk.S[i * 2 + 1, -1] = rr[ri * 2 + 1, idx_dtype(0)]

    def build_landmark_blocks(self, rr, Jrb, Jrp, Jrl):
        def index(A, i, j, safe=False):
            if is_own_sp_mx(A):
                B = np.empty(len(i), dtype=self.dtype)
                A.copyto((i, j), B)
                return B
            # safe is very slow, unsafe is dangerous as bugs won't be found
            return A[i, j] if safe else A._get_arrayXarray(i, j)

        blk_i = 0
        for i in range(len(self.problem.pts3d)):
            r_idxs = np.where(self.problem.pt3d_idxs == i)[0].astype(self.idx_dtype)
            pose_idxs = self.problem.pose_idxs[r_idxs]
            blk_np = len(r_idxs)

            if blk_np < 2:
                # too few observations of this 3d-point
                self.problem.valid_pts3d[i] = False
                continue

            # indexing sparse array faster this way instead of Jl[v_i, i:i+3]
            _, _, Jrl_i, Jrl_j = self._block_indexing(r_idxs, np.ones((len(r_idxs),), dtype=self.idx_dtype) * i, 2, 3)
            rr_i, _, Jrp_i, Jrp_j = self._block_indexing(r_idxs, pose_idxs, 2, self._pose_size)
            _, _, blk_Jp_i, blk_Jp_j = self._block_indexing(np.arange(blk_np, dtype=self.idx_dtype),
                                                            np.arange(blk_np, dtype=self.idx_dtype), 2, self._pose_size)

            # create residual vector, assign reprojection residuals
            blk_r = np.zeros((blk_np*2, 1), dtype=self.dtype)
            blk_r[:blk_np*2] = rr[rr_i]

            # batch jacobian related to reprojection residuals
            blk_Jb = None if Jrb is None else np.array(Jrb[rr_i, :], dtype=self.dtype)    # dense matrix here

            # create landmark jacobian matrix, assign the three columns related to this landmark
            blk_Jl = np.zeros((blk_np*2, 3), dtype=self.dtype)
            blk_Jl[:blk_np*2, :] = np.array(index(Jrl, Jrl_i, Jrl_j).reshape((-1, 3)), dtype=self.dtype)

            # create pose/frame jacobian matrix, assign all frames that observe this landmark
            blk_Jp = np.zeros((blk_np*2, blk_np*self._pose_size), dtype=self.dtype)
            blk_Jp[blk_Jp_i, blk_Jp_j] = index(Jrp, Jrp_i, Jrp_j)

            if len(self._blocks) <= blk_i:
                m, n_b, n_p, n_l = blk_r.size, blk_Jb.shape[1], blk_Jp.shape[1], blk_Jl.shape[1]
                self._blocks.append(ResidualBlock(r_idxs, pose_idxs, m, n_b, n_p, n_l, self._pose_size))
            self._blocks[blk_i].new_linearization_point(blk_r, blk_Jb, blk_Jp, blk_Jl)
            blk_i += 1

    @staticmethod
    def _block_indexing(row_idxs, col_idxs, block_rows, block_cols):
        row_idxs_r = InnerLinearizerQR._augment_idxs(row_idxs, block_rows)
        row_idxs_rc = np.repeat(row_idxs_r, block_cols)
        col_idxs_c = InnerLinearizerQR._augment_idxs(col_idxs, block_cols)
        col_idxs_cr = np.repeat(col_idxs_c.reshape((-1, block_cols)), block_rows, axis=0).flatten()
        return row_idxs_r, col_idxs_c, row_idxs_rc, col_idxs_cr

    @staticmethod
    def _augment_idxs(idxs, size):
        return (np.repeat(idxs[:, None] * size, size, axis=1) + np.arange(size, dtype=idxs.dtype)[None, :]).flatten()

    def _apply_huber(self, r, arr_J, total_nr, huber_coef):
        huber_coef = np.inf if huber_coef is None else huber_coef

        residual_weight = total_nr / len(r) if self.use_weighted_residuals else 1.0

        abs_r = np.abs(r)
        I = abs_r > huber_coef
        huber_weight = np.ones_like(r)
        huber_weight[I] *= np.sqrt(huber_coef / abs_r[I])
        dwr_dr = np.sqrt(residual_weight * huber_weight)

        wr = dwr_dr * r

        if arr_J is None:
            wJ = None
        else:
            wJ = []
            for i, J in enumerate(arr_J):
                if J is None:
                    wJ.append(J)
                elif sp.issparse(J):
                    wJ.append(sp.diags(dwr_dr.flatten()).dot(J))
                elif is_own_sp_mx(J):
                    J.imul_arr(dwr_dr.reshape((dwr_dr.size, -1)))
                    wJ.append(J)
                else:
                    J *= dwr_dr
                    wJ.append(J)

        err = np.sum(0.5 * (2 - huber_weight) * huber_weight * r ** 2)
        return wr, wJ, err

    def initial_cost(self):
        return self._total_error / self.residual_count()

    def residual_count(self):
        return self.problem.pts2d.size + self.problem.meas_r.size + self.problem.meas_aa.size

    def current_cost(self):
        nr, nx, na = self.problem.pts2d.size, self.problem.meas_r.size, self.problem.meas_aa.size
        rr, *rxra = self.problem.residual(parts=True)

        _, _, err = self._apply_huber(rr, None, nr+nx+na, self.huber_coef_repr)
        total_error = err

        if nx > 0:
            rx = rxra[0]
            _, _, err = self._apply_huber(rx, None, nr+nx+na, self.huber_coef_loc)
            total_error += err

        if na > 0:
            ra = rxra[0 if nx == 0 else 1]
            _, _, err = self._apply_huber(ra, None, nr+nx+na, self.huber_coef_ori)
            total_error += err

        return total_error / (nr + nx + na)

    def filter(self, max_repr_err):
        n_obs = len(self.problem.pts2d)
        obs_idx = self.problem.filter(max_repr_err)
        cost = self.current_cost()
        return len(obs_idx)/n_obs, cost

    def set_pose_damping(self, _lambda):
        self._pose_damping = _lambda

    def marginalize(self):
        assert self._state == self.STATE_LINEARIZED, 'not linearized yet'
        if NUMBA_LEVEL >= 3:
            with warnings.catch_warnings():
                # for some reason warns about non-contiguous arrays in n_p.dot, can't find any though
                warnings.filterwarnings("ignore")
                self._marginalize(nb.typed.List(self._blocks))
        else:
            self._marginalize(self._blocks)
        self._state = self.STATE_MARGINALIZED

    @staticmethod
    @maybe_decorate(nb.njit(nogil=True, parallel=False, cache=True), NUMBA_LEVEL >= 3)
    def _marginalize(blocks):
        for blk in blocks:
            blk.marginalize()

    def backsub_xl(self, delta_xbp):
        assert self._state == self.STATE_MARGINALIZED, 'not linearized yet'

        l_diff = 0
        delta_xb = delta_xbp[:self.nb]
        delta_xp = delta_xbp[self.nb:].reshape((-1, self._pose_size))
        delta_xl = np.zeros((len(self._blocks), self._blocks[0].n_l), dtype=self.dtype)
        for i, blk in enumerate(self.all_blocks):
            blk_delta_xp = delta_xp[blk.pose_idxs, :].reshape((-1, 1))
            if blk.n_l > 0:
                blk_delta_xbp = np.concatenate((delta_xb, blk_delta_xp), axis=0)
                blk_delta_xl = -np.linalg.solve(blk.R1, blk.Q1T_r + blk.Q1T_Jbp.dot(blk_delta_xbp))
                delta_xl[i, :] = blk_delta_xl.T * blk.Jl_col_scale
            else:
                blk_delta_xbp = blk_delta_xp

            # calculate linear cost diff
            blk.damp(0)
            QT_J_deltax = blk.QT_Jbp @ blk_delta_xbp
            if blk.n_l > 0:
                QT_J_deltax[:blk.n_l] += blk.R1.dot(blk_delta_xl)
            l_diff -= QT_J_deltax.T.dot(0.5 * QT_J_deltax + blk.QT_r).flatten()[0]

        return delta_xl, l_diff / self.residual_count()

    # def get_Q2Tr(self):
    #     return self._Q2.T.dot(self._r)
    #
    # def get_Q2TJp(self):
    #     return self._Q2.T.dot(self._Jp)
    #
    # def get_Q2TJp_premult_x(self, x_r):
    #     return x_r.T.dot(self._Q2.T).dot(self._Jp)
    #
    # def get_Q2TJp_postmult_x(self, xp):
    #     return self._Q2.T.dot(self._Jp.dot(xp))

    def get_Q2TJbp_T_Q2TJbp_mult_x(self, xp):
        assert self._state == self.STATE_MARGINALIZED
        assert len(self.problem.cam_param_idxs) == 0, 'not implemented for cam param estimation'
        assert len(self.problem.meas_aa) == 0, 'not implemented for absolute measures'
        assert len(self.problem.meas_r) == 0, 'not implemented for absolute measures'

        y = np.zeros((self._pose_n, self._pose_size))
        for blk in self._blocks:
            blk_xp = xp.reshape((-1, self._pose_size))[blk.pose_idxs, :].reshape((-1, 1))
            blk_y = blk.Q2T_Jbp.T.dot(blk.Q2T_Jbp.dot(blk_xp))
            y[blk.pose_idxs, :] += blk_y.reshape((-1, self._pose_size))

        return y.reshape((-1, 1))

    def get_Q2TJbp_T_Q2Tr(self):
        assert self._state == self.STATE_MARGINALIZED

        brb = np.zeros((self.nb, 1), dtype=self.dtype)
        bp = np.zeros((self._pose_n, self._pose_size), dtype=self.dtype)
        for blk in self.all_blocks:
            Q2T_Jbp = blk.Q2T_Jbp
            if sp.issparse(Q2T_Jbp):
                blk_b = (blk.Q2T_r.T @ Q2T_Jbp).T
            else:
                blk_b = (blk.Q2T_r.T.dot(Q2T_Jbp)).T
            if blk.n_b > 0:
                brb[:self.nb] += blk_b[:blk.n_b]
            bp[blk.pose_idxs, :] += blk_b[blk.n_b:].reshape((-1, self._pose_size))

        return np.concatenate((brb, bp.reshape((-1, 1))), axis=0)

    # def get_Q2TJp_diag2(self):
    #     pass

    def _epow2(self, arr):
        if type(arr) == np.ndarray:
            return arr ** 2
        else:
            assert sp.issparse(arr)
            arr = arr.copy()
            arr.data *= arr.data
            return arr

    def get_Jbp_diag2(self):
        assert self._state == self.STATE_LINEARIZED

        db2 = np.zeros((self.nb,), dtype=self.dtype)
        dp2 = np.zeros((self._pose_n, self._pose_size), dtype=self.dtype)
        for blk in self.all_blocks:
            if blk.n_b > 0:
                db2 += np.sum(self._epow2(blk.Jb), axis=0)
            dp2[blk.pose_idxs, :] += np.sum(self._epow2(blk.Jp), axis=0).reshape((-1, self._pose_size))

        return np.concatenate((db2, dp2.flatten()))

    def get_Q2TJbp_T_Q2TJbp_blockdiag(self):
        assert self._state == self.STATE_MARGINALIZED

        # TODO: don't use dense Hpp

        if 0:
            # too slow
            Hpp = self._get_Q2TJbp_T_Q2TJbp_blockdiag_sp(self.nb, self.np, self._pose_size,
                                                         self._pose_damping, self._blocks, self.dtype)
        elif NUMBA_LEVEL >= 3:
            # no extra memory used
            Hpp = DictArray2D((self.nb + self.np, self.nb + self.np), dtype=self.dtype)
            self._get_Q2TJbp_T_Q2TJbp_blockdiag_lv3(self.nb, self.np, self._pose_size,
                                                    self._pose_damping, self._blocks, Hpp)
        else:
            # uses way too much memory for large problems due to dense Hpp
            Hpp = self._get_Q2TJbp_T_Q2TJbp_blockdiag(self.nb, self.np, self._pose_size, self._pose_damping,
                                                      nb.typed.List([blk.Q2T_Jbp for blk in self._blocks]),
                                                      nb.typed.List([blk.pose_idxs for blk in self._blocks]))

        for blk in self.non_lm_blocks:
            blk_Hpp = blk.Q2T_Jbp.T.dot(blk.Q2T_Jbp)
            tmp = np.arange(len(blk.pose_idxs))
            _, _, bi, bj = self._block_indexing(tmp, tmp, self._pose_size, self._pose_size)
            _, _, i, j = self._block_indexing(blk.pose_idxs, blk.pose_idxs, self._pose_size, self._pose_size)
            if is_own_sp_mx(Hpp):
                Hpp.idx_isum_arr(i + self.nb, j + self.nb, np.array(blk_Hpp[bi, bj]).flatten())
            else:
                Hpp[i + self.nb, j + self.nb] += np.array(blk_Hpp[bi, bj]).flatten()

        if sp.issparse(Hpp) and sp.isspmatrix_csc(Hpp):
            return Hpp
        if sp.issparse(Hpp):
            return Hpp.tocsc()
        if is_own_sp_mx(Hpp):
            return own_sp_mx_to_coo(Hpp).tocsc()
        return sp.csc_matrix(Hpp)

    @staticmethod
    def _get_Q2TJbp_T_Q2TJbp_blockdiag_sp(n_b, n_p, psize, pose_damping, blocks, dtype):
        H = sp.dok_matrix((n_b + n_p, n_b + n_p), dtype=dtype)

        bj, bi = np.meshgrid(np.arange(n_b), np.arange(n_b))
        pj, pi = np.meshgrid(np.arange(psize), np.arange(psize))
        bj, bi, pj, pi = map(lambda x: x.flatten(), (bj, bi, pj, pi))

        for blk in blocks:
            if n_b > 0:
                A = blk.Q2T_Jbp[:, 0:n_b]  # should copy or call np.ascontiguousarray?
                blk_Hbb = A.T.dot(A).copy()
                H[bi, bj] += blk_Hbb.flatten()

            for j, idx in enumerate(blk.pose_idxs):
                g0, b0 = n_b + idx * psize, n_b + j * psize
                A = blk.Q2T_Jbp[:, b0:b0+psize].copy()
                blk_Hpp = A.T.dot(A)
                H[g0 + pi, g0 + pj] += blk_Hpp.flatten()

        if pose_damping is not None:
            i = np.arange(n_b + n_p, dtype=np.int32)
            H[i, i] += dtype.type(pose_damping)

        return H

    @staticmethod
    @maybe_decorate(nb.njit(nogil=True, parallel=False, cache=True), NUMBA_LEVEL >= 3)
    def _get_Q2TJbp_T_Q2TJbp_blockdiag_lv3(_nb, _np, psize, pose_damping, blocks, H):
        for blk in blocks:
            if _nb > 0:
                A = blk.Q2T_Jbp[:, 0:_nb].copy()
                blk_Hbb = A.T.dot(A)
                for i in range(_nb):
                    for j in range(_nb):
                        H[i, j] = H[i, j] + blk_Hbb[i, j]

            for j, idx in enumerate(blk.pose_idxs):
                g0, b0 = _nb + idx * psize, _nb + j * psize
                g1, b1 = g0 + psize, b0 + psize
                A = blk.Q2T_Jbp[:, b0:b1].copy()
                blk_Hpp = A.T.dot(A)
                for i, gi in enumerate(range(g0, g1)):
                    for j, gj in enumerate(range(g0, g1)):
                        H[gi, gj] = H[gi, gj] + blk_Hpp[i, j]

        if pose_damping is not None:
            val = np.float32(pose_damping)
            for i in range(_nb + _np):
                H[i, i] = H[i, i] + val

    @staticmethod
    @maybe_decorate(nb.njit(nogil=True, parallel=False, cache=True), NUMBA_LEVEL >= 1)
    def _get_Q2TJbp_T_Q2TJbp_blockdiag(_nb, _np, psize, pose_damping, arr_Q2T_Jbp, arr_pose_idxs):
        H = np.zeros((_nb + _np, _nb + _np), dtype=arr_Q2T_Jbp[0].dtype)
        for i, Q2T_Jbp in enumerate(arr_Q2T_Jbp):
            if _nb > 0:
                A = Q2T_Jbp[:, 0:_nb].copy()
                blk_Hbb = A.T.dot(A)
                H[:_nb, :_nb] += blk_Hbb

            for j, idx in enumerate(arr_pose_idxs[i]):
                g0, b0 = _nb + idx * psize, _nb + j * psize
                g1, b1 = g0 + psize, b0 + psize
                A = Q2T_Jbp[:, b0:b1].copy()
                blk_Hpp = A.T.dot(A)
                H[g0:g1, g0:g1] += blk_Hpp

        if pose_damping is not None:
            H += pose_damping * np.eye(H.shape[0], dtype=H.dtype)

        return H

    def get_Jbp_T_Jbp_blockdiag(self):
        assert self._state == self.STATE_LINEARIZED, 'can only use if state == LINEARIZED'
        assert len(self.problem.cam_param_idxs) == 0, 'not implemented for cam param estimation'

        Hpp = np.zeros((self.np, self.np), dtype=self.dtype)
        for blk in self.all_blocks:
            for i, idx in enumerate(blk.pose_idxs):
                g0, b0, br0 = blk.pose_idxs[i] * self._pose_size, i * self._pose_size, i * 2
                g1, b1, br1 = g0 + self._pose_size, b0 + self._pose_size, br0 + 2
                blk_Hpp = blk.Jp[br0:br1, b0:b1].T.dot(blk.Jp[br0:br1, b0:b1])
                Hpp[g0:g1, g0:g1] += blk_Hpp

        if self._pose_damping is not None:
            Hpp += self._pose_damping * np.eye(Hpp.shape[0], dtype=self.dtype)

        return sp.csc_matrix(Hpp)

    def scale_Jl_cols(self):
        assert self._state == self.STATE_LINEARIZED, 'can only use if state == LINEARIZED'
        for blk in self._blocks:
            blk.Jl_col_scale = 1 / (self.jacobi_scaling_eps + np.linalg.norm(blk.Jl, axis=0, keepdims=True))
            blk.Jl *= blk.Jl_col_scale

    def scale_Jbp_cols(self, jac_scaling):
        assert self._state == self.STATE_MARGINALIZED, 'can only use if state == MARGINALIZED'
        jac_rb_scaling = jac_scaling[None, :self.nb]
        jac_rp_scaling = jac_scaling[self.nb:].reshape((-1, self._pose_size))
        for blk in self.all_blocks:
            assert not blk.is_damped() or blk.n_l == 0, 'apply scaling before damping'
            blk_rp = jac_rp_scaling[blk.pose_idxs, :].reshape((1, -1))
            blk_jsc = np.concatenate((jac_rb_scaling, blk_rp), axis=1) if blk.n_l > 0 else blk_rp
            if sp.issparse(blk.QT_Jbp):
                blk.QT_Jbp = blk.QT_Jbp * sp.diags(blk_jsc.flatten())
            else:
                blk_QT_Jbp = blk.QT_Jbp  # direct blk.QT_Jbp inplace multiplication will fail because of @property decorator
                blk_QT_Jbp *= blk_jsc    # same as blk.QT_Jbp = blk.QT_Jbp.dot(n_p.diag(blk_jsc))

    def set_landmark_damping(self, _lambda):
        assert self._state == self.STATE_MARGINALIZED, 'can only use if state == MARGINALIZED'
        for blk in self._blocks:
            blk.damp(_lambda)

    def get_stage1(self, precond_mx):
        assert False, 'not implemented'
        # TODO: would need to restructure so that all block level operations implemented at ResidualBlock

    def get_stage2(self, _lambda, jac_scaling_D, precond_mx):
        assert False, 'not implemented'

    def right_multiply(self, xp):
        return self.get_Q2TJbp_T_Q2TJbp_mult_x(xp)

    def nb(self) -> int:
        return len(self.problem.xb)


nb_type, idx_type, np_type = nb.float32, nb.int32, np.float32
array_type = nb_type[:, :]

@maybe_decorate(
    nb.experimental.jitclass([
        ('r_idxs', idx_type[:]),
        ('pose_idxs', idx_type[:]),
        ('m', nb.intp),
        ('n_b', nb.intp),
        ('n_p', nb.intp),
        ('n_l', nb.intp),
        ('psize', nb.intp),
        ('pnum', nb.intp),
        ('S', array_type),
        ('Jl_col_scale', nb.optional(array_type)),
        ('_marginalized', nb.boolean),
        ('_damping_rots', nb.optional(nb.types.ListType(array_type))),
    ]), NUMBA_LEVEL >= 3)
class ResidualBlock:
    def __init__(self, r_idxs, pose_idxs, m, n_b, n_p, n_l, psize=6):
        self.r_idxs = r_idxs
        self.pose_idxs = pose_idxs
        self.psize = psize
        self.m, self.n_b, self.n_p, self.n_l = m, n_b, n_p, n_l
        self.pnum = self.n_p // self.psize

        self.S = np.empty((self.m + self.n_l, self.n_b + self.n_p + self.n_l + 1), dtype=np_type)
        self.Jl_col_scale = None
        self._marginalized = False
        self._damping_rots = None

    def reset(self):
        self._marginalized = False
        self._damping_rots = None
        self.S[:] = self.S.dtype.type(0.0)

    def new_linearization_point(self, r, Jb, Jp, Jl):
        self.reset()
        self.r, self.Jb, self.Jp, self.Jl = r, Jb, Jp, Jl

    def marginalize(self):
        assert not self.is_marginalized(), 'already marginalized'
        self._marginalize(self.S, self.m, self.n_b, self.n_p, self.n_l)
        self._marginalized = True

    @staticmethod
    @maybe_decorate(nb.njit(nogil=True, parallel=False, cache=True), NUMBA_LEVEL == 2)
    def _marginalize(_S, m, n_b, n_p, n_l):
        Q, R = np.linalg.qr(_S[:m, n_b + n_p:-1], mode='complete')  # NOTE: numba 0.51.1 doesn't support mode argument,
                                                                    #       this uses own qr function at algo/linalg.py
        if 0:
            # S[:m, ...] is not contiguous, need to ignore warnings in non jitted code
            _S[:m, :n_b + n_p] = Q.T.dot(_S[:m, :n_b + n_p])   # Q.T.dot(Jbp)
            _S[:m, n_b + n_p:-1] = R                           # Q.T.dot(Jl)
            _S[:m, -1:] = Q.T.dot(_S[:m, -1:])                 # Q.T.dot(r)
        else:
            # similar speed as the alternative above, no warnings, maybe needs a bit more memory?
            A = np.ascontiguousarray(_S[:m, :])
            _S[:m, :] = Q.T.dot(A)

    def damp(self, _lambda):
        assert self.is_marginalized(), 'not yet marginalized'
        assert _lambda >= 0, 'lambda must be >= 0'

        if self.n_l == 0:
            return

        if self.is_damped():
            self.undamp()

        if _lambda == 0:
            pass  # by default lambda already zero if no damping
        else:
            # make and apply givens rotations
            self.S[-self.n_l:, self.n_b + self.n_p:self.n] = np.eye(self.n_l, dtype=self.S.dtype) * np.sqrt(_lambda)
            self._damping_rots = self._damp(self.S, self.n_b + self.n_p, self.n_l)

    @staticmethod
    @maybe_decorate(nb.njit(nogil=True, parallel=False, cache=True), 3 > NUMBA_LEVEL >= 1)
    def _damp(S, l_idx, k):
        damping_rots = nb.typed.List.empty_list(array_type)
        for n in range(k):
            for m in range(n + 1):
                G = tools.make_givens(S[n, l_idx + n], S[-k + n - m, l_idx + n], dtype=S.dtype)
                tools.apply_givens(S, G, S.shape[0] - k + n - m, n)
                damping_rots.append(G)
        return damping_rots

    def undamp(self):
        assert self.is_damped(), 'not damped yet'
        self._undamp(self._damping_rots, self.S, self.n_l)
        self._damping_rots = None

    @staticmethod
    @maybe_decorate(nb.njit(nogil=True, parallel=False, cache=True), 3 > NUMBA_LEVEL >= 1)
    def _undamp(damping_rots, S, k):
        for n in range(k-1, -1, -1):
            for m in range(n, -1, -1):
                G = damping_rots.pop()
                tools.apply_givens(S, G.T, S.shape[0] - k + n - m, n)    # TODO: contiguous

    def is_marginalized(self):
        return self._marginalized

    def is_damped(self):
        return self._damping_rots is not None or self.n_l == 0

    @property
    def r(self):
        assert not self.is_marginalized(), 'r only defined for non-marginalized landmark blocks'
        return self.S[:self.m, self.n:]

    @r.setter
    def r(self, r):
        assert not self.is_marginalized(), 'r only defined for non-marginalized landmark blocks'
        self.S[:self.m, self.n:] = r

    @property
    def Jb(self):
        assert not self.is_marginalized(), 'Jb only defined for non-marginalized landmark blocks'
        return self.S[:self.m, :self.n_b]

    @Jb.setter
    def Jb(self, Jb):
        assert not self.is_marginalized(), 'Jb only defined for non-marginalized landmark blocks'
        self.S[:self.m, :self.n_b] = Jb

    @property
    def Jp(self):
        assert not self.is_marginalized(), 'Jp only defined for non-marginalized landmark blocks'
        return self.S[:self.m, self.n_b: self.n_b + self.n_p]

    @Jp.setter
    def Jp(self, Jp):
        assert not self.is_marginalized(), 'Jp only defined for non-marginalized landmark blocks'
        self.S[:self.m, self.n_b: self.n_b + self.n_p] = Jp

    @property
    def Jl(self):
        assert not self.is_marginalized(), 'Jl only defined for non-marginalized landmark blocks'
        return self.S[:self.m, self.n_b + self.n_p:-1]

    @Jl.setter
    def Jl(self, Jl):
        assert not self.is_marginalized(), 'Jl only defined for non-marginalized landmark blocks'
        self.S[:self.m, self.n_b + self.n_p:-1] = Jl

    @property
    def Jbp(self):
        assert not self.is_marginalized(), 'Jbp only defined for non-marginalized landmark blocks'
        return self.S[:self.m, :self.n_b + self.n_p]

    @Jbp.setter
    def Jbp(self, Jbp):
        assert not self.is_marginalized(), 'Jbp only defined for non-marginalized landmark blocks'
        self.S[:self.m, :self.n_b + self.n_p] = Jbp

    @property
    def QT_r(self):
        assert self.is_marginalized(), 'QT_r only defined for marginalized landmark blocks'
        return self.S[:, self.n:] if self.is_damped() else self.S[:-self.n_l, self.n:]

    @property
    def QT_Jbp(self):
        assert self.is_marginalized(), 'QT_Jbp only defined for marginalized landmark blocks'
        return self.S[:, :self.n_b + self.n_p] if self.is_damped() else self.S[:self.m, :self.n_b + self.n_p]

    @QT_Jbp.setter
    def QT_Jbp(self, QT_Jbp):
        assert self.is_marginalized(), 'QT_Jbp only defined for marginalized landmark blocks'
        if self.is_damped():
            self.S[:, :self.n_b + self.n_p] = QT_Jbp
        else:
            self.S[:self.m, :self.n_b + self.n_p] = QT_Jbp

    @property
    def R(self):
        assert self.is_marginalized(), 'R only defined for marginalized landmark blocks'
        return self.S[:, self.n_b + self.n_p:self.n] if self.is_damped() \
            else self.S[:self.m, self.n_b + self.n_p: self.n]

    @property
    def QT_Jl(self):
        assert self.is_marginalized(), 'QT_Jl only defined for marginalized landmark blocks'
        return self.R

    @property
    def R1(self):
        assert self.is_marginalized(), 'R1 only defined for marginalized landmark blocks'
        return self.S[:self.n_l, self.n_b + self.n_p: self.n]

    @property
    def Q1T_Jl(self):
        assert self.is_marginalized(), 'Q1T_Jl only defined for marginalized landmark blocks'
        return self.R1

    @property
    def Q1T_r(self):
        assert self.is_marginalized(), 'Q1T_r only defined for marginalized landmark blocks'
        assert self.S is not None, 'R only defined for marginalized landmark blocks'
        return self.S[:self.n_l, self.n:]

    @property
    def Q2T_r(self):
        assert self.is_marginalized(), 'Q2T_r only defined for marginalized landmark blocks'
        if self.S is not None:
            return self.S[self.n_l:, self.n:] if self.is_damped() else self.S[self.n_l:-self.n_l, self.n:]
        return self.QT_r

    @property
    def Q1T_Jbp(self):
        assert self.is_marginalized(), 'Q1T_Jbp only defined for marginalized landmark blocks'
        assert self.S is not None, 'R only defined for marginalized landmark blocks'
        return self.S[:self.n_l, :self.n_b + self.n_p]

    @property
    def Q2T_Jbp(self):
        assert self.is_marginalized(), 'Q2T_Jbp only defined for marginalized landmark blocks'
        return self.S[self.n_l:, :self.n_b + self.n_p] if self.is_damped() else self.S[self.n_l:self.m, :self.n_b + self.n_p]

    @property
    def n(self):
        return self.n_b + self.n_p + self.n_l


if NUMBA_LEVEL >= 3:
    ResidualBlockType = ResidualBlock.class_type.instance_type


class NonLMResidualBlock:
    def __init__(self, pose_idxs, r, Jb, Jp, psize=6):
        self.pose_idxs = pose_idxs
        self.r = r
        self.Jb = Jb
        self.Jp = Jp
        self.Jl_col_scale = None
        self.psize = psize

        self.m = r.size
        self.n_b = Jb.shape[1] if Jb is not None else 0
        self.n_p = Jp.shape[1] if Jp is not None else 0
        self.pnum = self.n_p // self.psize

    @property
    def n_l(self):
        return 0

    @property
    def n(self):
        return self.n_b + self.n_p

    def is_marginalized(self):
        return True

    def is_damped(self):
        return True

    def damp(self, _lambda):
        pass

    def undamp(self):
        pass

    @property
    def QT_r(self):
        return self.r

    @property
    def QT_Jbp(self):
        if self.Jb is not None and self.Jp is not None:
            return sp.hstack(self.Jb, self.Jp)
        if self.Jb is not None:
            return self.Jb
        return self.Jp

    @QT_Jbp.setter
    def QT_Jbp(self, QT_Jbp):
        assert sp.issparse(QT_Jbp), 'for dense matrices, manipulate in-place'
        if self.Jb is not None and self.Jp is not None:
            self.Jb = QT_Jbp[:, :self.n_b]
            self.Jp = QT_Jbp[:, self.n_b:]
        elif self.Jb is not None:
            self.Jb = QT_Jbp
        else:
            self.Jp = QT_Jbp

    @property
    def R(self):
        assert False, 'non-landmark block'

    @property
    def QT_Jl(self):
        assert False, 'non-landmark block'

    @property
    def R1(self):
        assert False, 'non-landmark block'

    @property
    def Q1T_Jl(self):
        assert False, 'non-landmark block'

    @property
    def Q1T_r(self):
        assert False, 'non-landmark block'

    @property
    def Q2T_r(self):
        return self.r

    @property
    def Q1T_Jbp(self):
        assert False, 'non-landmark block'

    @property
    def Q2T_Jbp(self):
        return self.QT_Jbp
