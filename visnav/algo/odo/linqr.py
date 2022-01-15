
import numpy as np
from scipy import sparse as sp
import numba as nb

from visnav.algo import tools


class InnerLinearizerQR:

    (
        STATE_INITIALIZED,
        STATE_LINEARIZED,
        STATE_MARGINALIZED,
        STATE_NUMERICAL_FAILURE,
    ) = range(4)

    def __init__(self, problem, jacobi_scaling_eps=1e-5, huber_coefs=None, use_weighted_residuals=False):
        self.problem = problem
        self.dtype = self.problem.dtype
        self.jacobi_scaling_eps = jacobi_scaling_eps

        huber_coefs = [None]*3 if huber_coefs is None else huber_coefs
        self.huber_coef_repr = huber_coefs[0]
        self.huber_coef_loc = huber_coefs[1]
        self.huber_coef_ori = huber_coefs[2]
        self.use_weighted_residuals = use_weighted_residuals

        self.use_valid_projections_only = True
        self._pose_damping = None

        self._blocks = None
        self._x_block = None
        self._a_block = None
        self._pose_size = self.problem.pose_size
        self._pose_n = len(self.problem.poses)
        self.nb = len(self.problem.cam_param_idxs)
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
        return self._blocks + self.non_lm_blocks

    def linearize(self):
        mr, mx, ma = self.problem.pts2d.size, self.problem.meas_r.size, self.problem.meas_aa.size
        m = mr + mx + ma

        rr, *rxra = self.problem.residual(parts=True)
        (Jrb, Jrp, Jrl), *JxJa = self.problem.jacobian(parts=True, fmt=('dense', 'csr', 'csr'))
        rx, ra, Jxb, Jxp, Jab, Jap = [None] * 6

        if self.use_valid_projections_only:
            numerically_valid = np.all(np.isfinite(Jrl.data if sp.issparse(Jrl) else Jrl)) \
                                and np.all(np.isfinite(Jrp.data if sp.issparse(Jrp) else Jrp)) \
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
            self._x_block = ResidualBlock(None, self.problem.meas_idxs, rx, Jxb, Jxp, None, psize=self._pose_size)
            self._total_error += err

        if ma > 0:
            Jab, Jap, _ = JxJa[0 if mx == 0 else 1]
            ra = rxra[0 if mx == 0 else 1]
            ra, (Jab, Jap), err = self._apply_huber(ra, (Jab, Jap), m, self.huber_coef_ori)
            self._a_block = ResidualBlock(None, self.problem.meas_idxs, ra, Jab, Jap, None, psize=self._pose_size)
            self._total_error += err

        safe = False    # safe is very slow, unsafe is dangerous as bugs won't be found
        index = lambda A, i, j: A[i, j] if safe else A._get_arrayXarray(i, j)

        # make landmark blocks
        self._blocks = []
        for i in range(len(self.problem.pts3d)):
            r_idxs = np.where(self.problem.pt3d_idxs == i)[0]
            pose_idxs = self.problem.pose_idxs[r_idxs]
            blk_np = len(r_idxs)

            if blk_np < 2:
                # too few observations of this 3d-point
                self.problem.valid_pts3d[i] = False
                continue

            # indexing sparse array faster this way instead of Jl[v_i, i:i+3]
            _, _, Jrl_i, Jrl_j = self._block_indexing(r_idxs, np.ones((len(r_idxs),)) * i, 2, 3)
            rr_i, _, Jrp_i, Jrp_j = self._block_indexing(r_idxs, pose_idxs, 2, self._pose_size)
            _, _, blk_Jp_i, blk_Jp_j = self._block_indexing(np.arange(blk_np), np.arange(blk_np), 2, self._pose_size)

            # create residual vector, assign reprojection residuals
            blk_r = np.zeros((blk_np*2, 1), dtype=self.dtype)
            blk_r[:blk_np*2] = rr[rr_i]

            # batch jacobian related to reprojection residuals
            blk_Jb = None if Jrb is None else np.array(Jrb[rr_i, :])    # dense matrix here

            # create landmark jacobian matrix, assign the three columns related to this landmark
            blk_Jl = np.zeros((blk_np*2, 3), dtype=self.dtype)
            blk_Jl[:blk_np*2, :] = np.array(index(Jrl, Jrl_i, Jrl_j).reshape((-1, 3)))

            # create pose/frame jacobian matrix, assign all frames that observe this landmark
            blk_Jp = np.zeros((blk_np*2, blk_np*self._pose_size), dtype=self.dtype)
            blk_Jp[blk_Jp_i, blk_Jp_j] = index(Jrp, Jrp_i, Jrp_j)

            self._blocks.append(ResidualBlock(r_idxs, pose_idxs, blk_r, blk_Jb, blk_Jp, blk_Jl, psize=self._pose_size))

        self._state = self.STATE_LINEARIZED

    @staticmethod
    def _block_indexing(row_idxs, col_idxs, block_rows, block_cols):
        row_idxs_r = InnerLinearizerQR._augment_idxs(row_idxs, block_rows)
        row_idxs_rc = np.repeat(row_idxs_r, block_cols)
        col_idxs_c = InnerLinearizerQR._augment_idxs(col_idxs, block_cols)
        col_idxs_cr = np.repeat(col_idxs_c.reshape((-1, block_cols)), block_rows, axis=0).flatten()
        return row_idxs_r, col_idxs_c, row_idxs_rc, col_idxs_cr

    @staticmethod
    def _augment_idxs(idxs, size):
        return (np.repeat(idxs[:, None] * size, size, axis=1) + np.arange(size)[None, :]).flatten()

    def _apply_huber(self, r, arr_J, total_nr, huber_coef):
        huber_coef = np.inf if huber_coef is None else huber_coef

        residual_weight = total_nr / len(r) if self.use_weighted_residuals else 1.0

        abs_r = np.abs(r)
        I = abs_r > huber_coef
        huber_weight = np.ones_like(r)
        huber_weight[I] *= np.sqrt(huber_coef / abs_r[I])
        dwr_dr = np.sqrt(residual_weight * huber_weight)

        wr = dwr_dr * r

        wJ = map(lambda J: None if J is None else (
            sp.diags(dwr_dr.flatten()).dot(J) if sp.issparse(J) else J * dwr_dr
        ), arr_J) if arr_J is not None else None

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
        for blk in self._blocks:
            blk.marginalize()
        self._state = self.STATE_MARGINALIZED

    def backsub_xl(self, delta_xbp):
        assert self._state == self.STATE_MARGINALIZED, 'not linearized yet'

        l_diff = 0
        delta_xb = delta_xbp[:self.nb]
        delta_xp = delta_xbp[self.nb:].reshape((-1, self._pose_size))
        delta_xl = np.zeros((len(self._blocks), self._blocks[0].nl))
        for i, blk in enumerate(self.all_blocks):
            blk_delta_xp = delta_xp[blk.pose_idxs, :].reshape((-1, 1))
            if blk.nl > 0:
                blk_delta_xbp = np.concatenate((delta_xb, blk_delta_xp), axis=0)
                blk_delta_xl = -np.linalg.solve(blk.R1, blk.Q1T_r + blk.Q1T_Jbp.dot(blk_delta_xbp))
                delta_xl[i, :] = blk_delta_xl.T * blk.Jl_col_scale
            else:
                blk_delta_xbp = blk_delta_xp

            # calculate linear cost diff
            blk.damp(0)
            QT_J_deltax = blk.QT_Jbp @ blk_delta_xbp
            if blk.nl > 0:
                QT_J_deltax[:blk.nl] += blk.R1.dot(blk_delta_xl)
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

        brb = np.zeros((self.nb, 1))
        bp = np.zeros((self._pose_n, self._pose_size))
        for blk in self.all_blocks:
            Q2T_Jbp = blk.Q2T_Jbp
            if sp.issparse(Q2T_Jbp):
                blk_b = (blk.Q2T_r.T @ Q2T_Jbp).T
            else:
                blk_b = (blk.Q2T_r.T.dot(Q2T_Jbp)).T
            if blk.nb > 0:
                brb[:self.nb] += blk_b[:blk.nb]
            bp[blk.pose_idxs, :] += blk_b[blk.nb:].reshape((-1, self._pose_size))

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

        db2 = np.zeros((self.nb,))
        dp2 = np.zeros((self._pose_n, self._pose_size))
        for blk in self.all_blocks:
            if blk.nb > 0:
                db2 += np.sum(self._epow2(blk.Jb), axis=0)
            dp2[blk.pose_idxs, :] += np.sum(self._epow2(blk.Jp), axis=0).reshape((-1, self._pose_size))

        return np.concatenate((db2, dp2.flatten()))

    def get_Q2TJbp_T_Q2TJbp_blockdiag(self):
        assert self._state == self.STATE_MARGINALIZED

        if self.dtype == np.float64:
            Hpp = self._get_Q2TJbp_T_Q2TJbp_blockdiag_f8(self.nb, self.np, self._pose_size, self._pose_damping,
                                                         nb.typed.List([blk.Q2T_Jbp for blk in self._blocks]),
                                                         nb.typed.List([blk.pose_idxs for blk in self._blocks]))

            for blk in self.non_lm_blocks:
                blk_Hpp = blk.Q2T_Jbp.T.dot(blk.Q2T_Jbp)
                tmp = np.arange(len(blk.pose_idxs))
                _, _, bi, bj = self._block_indexing(tmp, tmp, self._pose_size, self._pose_size)
                _, _, i, j = self._block_indexing(blk.pose_idxs, blk.pose_idxs, self._pose_size, self._pose_size)
                Hpp[i+self.nb, j+self.nb] += np.array(blk_Hpp[bi, bj]).flatten()

            return sp.csc_matrix(Hpp)

        assert len(self.problem.cam_param_idxs) == 0, 'not implemented for cam param estimation'

        # Note: invalid result & very slow if constructing with sp.lil_matrix,
        #       maybe block sparse would work ok and be faster than lil?
        Hpp = np.zeros((self.np, self.np), dtype=self.dtype)

        for blk in self.all_blocks:
            for i, idx in enumerate(blk.pose_idxs):
                g0, b0 = blk.pose_idxs[i] * self._pose_size, i * self._pose_size
                g1, b1 = g0 + self._pose_size, b0 + self._pose_size
                blk_Hpp = blk.Q2T_Jbp[:, b0:b1].T.dot(blk.Q2T_Jbp[:, b0:b1])
                Hpp[g0:g1, g0:g1] += blk_Hpp

        if self._pose_damping is not None:
            Hpp += self._pose_damping * np.eye(Hpp.shape[0], dtype=self.dtype)

        return sp.csc_matrix(Hpp)

    @staticmethod
    @nb.njit(nogil=True, parallel=False, cache=True)
    def _get_Q2TJbp_T_Q2TJbp_blockdiag_f8(_nb, _np, psize, pose_damping, arr_Q2T_Jbp, arr_pose_idxs):
        H = np.zeros((_nb + _np, _nb + _np), dtype=np.float64)
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
            H += pose_damping * np.eye(H.shape[0], dtype=np.float64)

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
            assert not blk.is_damped() or blk.nl == 0, 'apply scaling before damping'
            blk_rp = jac_rp_scaling[blk.pose_idxs, :].reshape((1, -1))
            blk_jsc = np.concatenate((jac_rb_scaling, blk_rp), axis=1) if blk.nl > 0 else blk_rp
            if sp.issparse(blk.QT_Jbp):
                blk.QT_Jbp = blk.QT_Jbp * sp.diags(blk_jsc.flatten())
            else:
                blk_QT_Jbp = blk.QT_Jbp  # direct blk.QT_Jbp inplace multiplication will fail because of @property decorator
                blk_QT_Jbp *= blk_jsc    # same as blk.QT_Jbp = blk.QT_Jbp.dot(np.diag(blk_jsc))

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
        return len(self.problem.cam_param_idxs)


class ResidualBlock:
    def __init__(self, r_idxs, pose_idxs, r, Jb, Jp, Jl, psize=6):
        self.r_idxs = r_idxs
        self.pose_idxs = pose_idxs

        # TODO: store directly into correct size storage so that no need np.concatenate, np.zeros at self.marginalize

        self.r = r
        self.Jb = Jb
        self.Jp = Jp
        self.Jl = Jl
        self.Jl_col_scale = None
        self.psize = psize

        self.nb = Jb.shape[1] if Jb is not None else 0
        self.np = Jp.shape[1] if Jp is not None else 0
        self.nl = Jl.shape[1] if Jl is not None else 0

        self.pnum = self.np // self.psize
        self.m = r.size

        # self._Q = None
        self._S = None
        self._marginalized = Jl is None
        self._damping_rots = None

    def marginalize(self):
        assert not self.is_marginalized(), 'already marginalized'

        if self.Jl is not None:
            Q, R = np.linalg.qr(self.Jl, 'complete')
            self._S = np.zeros((self.m + self.nl, self.nb + self.np + self.nl + 1), dtype=Q.dtype)
            Jbp = self.Jp if self.Jb is None else np.concatenate((self.Jb, self.Jp), axis=1)
            self._S[:self.m, :self.nb + self.np] = Q.T.dot(Jbp)
            self._S[:self.m, self.nb + self.np:-1] = R
            self._S[:self.m, -1:] = Q.T.dot(self.r)
            # self._S = np.concatenate((
            #         np.concatenate((Q.T.dot(self.Jp), R, Q.T.dot(self.r)), axis=1),
            #         np.zeros((self.nl, self.np + self.nl + 1), dtype=Q.dtype)), axis=0)
            del self.Jb, self.Jp, self.Jl, self.r

        self._marginalized = True

    def damp(self, _lambda):
        assert self.is_marginalized(), 'not yet marginalized'
        assert _lambda >= 0, 'lambda must be >= 0'

        if self.nl == 0:
            return

        if self.is_damped():
            self.undamp()

        if _lambda == 0:
            pass  # by default lambda already zero if no damping
        else:
            # make and apply givens rotations
            self._S[-self.nl:, self.nb + self.np:self.n] = np.eye(self.nl, dtype=self._S.dtype) * np.sqrt(_lambda)
            self._damping_rots = self._damp_f8(self._S, self.nb + self.np, self.nl)

    @staticmethod
    @nb.njit(nogil=True, parallel=False, cache=True)
    def _damp_f8(S, l_idx, k):
        damping_rots = nb.typed.List()
        for n in range(k):
            for m in range(n + 1):
                G = tools.make_givens(S[n, l_idx + n], S[-k + n - m, l_idx + n])
                tools.apply_givens(S, G, S.shape[0] - k + n - m, n)
                damping_rots.append(G)
        return damping_rots

    def undamp(self):
        assert self.is_damped(), 'not damped yet'
        self._undamp_f8(self._damping_rots, self._S, self.nl)
        self._damping_rots = None

    @staticmethod
    @nb.njit(nogil=True, parallel=False, cache=True)
    def _undamp_f8(damping_rots, S, k):
        for n in range(k-1, -1, -1):
            for m in range(n, -1, -1):
                G = damping_rots.pop()
                tools.apply_givens(S, G.T, S.shape[0] - k + n - m, n)

    def is_marginalized(self):
        return self._marginalized

    def is_damped(self):
        return self._damping_rots is not None or self.nl == 0

    @property
    def QT_r(self):
        if self._S is not None:
            return self._S[:, self.n:] if self.is_damped() else self._S[:-self.nl, self.n:]
        return self.r

    @property
    def QT_Jbp(self):
        if self._S is not None:
            return self._S[:, :self.nb + self.np] if self.is_damped() else self._S[:-self.nl, :self.nb + self.np]
        if self.Jb is not None and self.Jp is not None:
            return sp.hstack(self.Jb, self.Jp)
        if self.Jb is not None:
            return self.Jb
        return self.Jp

    @QT_Jbp.setter
    def QT_Jbp(self, QT_Jbp):
        assert sp.issparse(QT_Jbp) and self._S is None, 'for dense matrices, manipulate in-place'
        if self.Jb is not None and self.Jp is not None:
            self.Jb = QT_Jbp[:, :self.nb]
            self.Jp = QT_Jbp[:, self.nb:]
        elif self.Jb is not None:
            self.Jb = QT_Jbp
        else:
            self.Jp = QT_Jbp

    @property
    def R(self):
        assert self._S is not None, 'R only defined for marginalized landmark blocks'
        return self._S[:, self.nb + self.np:self.n] if self.is_damped() else self._S[:-self.nl, self.nb + self.np:self.n]

    @property
    def QT_Jl(self):
        return self.R

    @property
    def R1(self):
        assert self._S is not None, 'R1 only defined for marginalized landmark blocks'
        return self._S[:self.nl, self.nb + self.np:self.n]

    @property
    def Q1T_Jl(self):
        assert self._S is not None, 'R only defined for marginalized landmark blocks'
        return self.R1

    @property
    def Q1T_r(self):
        assert self.is_marginalized(), 'not marginalized yet'
        assert self._S is not None, 'R only defined for marginalized landmark blocks'
        return self._S[:self.nl, self.n:]

    @property
    def Q2T_r(self):
        assert self.is_marginalized(), 'not marginalized yet'
        if self._S is not None:
            return self._S[self.nl:, self.n:] if self.is_damped() else self._S[self.nl:-self.nl, self.n:]
        return self.QT_r

    @property
    def Q1T_Jbp(self):
        assert self.is_marginalized(), 'not marginalized yet'
        assert self._S is not None, 'R only defined for marginalized landmark blocks'
        return self._S[:self.nl, :self.nb + self.np]

    @property
    def Q2T_Jbp(self):
        # assert self.is_marginalized(), 'not marginalized yet'
        if self._S is not None:
            return self._S[self.nl:, :self.nb + self.np] if self.is_damped() else self._S[self.nl:-self.nl, :self.nb + self.np]
        return self.QT_Jbp

    @property
    def n(self):
        return self.nb + self.np + self.nl
