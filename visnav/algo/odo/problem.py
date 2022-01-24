from collections import namedtuple

import numpy as np
import quaternion
from scipy import sparse as sp

from visnav.algo import tools
from visnav.algo.tools import Manifold

from memory_profiler import profile
from visnav.algo.odo.linqr import mem_prof_logger


class Problem:
    IDX_DTYPE = np.int32

    @profile(stream=mem_prof_logger)
    def __init__(self, pts2d, batch_idxs, cam_params, cam_param_idxs, poses, pose_idxs, pts3d, pt3d_idxs, meas_r, meas_aa,
                 meas_idxs, px_err_sd, loc_err_sd, ori_err_sd, dtype):

        self.pts2d = pts2d.astype(dtype)

        self.batch_idxs = np.zeros((len(pts2d),), dtype=Problem.IDX_DTYPE) if batch_idxs is None else batch_idxs
        self.cam_params = np.atleast_2d(np.array(cam_params)).astype(dtype)
        self.cam_param_idxs = np.atleast_2d(cam_param_idxs).astype(Problem.IDX_DTYPE)

        self.poses = Manifold(poses.shape, buffer=poses.astype(dtype), dtype=dtype)
        self.poses.set_so3_groups(self._get_so3_grouping(len(poses)))
        self.pose_idxs = pose_idxs.astype(Problem.IDX_DTYPE)

        self.pts3d = pts3d.astype(dtype)
        self.pt3d_idxs = pt3d_idxs.astype(Problem.IDX_DTYPE)
        self.valid_pts3d = np.ones((len(self.pts3d),), dtype=bool)

        self.meas_r = np.array([], dtype=dtype) if meas_r is None else meas_r.astype(dtype)
        self.meas_aa = np.array([], dtype=dtype) if meas_aa is None else meas_aa.astype(dtype)
        self.meas_idxs = np.array([], dtype=Problem.IDX_DTYPE) if meas_idxs is None else meas_idxs.astype(Problem.IDX_DTYPE)

        self.px_err_sd = np.atleast_1d(np.array(px_err_sd, dtype=dtype))
        self.loc_err_sd = np.atleast_1d(np.array(loc_err_sd, dtype=dtype))
        self.ori_err_sd = np.atleast_1d(np.array(ori_err_sd, dtype=dtype))
        self.dtype = dtype

        self.pose_size = self.poses.shape[1]
        self.landmark_size = self.pts3d.shape[1]
        self.restrict_3d_point_y = False    # don't allow 3d point adjustment in y-direction
        self.cache = None
        self._J_cache = None
        self._cached_repr_err = None

    def _get_so3_grouping(self, n):
        return np.concatenate((np.ones((n, 3), dtype=bool), np.zeros((n, 3), dtype=bool)), axis=1)

    @property
    def xb(self):
        return np.array([cp[cpi] for cp, cpi in zip(self.cam_params, self.cam_param_idxs)], dtype=self.dtype).reshape((-1, 1))

    @xb.setter
    def xb(self, new_xb):
        if len(new_xb) > 0:
            offset = 0
            for i in range(self.cam_params.shape[0]):
                n = len(self.cam_param_idxs[i])
                self.cam_params[i][self.cam_param_idxs[i]] = new_xb.flatten()[offset: offset + n]
                if 0 in self.cam_param_idxs[i] and 1 not in self.cam_param_idxs[i]:
                    # if shared focal length, set fy = fx
                    self.cam_params[i][1] = self.cam_params[i][0]
                offset += n
        self.clear_cache()

    @property
    def xp(self):
        return self.poses.reshape((-1, 1))

    @xp.setter
    def xp(self, new_xp):
        assert type(new_xp) in (np.ndarray, Manifold), 'needs to be a numpy array or a Manifold'
        new_xp = new_xp.reshape((-1, self.pose_size))
        if type(new_xp) == np.ndarray:
            new_xp = Manifold(new_xp.shape, buffer=new_xp.astype(self.dtype), dtype=self.dtype)
            new_xp.set_so3_groups(self._get_so3_grouping(len(new_xp)))
        self.poses = new_xp
        self.clear_cache()

    @property
    def xl(self):
        return self.pts3d[self.valid_pts3d, :].reshape((-1, 1))

    @xl.setter
    def xl(self, new_xl):
        self.pts3d[self.valid_pts3d, :] = new_xl.reshape((-1, self.landmark_size))
        self.clear_cache()

    @property
    def x(self):
        return np.concatenate((self.xb, self.xp.to_array(), self.xl), axis=0)

    @x.setter
    def x(self, new_x):
        m1, m2 = len(self.xb), len(self.xp)
        self.xb = new_x[:m1]
        self.xp = new_x[m1:m1+m2]
        self.xl = new_x[m1+m2:]

    def filter(self, max_repr_err):
        # for now, just remove difficult to fit 3d point observations
        rr = self._cached_repr_err if self._cached_repr_err is not None else self.residual_repr()
        I = np.linalg.norm(rr.reshape((-1, 2)), axis=1) <= max_repr_err
        self.pt3d_idxs = self.pt3d_idxs[I]
        self.pose_idxs = self.pose_idxs[I]
        self.pts2d = self.pts2d[I, :]
        rr = rr.reshape((-1, 2))[I].reshape((-1, 1))
        self._cached_repr_err = rr
        self._J_cache = None
        self.cache = None
        return np.where(np.logical_not(I))[0]

    @profile(stream=mem_prof_logger)
    def residual(self, parts=False):
        self.maybe_populate_cache()
        errs = [self.residual_repr()]
        if self.meas_r is not None and len(self.meas_r) > 0:
            errs.append(self.residual_loc())
        if self.meas_aa is not None and len(self.meas_aa) > 0:
            errs.append(self.residual_ori())
        return errs if parts else np.concatenate(errs, axis=0)

    @profile(stream=mem_prof_logger)
    def jacobian(self, parts=False, fmt=('dense', 'csr', 'csr')):
        self.maybe_populate_cache()
        fmt_b, fmt_p, fmt_l = fmt

        Jrb = self.jacobian_repr_batch(fmt_b) if len(self.cam_param_idxs) > 0 else None
        Jrp = self.jacobian_repr_frame(fmt_p)
        Jrl = self.jacobian_repr_landmark(fmt_l)
        jacs = [(Jrb, Jrp, Jrl)]

        if self.meas_r is not None and len(self.meas_r) > 0:
            Jxb = self.jacobian_loc_batch(fmt_b, True)
            Jxp = self.jacobian_loc_frame(fmt_p, True)
            jacs.append((Jxb, Jxp, None))
        if self.meas_aa is not None and len(self.meas_aa) > 0:
            Jab = self.jacobian_ori_batch(fmt_b, True)
            Jap = self.jacobian_ori_frame(fmt_p, True)
            jacs.append((Jab, Jap, None))

        if parts:
            jacs_csr = [[c if not sp.issparse(c) else
                         getattr(c, 'to' + fmt[i])() for i, c in enumerate(r)] for r in jacs]
            return jacs_csr

        ms = [[(c.shape[0] if c is not None else 0) for c in r] for r in jacs]
        ns = [[(c.shape[1] if c is not None else 0) for c in r] for r in jacs]
        m = [0] + list(np.cumsum(np.max(ms, axis=1)))
        n = [0] + list(np.cumsum(np.max(ns, axis=0)))

        if not np.all([f == 'dense' for f in fmt]):
            J = sp.lil_matrix((m[-1], n[-1]), dtype=self.dtype)
        else:
            J = np.zeros((m[-1], n[-1]), dtype=self.dtype)

        for i, r in enumerate(jacs):
            for j, c in enumerate(r):
                if c is not None:
                    J[m[i]:m[i+1], n[j]:n[j+1]] = c

        return getattr(J, 'to'+fmt_l)() if sp.issparse(J) else J

    Cache = namedtuple('Cache', ('K', 'k1', 'k2', 'dist_coefs', 'pts3d_rot', 'pts3d_cf', 'pts3d_norm', 'iZciSD',
                                 'R2', 'alpha_xy', 'y_gamma_r', 'x_gamma_r', 'gamma_x', 'gamma_y'))

    def clear_cache(self):
        self.cache = None
        self._cached_repr_err = None

    @profile(stream=mem_prof_logger)
    def maybe_populate_cache(self):
        if self.cache is not None:
            return

        fx, fy, cx, cy, *dist_coefs = self.cam_params.T
        n_batches = len(fx)
        dist_coefs = np.array(dist_coefs).T
        K = np.zeros((n_batches, 3, 3), dtype=self.dtype)
        k1 = np.zeros((n_batches,), dtype=self.dtype)
        k2 = np.zeros((n_batches,), dtype=self.dtype)
        for i in range(n_batches):
            K[i, :, :] = np.array([[fx[i], 0, cx[i]],
                                   [0, fy[i], cy[i]],
                                   [0,  0,  1]])
            if len(dist_coefs) > i:
                if len(dist_coefs[i]) > 0:
                    k1[i] = dist_coefs[i][0]
                if len(dist_coefs[i]) > 1:
                    k2[i] = dist_coefs[i][1]

        pts3d_rot = tools.rotate_points_aa(self.pts3d[self.pt3d_idxs, :], self.poses[self.pose_idxs, 0:3].to_array())
        pts3d_cf = pts3d_rot + self.poses[self.pose_idxs, 3:6].to_array()

        iZc = 1 / pts3d_cf[:, 2]
        pts3d_norm = pts3d_cf[:, 0:2] * iZc[:, None]
        iZciSD = iZc / self.px_err_sd
        Xn, Yn = pts3d_norm[:, 0], pts3d_norm[:, 1]

        if np.count_nonzero(k1) + np.count_nonzero(k2) > 0:
            # observations are not undistorted, only k1 and k2 are supported
            # see method jacobian_repr_frame for details
            R2 = Xn ** 2 + Yn ** 2
            R2k1 = R2 * k1[self.batch_idxs]
            R4k2 = R2 ** 2 * k2[self.batch_idxs]
            alpha_xy = Xn * Yn * (4 * R2 * k2[self.batch_idxs] + 2 * k1[self.batch_idxs])
            y_gamma_r = Yn * (3 * R2k1 + 5 * R4k2 + 1)
            x_gamma_r = Xn * (3 * R2k1 + 5 * R4k2 + 1)
            gamma_x = R2k1 + R4k2 + Xn ** 2 * (4 * R2 * k2[self.batch_idxs] + 2 * k1[self.batch_idxs]) + 1
            gamma_y = R2k1 + R4k2 + Yn ** 2 * (4 * R2 * k2[self.batch_idxs] + 2 * k1[self.batch_idxs]) + 1
        else:
            R2, alpha_xy, y_gamma_r, x_gamma_r, gamma_x, gamma_y = [None] * 6

        self.cache = Problem.Cache(K=K, k1=k1, k2=k2, dist_coefs=dist_coefs, pts3d_rot=pts3d_rot, pts3d_cf=pts3d_cf,
                                   pts3d_norm=pts3d_norm, iZciSD=iZciSD, R2=R2, alpha_xy=alpha_xy, y_gamma_r=y_gamma_r,
                                   x_gamma_r=x_gamma_r, gamma_x=gamma_x, gamma_y=gamma_y)

    def residual_repr(self):
        """Convert 3-D points to 2-D by projecting onto images."""
        if self._cached_repr_err is not None:
            return self._cached_repr_err
        self.maybe_populate_cache()

        c = self.cache
        P = c.pts3d_norm

        if np.count_nonzero(c.k1) + np.count_nonzero(c.k2) > 0:
            P = P * (1 + c.k1[self.batch_idxs] * c.R2 + c.k2[self.batch_idxs] * c.R2 ** 2)[:, None]

        pts2d_proj = np.matmul(c.K[self.batch_idxs, :2, :2], P[:, :, None]).squeeze() + c.K[self.batch_idxs, :2, 2]
        repr_err = ((self.pts2d - pts2d_proj) / self.px_err_sd[:, None]).reshape((-1, 1))
        self._cached_repr_err = repr_err
        return repr_err

    def residual_loc(self):
        cam_rot_wf = -self.poses[self.meas_idxs, :3].to_array()
        cam_loc_wf = tools.rotate_points_aa(-self.poses[self.meas_idxs, 3:6].to_array(), cam_rot_wf)
        loc_err = ((self.meas_r - cam_loc_wf) / self.loc_err_sd).reshape((-1, 1))
        return loc_err

    def residual_ori(self):
        if 0:
            # TODO: rot_err_aa can be 2*pi shifted, what to do?
            rot_err_aa = tools.rotate_rotations_aa(meas_aa, -cam_rot_wf)
        else:
            Rm = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(self.meas_aa))
            Rw = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(self.poses[self.meas_idxs, :3].to_array()))
            E = np.matmul(Rm, Rw)
            rot_err_aa = tools.logR(E)
        ori_err = (rot_err_aa / ori_err_sd).reshape((-1, 1))
        return ori_err

    def _init_J(self, name, m, n, fmt):
        if self._J_cache is None:
            self._J_cache = {}
        if name not in self._J_cache:
            self._J_cache[name] = (np.zeros if fmt == 'dense' else sp.lil_matrix)((m, n), dtype=self.dtype)
        return self._J_cache[name]

    def jacobian_repr_batch(self, fmt):
        # distortion coefficients (D) affect reprojection error terms
        ####################################################
        # similar to above, first separate de/dD => de/dU' * dU'/dD
        # first one (de/dU') is -[[fx, 0],
        #                         [0, fy]]
        #
        # U'(Uc) = [[Xc/Zc], [Yc/Zc]] * (1 + k1*R2 + k2*R4), so:
        # second one (dU'/dD) is [[Xc/Zc*R2, Xc/Zc*R4],
        #                         [Yc/Zc*R2, Yc/Zc*R4]]
        #
        # total:
        # de/dU' = [[-fx*Xc/Zc*R**2, -fx*Xc/Zc*R**4],
        #           [-fy*Yc/Zc*R**2, -fy*Yc/Zc*R**4]]
        #

        c = self.cache
        m, n = self.pts2d.size, sum([len(cpi) for cpi in self.cam_param_idxs])
        J = self._init_J('_Jrb', m, n, fmt)
        i = np.arange(len(self.pts2d))

        fx, fy, cx, cy, *_ = self.cam_params.T
        Xn, Yn = c.pts3d_norm[:, 0], c.pts3d_norm[:, 1]

        # camera intrinsics (I=[fl, cx, cy]) affect reprojection error terms
        # [[e_u], [e_v]] = [[u - fl*Xn - cx],
        #                   [v - fl*Yn - cy]]
        # de/dI = [[-Xn, -1, 0],
        #          [-Yn, 0, -1]]
        j = 0
        for bi, cpi in enumerate(self.cam_param_idxs):
            I = self.batch_idxs == bi

            if {0, 1}.intersection(cpi):
                both = len({0, 1}.intersection(cpi)) == 2
                J[2 * i[I] + 0, j + 0] = -Xn[I]
                J[2 * i[I] + 1, j + (1 if both else 0)] = -Yn[I]
                j += 1 + (1 if both else 0)

            if 2 in cpi:
                J[2 * i[I] + 0, j] = -1
                j += 1

            if 3 in cpi:
                J[2 * i[I] + 1, j] = -1
                j += 1

            if {4, 5}.intersection(cpi):
                tmp_x = -fx[bi] * Xn[I] * c.R2[I]
                tmp_y = -fy[bi] * Yn[I] * c.R2[I]

            if 4 in cpi:
                J[2 * i[I] + 0, j] = tmp_x
                J[2 * i[I] + 1, j] = tmp_y
                j += 1

            if 5 in cpi:
                J[2 * i[I] + 0, j] = tmp_x * c.R2[I]
                J[2 * i[I] + 1, j] = tmp_y * c.R2[I]
                j += 1

        J /= self.px_err_sd[:, None]
        return J

    def jacobian_repr_frame(self, fmt):
        """
        cost function jacobian, mostly from https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
        """

        m, n = self.pts2d.size, self.poses.size
        J = self._init_J('_Jrf', m, n, fmt)
        i = np.arange(len(self.pts2d))

        # poses affect reprojection error terms
        ########################################
        #  - NOTE: for the following to work, need the poses in cam-to-world, i.e. rotation of world origin first,
        #    then translation in rotated coordinates
        #  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6891346/  equation (11)
        #  - also https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf, section 10.3.5, eq 10.23
        #  - U': points in distorted camera coords, Uc: points in undistorted camera coords, U: points in world coords
        #  - ksi: first three location, second three rotation
        #  - de/dksi = de/dU' * dU'/dUc * dUc/dksi
        #  - de/dU' = -[[fx, 0],
        #               [0, fy]]
        #
        #  Using SymPy diff on U'(Uc) = Matrix([[Xc/Zc], [Yc/Zc]]) * (1 + k1*R**2 + k2*R**4):
        #  - dU'/dUc = [[   (R2*k1 + R2**2*k2 + Xn**2*(4*R2*k2 + 2*k1) + 1)/Zc,
        #                   Xn*Yn*(4*R2*k2 + 2*k1)/Zc,
        #                   -Xn*(R2*k1 + R2**2*k2 + (Xn**2 + Yn**2)*(4*R2*k2 + 2*k1) + 1)/Zc
        #               ],
        #               [   Xn*Yn*(4*R2*k2 + 2*k1)/Zc,
        #                   (R2*k1 + R2**2*k2 + Yn**2*(4*R2*k2 + 2*k1) + 1)/Zc,
        #                   -Yn*(R2*k1 + R2**2*k2 + (Xn**2 + Yn**2)*(4*R2*k2 + 2*k1) + 1)/Zc
        #               ]],
        #               where R2 = Xn**2 + Yn**2, Xn=Xc/Zc, Yn=Yc/Zc
        #
        #  Alternatively, if k1 and k2 are zero because undistorted keypoint measures are used:
        #  - dU'/dUc = [[1/Zc, 0, -Xc/Zc**2],
        #               [0, 1/Zc, -Yc/Zc**2]]
        #
        #  - dUc/dw = [I3 | -[Uc]^] = [[1, 0, 0, 0, Zr, -Yr],
        #                              [0, 1, 0, -Zr, 0, Xr],
        #                              [0, 0, 1, Yr, -Xr, 0]]
        #
        #  if k1 and k2 are zero:
        #  - -[[fx/Zc, 0, -Xc*fx/Zc**2, | -Xc*Yr*fx/Zc**2, Xc*Xr*fx/Zc**2 + fx*Zr/Zc, -Yr*fx/Zc],
        #      [0, fy/Zc, -Yc*fy/Zc**2, | -Yc*Yr*fy/Zc**2 - fy*Zr/Zc, Xr*Yc*fy/Zc**2, Xr*fy/Zc]]    / px_err_sd
        #
        #  else:
        #  - [[-fx*(R2*k1 + R4*k2 + Xn**2*(4*R2*k2 + 2*k1) + 1)/Zc,
        #      -Xn*Yn*fx*(4*R2*k2 + 2*k1)/Zc,
        #       Xn*fx*(R2*k1 + R4*k2 + (Xn**2 + Yn**2)*(4*R2*k2 + 2*k1) + 1)/Zc,
        #        |
        #       Xn*fx*(Yn*Zr*(4*R2*k2 + 2*k1) + Yr*(R2*k1 + R4*k2 + (Xn**2 + Yn**2)*(4*R2*k2 + 2*k1) + 1))/Zc,
        #      -fx*(Xn*Xr*(R2*k1 + R4*k2 + (Xn**2 + Yn**2)*(4*R2*k2 + 2*k1) + 1) + Zr*(R2*k1 + R4*k2 + Xn**2*(4*R2*k2 + 2*k1) + 1))/Zc,
        #       fx*(-Xn*Xr*Yn*(4*R2*k2 + 2*k1) + Yr*(R2*k1 + R4*k2 + Xn**2*(4*R2*k2 + 2*k1) + 1))/Zc
        #     ],
        #     [-fy*Xn*Yn*(4*R2*k2 + 2*k1)/Zc,
        #      -fy*(R2*k1 + R4*k2 + Yn**2*(4*R2*k2 + 2*k1) + 1)/Zc,
        #       fy*Yn*(R2*k1 + R4*k2 + (Xn**2 + Yn**2)*(4*R2*k2 + 2*k1) + 1)/Zc,
        #       |
        #       fy*(Yn*Yr*(R2*k1 + R4*k2 + (Xn**2 + Yn**2)*(4*R2*k2 + 2*k1) + 1) + Zr*(R2*k1 + R4*k2 + Yn**2*(4*R2*k2 + 2*k1) + 1))/Zc,
        #      -Yn*fy*(Xn*Zr*(4*R2*k2 + 2*k1) + Xr*(R2*k1 + R4*k2 + (Xn**2 + Yn**2)*(4*R2*k2 + 2*k1) + 1))/Zc,
        #       fy*(Xn*Yn*Yr*(4*R2*k2 + 2*k1) - Xr*(R2*k1 + R4*k2 + Yn**2*(4*R2*k2 + 2*k1) + 1))/Zc
        #     ]]

        # TODO: debug y-px coord at J

        c = self.cache
        fx, fy, cx, cy, *_ = self.cam_params.T
        Xr, Yr, Zr = c.pts3d_rot[:, 0], c.pts3d_rot[:, 1], c.pts3d_rot[:, 2]
        Xn, Yn = c.pts3d_norm[:, 0], c.pts3d_norm[:, 1]
        fx, fy = fx[self.batch_idxs], fy[self.batch_idxs]

        if np.count_nonzero(c.k1) + np.count_nonzero(c.k2) == 0:
            # camera location
            J[2 * i + 0, self.pose_idxs * 6 + 3] = -fx * c.iZciSD
            J[2 * i + 0, self.pose_idxs * 6 + 5] = fx * Xn * c.iZciSD
            J[2 * i + 1, self.pose_idxs * 6 + 4] = -fy * c.iZciSD
            J[2 * i + 1, self.pose_idxs * 6 + 5] = fy * Yn * c.iZciSD

            # camera rotation
            J[2 * i + 0, self.pose_idxs * 6 + 0] = fx * Xn * Yr * c.iZciSD
            J[2 * i + 0, self.pose_idxs * 6 + 1] = -fx * (Xn * Xr + Zr) * c.iZciSD
            J[2 * i + 0, self.pose_idxs * 6 + 2] = fx * Yr * c.iZciSD
            J[2 * i + 1, self.pose_idxs * 6 + 0] = fy * (Yn * Yr + Zr) * c.iZciSD
            J[2 * i + 1, self.pose_idxs * 6 + 1] = -fy * Yn * Xr * c.iZciSD
            J[2 * i + 1, self.pose_idxs * 6 + 2] = -fy * Xr * c.iZciSD
        else:
            # camera location
            J[2 * i + 0, self.pose_idxs * 6 + 3] = -fx * c.iZciSD * c.gamma_x
            J[2 * i + 0, self.pose_idxs * 6 + 4] = -fx * c.iZciSD * c.alpha_xy
            J[2 * i + 0, self.pose_idxs * 6 + 5] = fx * c.iZciSD * c.x_gamma_r
            J[2 * i + 1, self.pose_idxs * 6 + 3] = -fy * c.iZciSD * c.alpha_xy
            J[2 * i + 1, self.pose_idxs * 6 + 4] = -fy * c.iZciSD * c.gamma_y
            J[2 * i + 1, self.pose_idxs * 6 + 5] = fy * c.iZciSD * c.y_gamma_r

            # camera rotation
            J[2 * i + 0, self.pose_idxs * 6 + 0] = fx * c.iZciSD * (Zr * c.alpha_xy + Yr * c.x_gamma_r)
            J[2 * i + 0, self.pose_idxs * 6 + 1] = -fx * c.iZciSD * (Xr * c.x_gamma_r + Zr * c.gamma_x)
            J[2 * i + 0, self.pose_idxs * 6 + 2] = -fx * c.iZciSD * (Xr * c.alpha_xy - Yr * c.gamma_x)
            J[2 * i + 1, self.pose_idxs * 6 + 0] = fy * c.iZciSD * (Yr * c.y_gamma_r + Zr * c.gamma_y)
            J[2 * i + 1, self.pose_idxs * 6 + 1] = -fy * c.iZciSD * (Zr * c.alpha_xy + Xr * c.y_gamma_r)
            J[2 * i + 1, self.pose_idxs * 6 + 2] = fy * c.iZciSD * (Yr * c.alpha_xy - Xr * c.gamma_y)

        # for debugging
        if 0:
            import matplotlib.pyplot as plt
            plt.figure(2)
            plt.imshow((J.toarray() != 0).astype(int))
            plt.show()

        return J

    def jacobian_repr_landmark(self, fmt):
        # keypoint locations affect reprojection error terms
        ####################################################
        # similar to above, first separate de/dU => de/dU' * dU'/dUc * dUc/dU,
        # first one (de/dU') is -[[fx, 0],
        #                         [0, fy]]
        # second one (dU'/dUc) is [[1/Zc, 0, -Xc/Zc**2],
        #                          [0, 1/Zc, -Yc/Zc**2]]  (or the monstrosity from above if k1 or k2 are non-zero)
        # third one  (dUc/dU => d/dU (RU + P) = R) is the camera rotation matrix R
        # => i.e. rotate (de/dU' * dU'/dUc) by R^-1

        c = self.cache
        m, n = self.pts2d.size, self.pts3d.size
        J = self._init_J('_Jrl', m, n, fmt)
        i = np.arange(len(self.pts2d))

        fx, fy, cx, cy, *_ = self.cam_params.T
        Xn, Yn = c.pts3d_norm[:, 0], c.pts3d_norm[:, 1]
        fx, fy = fx[self.batch_idxs], fy[self.batch_idxs]

        if np.count_nonzero(c.k1) + np.count_nonzero(c.k2) == 0:
            dEu = -np.stack((fx * np.ones((len(Xn),)), np.zeros((len(Xn),)), -fx * Xn), axis=1) \
                  * c.iZciSD[:, None]
            dEv = -np.stack((np.zeros((len(Xn),)), fy * np.ones((len(Xn),)), -fy * Yn), axis=1) \
                  * c.iZciSD[:, None]
        else:
            dEu = np.zeros((len(Xn), 3))
            dEv = np.zeros_like(dEu)
            dEu[:, 0] = -fx * c.iZciSD * c.gamma_x
            dEu[:, 1] = -fx * c.iZciSD * c.alpha_xy
            dEu[:, 2] = fx * c.iZciSD * c.x_gamma_r
            dEv[:, 0] = -fy * c.iZciSD * c.alpha_xy
            dEv[:, 1] = -fy * c.iZciSD * c.gamma_y
            dEv[:, 2] = fy * c.iZciSD * c.y_gamma_r

        dEuc = tools.rotate_points_aa(dEu, -self.poses[self.pose_idxs, :3].to_array())
        dEvc = tools.rotate_points_aa(dEv, -self.poses[self.pose_idxs, :3].to_array())
        J[2 * i + 0, self.pt3d_idxs * 3 + 0] = dEuc[:, 0]
        J[2 * i + 0, self.pt3d_idxs * 3 + 1] = 0 if self.restrict_3d_point_y else dEuc[:, 1]
        J[2 * i + 0, self.pt3d_idxs * 3 + 2] = dEuc[:, 2]
        J[2 * i + 1, self.pt3d_idxs * 3 + 0] = dEvc[:, 0]
        J[2 * i + 1, self.pt3d_idxs * 3 + 1] = 0 if self.restrict_3d_point_y else dEvc[:, 1]
        J[2 * i + 1, self.pt3d_idxs * 3 + 2] = dEvc[:, 2]

        # # time offsets affect reprojection error terms (possible to do in a better way?)
        # if ENABLE_DT_ADJ:
        #     p_offset = n1 + n2
        #     cam2meas = np.ones((np.max(pose_idxs)+1,), dtype=np.int) * -1
        #     cam2meas[meas_idxs] = np.arange(meas_idxs.size)
        #     i = np.where(cam2meas[pose_idxs] >= 0)[0]
        #     mc_idxs = cam2meas[pose_idxs[i]]
        #     A[2 * i, p_offset + mc_idxs] = 1
        #     A[2 * i + 1, p_offset + mc_idxs] = 1

        return J

    def jacobian_loc_batch(self, fmt, tight=True):
        # could use this to estimate the abs loc frame -> cam frame transformation
        return None

    def jacobian_loc_frame(self, fmt, tight=True):
        # frame loc affect loc measurement error terms (x~x, y~y, z~z only)
        ###################################################################
        # need to convert cam-to-world pose into world-cam location so that can compare
        #   err = ((Rm^-1 * -Pm) - (R^-1 * -P)) / loc_err_sd
        #   derr/dP = R^-1 / loc_err_sd
        #
        # https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf, section 10.3.6, eq 10.25:
        #
        #   derr/dR = -[[R[2,1]*pz - R[3,1]*py, ...], [...], [...]]
        #
        #  #### derr/dw = d/dw [R(w0)R(w)]^-1 * (-P/loc_err_sd) = d/dR(w0)R(w) [R(w0)R(w)]' * (-P/loc_err_sd)
        #  ###                                                 * d/dw R(w0)R(w)
        #  ###=> -[R(w0)R(w)]' * (-P/loc_err_sd) * R(w0) * np.vstack([-e1^, -e2^, -e3^])

        m, n = self.meas_r.size, (len(self.meas_idxs) if tight else len(self.poses)) * self.pose_size
        J = self._init_J('_Jlf', m, n, fmt)
        i = np.arange(len(self.meas_idxs))
        meas_idxs = i if tight else self.meas_idxs

        # cam locations affecting location measurement error
        iR = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(-self.poses[self.meas_idxs, :3]))
        iRs = iR / self.loc_err_sd
        for s in range(3):
            for r in range(3):
                J[3 * i + s, meas_idxs * 6 + 3 + r] = iRs[:, s, r]

        # cam orientations affecting location measurement error
        R = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(self.poses[self.meas_idxs, :3]))
        Rs = R / self.loc_err_sd
        for s in range(3):
            for r in range(3):
                a, b = [1, 0, 0][r], [2, 2, 1][r]
                sign = -1 if r == 1 else 1
                J[3 * i + s, meas_idxs * 6 + r] = sign * (Rs[:, a, s] * self.poses[self.meas_idxs, 3 + b].to_array()
                                                          - Rs[:, b, s] * self.poses[self.meas_idxs, 3 + a].to_array())
        return J

    def jacobian_ori_batch(self, fmt, tight=True):
        # could use this to estimate the abs ori frame -> cam frame transformation
        return None

    def jacobian_ori_frame(self, fmt, tight=True):
        # orientation components affect ori measurement error terms
        ###########################################################
        # dw_e = log(Rm*exp(w)*R(w0)) / ori_err_sd
        # based on https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
        # by chain rule:
        # dw_e/dw = 1/ori_err_sd * d/dw log(Rm*exp(w)*R(w0))
        #         = 1/ori_err_sd
        #           * d/d(Rm*exp(w)*R(w0)) log(Rm*exp(w)*R(w0))
        #           * d/dw Rm*exp(w)*R(w0)
        # 1) d/dR log(R) = <see the source, long expression>  (3x9, section 10.3.2, eq 10.11)
        # 2) d/dw Rm*exp(w)*R(w0) = [-Rm*dc1^, -Rm*dc2^, -Rm*dc3^].T  (9x3, section 10.3.7, eq 10.28)

        m, n = self.meas_aa.size, (len(self.meas_idxs) if tight else len(self.poses)) * self.pose_size
        J = self._init_J('_Jof', m, n, fmt)
        meas_idxs = np.arange(len(self.meas_idxs), dtype=Problem.IDX_DTYPE) if tight else self.meas_idxs

        Rm = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(self.meas_aa))
        Rw0 = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(self.poses[self.meas_idxs, :3]))
        for u in range(len(self.meas_idxs)):
            D1 = tools.dlogR_dR(Rm[u].dot(Rw0[u]))
            D2 = np.vstack((-Rm[u].dot(tools.wedge(Rw0[u, :, 0])),
                            -Rm[u].dot(tools.wedge(Rw0[u, :, 1])),
                            -Rm[u].dot(tools.wedge(Rw0[u, :, 2]))))
            e_off = 3 * u
            p_off = meas_idxs[u] * 6
            J[e_off:e_off + 3, p_off:p_off + 3] = D1.dot(D2) / self.ori_err_sd

        # if n4 > 0:
        #     # measurement rotation offset affects all measurement error terms
        #     d, p_offset = 0, n1 + n2 + n3
        #     if n4 > 4:
        #         d = 3
        #         for s in range(6):
        #             A[m1 + 6 * i + s, p_offset:p_offset + 3] = 1
        #
        #     # measurement location offset affects loc measurement error terms
        #     for s in range(3):
        #         A[m1 + 6 * i + s, p_offset + d + s] = 1
        #
        #     # measurement scale offset affects loc measurement error terms
        #     for s in range(3):
        #         A[m1 + 6 * i + s, p_offset + d + 3] = 1

        return J

    def plot_current_r_J(self):
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(self.residual())
        self.plot_J(self.jacobian(), 2)

    @staticmethod
    def plot_J(J, fig_n=None, show=True):
        import matplotlib.pyplot as plt
        plt.figure(fig_n)
        J = J.toarray() if sp.issparse(J) else J
        plt.imshow(np.sign(J) * np.abs(J) ** (1 / 8))
        if show:
            plt.show()
