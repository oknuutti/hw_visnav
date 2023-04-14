"""
Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""
import math
import sys
import logging
from functools import lru_cache
import warnings

import numpy as np
import quaternion

import scipy.sparse as sp
import scipy.linalg as spl
from scipy.optimize import least_squares

from visnav.algo import tools
from visnav.algo.tools import Manifold, LogWriter

GLOBAL_ADJ = 0      # 3: all incl rotation, 2: location and scale, 1: scale only
FIXED_PITCH_AND_ROLL = 0
ANALYTICAL_JACOBIAN = 1
CHECK_JACOBIAN = 0
ENABLE_DT_ADJ = 0
DT_ADJ_MEAS_VEL = 1
USE_WORLD_CAM_FRAME = 0
USE_OWN_PSEUDO_HUBER_LOSS = 1
RESTRICT_3D_POINT_Y = False


def vis_gps_bundle_adj(poses: np.ndarray, pts3d: np.ndarray, pts2d: np.ndarray, v_pts2d: np.ndarray,
                       cam_idxs: np.ndarray, pt3d_idxs: np.ndarray, K: np.ndarray, dist_coefs: np.ndarray,
                       px_err_sd: np.ndarray, meas_r: np.ndarray, meas_aa: np.ndarray, t_off: np.ndarray,
                       meas_idxs: np.ndarray, loc_err_sd: float = np.inf, ori_err_sd: float = np.inf,
                       prior_r: np.ndarray = None, prior_J: np.ndarray = None, prior_x: np.ndarray = None,
                       prior_k: np.ndarray = None, prior_cam_idxs: np.ndarray = None, prior_weight = 1.0,
                       marginalize_pose_idxs: np.ndarray = None, marginalize_pt3d_idxs: np.ndarray = None,
                       dtype=np.float64, px_err_weight=1,
                       n_cam_intr=0, log_writer=None, max_nfev=None, skip_pose_n=1, poses_only=False, huber_coef=False,
                       weighted_residuals=True, just_return_r_J=False):
    """
    Returns the bundle adjusted parameters, in this case the optimized rotation and translation vectors.

    basepose with shape (6,) is the pose of the first frame that is used as an anchor

    poses with shape (n_cameras, 6) contains initial estimates of parameters for all cameras.
            First 3 components in each row form a rotation vector,
            next 3 components form a translation vector

    pts3d with shape (n_points, 3)
            contains initial estimates of point coordinates in the world frame.

    pts2d with shape (n_observations, 2)
            contains measured 2-D coordinates of points projected on images in each observations.

    pose_idxs with shape (n_observations,)
            contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.

    pt3d_idxs with shape (n_observations,)
            contains indices of points (from 0 to n_points - 1) involved in each observation.

    """
    assert len(poses.shape) == 2 and poses.shape[1] == 6, 'wrong shape poses: %s' % (poses.shape,)
    assert len(pts3d.shape) == 2 and pts3d.shape[1] == 3, 'wrong shape pts3d: %s' % (pts3d.shape,)
    assert len(pts2d.shape) == 2 and pts2d.shape[1] == 2, 'wrong shape pts2d: %s' % (pts2d.shape,)
    assert len(cam_idxs.shape) == 1, 'wrong shape pose_idxs: %s' % (cam_idxs.shape,)
    assert len(pt3d_idxs.shape) == 1, 'wrong shape pt3d_idxs: %s' % (pt3d_idxs.shape,)
    assert K.shape == (3, 3), 'wrong shape K: %s' % (K.shape,)
    assert marginalize_pose_idxs is None or skip_pose_n == 0, 'cant skip poses if marginalizing'

    #assert not skip_pose0, 'some bug with skipping first pose optimization => for some reason cost stays high, maybe problem with A?'

    n_cams = poses.shape[0]
    n_pts = pts3d.shape[0]
    n_dist = 0
    if dist_coefs is not None:
        n_dist = np.where(np.array(dist_coefs) != 0)[0][-1] + 1

    A = _bundle_adjustment_sparsity(n_cams, n_pts, n_dist, n_cam_intr, cam_idxs, pt3d_idxs, meas_idxs, poses_only)   # n_cams-a or n_cams?
    if skip_pose_n > 0:
        A = A[:, skip_pose_n * 6:]

    # poses[:, :3] = poses[:, :3] / np.pi * 180

    pose0 = poses[0:skip_pose_n].ravel()
    x0 = [poses[skip_pose_n:].ravel()]

    if ENABLE_DT_ADJ:
        x0.append(t_off.ravel())

    if poses_only:
        fixed_pt3d = pts3d
    else:
        fixed_pt3d = np.zeros((0, 3))
        x0.append(pts3d.ravel())

    if len(meas_idxs) > 1 and GLOBAL_ADJ:
        x0.append(np.array([0] * (7 if GLOBAL_ADJ > 2 else 4)))

    if n_dist > 0:
        x0.append(dist_coefs[:n_dist])

    if n_cam_intr > 0:
        x0.append(([(K[0, 0] + K[1, 1])/2] if n_cam_intr != 2 else [])
                  + ([K[0, 2], K[1, 2]] if n_cam_intr > 1 else []))

    x0 = np.hstack(x0)
    I = np.concatenate((
        np.stack((np.ones((len(poses) - skip_pose_n, 3), dtype=bool),
                  np.zeros((len(poses) - skip_pose_n, 3), dtype=bool)), axis=1).ravel(),
        np.zeros((len(x0) - (len(poses) - skip_pose_n) * 6,), dtype=bool)
    ))
    x0 = Manifold(x0.shape, buffer=x0.astype(dtype), dtype=dtype)
    x0.set_so3_groups(I)

    m1, m2, m3 = cam_idxs.size * 2, meas_idxs.size * 3, meas_idxs.size * 3
    m = m1 + m2 + m3
    if isinstance(huber_coef, (tuple, list)):
        huber_coef = np.array([huber_coef[0]] * m1 + [huber_coef[1]] * m2 + [huber_coef[2]] * m3, dtype=dtype)
    weight = 1 if not weighted_residuals else np.array([px_err_weight * m / m1] * m1
                                                      + [(m / m2) if m2 else 0] * m2
                                                      + [(m / m3) if m3 else 0] * m3, dtype=dtype)

    orig_dtype = pts2d.dtype
    if dtype != orig_dtype:
        pose0, fixed_pt3d, pts2d, v_pts2d, K, px_err_sd, meas_r, meas_aa = map(
            lambda x: x.astype(dtype), (pose0, fixed_pt3d, pts2d, v_pts2d, K, px_err_sd, meas_r, meas_aa))

    curr_prior_J, prior_x_idxs = None, None
    if prior_J is not None and len(prior_r) > 0:
        a = n_cams * 6
        b = a + len(meas_idxs) * (1 if ENABLE_DT_ADJ else 0)
        c = b + n_pts * 3
        d = c + (7 if GLOBAL_ADJ > 2 else 4 if GLOBAL_ADJ else 0)
        e = d + n_dist
        f = e + n_cam_intr

        prior_x_idxs = np.zeros((len(x0),), dtype=bool)
        pa = 6 * len(prior_cam_idxs)
        for i, j in enumerate(prior_cam_idxs):
            prior_x_idxs[i*6: (i+1)*6] = prior_k[j*6: (j+1)*6]
        if ENABLE_DT_ADJ:
            mi = np.where(np.in1d(meas_idxs, prior_cam_idxs))[0]    # TODO: debug, probably wrong
            prior_x_idxs[a + mi] = prior_k[pa: pa + len(mi)]
            pa += len(mi)
        if f - c > 0:
            prior_x_idxs[-(f - c):] = prior_k[-(f - c):]

        curr_prior_J = sp.lil_matrix((len(prior_r), len(x0)), dtype=dtype)
        curr_prior_J[:, prior_x_idxs] = prior_J * prior_weight
        curr_prior_J = curr_prior_J.tocsr()

    @lru_cache(1)
    def _cfun(x):
        return _costfun(x, pose0, fixed_pt3d, n_cams, n_pts, n_dist, n_cam_intr, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K,
                        px_err_sd, meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef)

    def cfun(x, use_own_pseudo_huber_loss=USE_OWN_PSEUDO_HUBER_LOSS):
        err = _cfun(tuple(x))
        if use_own_pseudo_huber_loss:
            sqrt_phl_err = np.sqrt(weight * tools.pseudo_huber_loss(err, huber_coef)) if huber_coef is not False else err
        else:
            sqrt_phl_err = err

        if prior_x_idxs is not None:
            prior_r_current = (prior_r + prior_J.dot(x[prior_x_idxs] - prior_x)) * prior_weight
            sqrt_phl_err = np.concatenate((sqrt_phl_err, prior_r_current), axis=0)

        return sqrt_phl_err

    def jac(x, use_own_pseudo_huber_loss=USE_OWN_PSEUDO_HUBER_LOSS):
        # apply pseudo huber loss derivative
        J = _jacobian(x, pose0, fixed_pt3d, n_cams, n_pts, n_dist, n_cam_intr, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K,
                      px_err_sd, meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef)

        if use_own_pseudo_huber_loss:
            ## derivative of pseudo huber loss can be applied afterwards by multiplying with the err derivative
            # NOTE: need extra (.)**0.5 term as least_squares applies extra (.)**2 on top of our cost function
            # - d (weight * phl(e))**0.5/dksi = d (weight * phl(e))**0.5/d phl(e) * d phl(e)/de * de/dksi
            # - d (weight * phl(e))**0.5/d phl(e) = weight**0.5  * 0.5 * phl(e)**(-0.5)
            # - d phl(e)/de = e/sqrt(1+e**2/delta**2)
            # - de/dksi is solved next
            err = _cfun(tuple(x))
            phl_err = tools.pseudo_huber_loss(err, huber_coef)
            dhuber = 0.5 * np.sqrt(weight / phl_err) * err / np.sqrt(1 + err ** 2 / huber_coef ** 2)
            J = J.multiply(dhuber.reshape((-1, 1)))

        if curr_prior_J is not None:
            J = sp.vstack((J, curr_prior_J))

        return J

    if CHECK_JACOBIAN and n_cam_intr > 0: #  and n_dist > 0:
        if 1:
            J = jac(x0, False).toarray()
            J_ = numerical_jacobian(lambda x: _costfun(x, pose0, fixed_pt3d, n_cams, n_pts, n_dist, n_cam_intr, cam_idxs, pt3d_idxs,
                                                         pts2d, v_pts2d, K, px_err_sd, meas_r, meas_aa,
                                                         meas_idxs, loc_err_sd, ori_err_sd, huber_coef), x0, 1e-4)
            if curr_prior_J is not None:
                J_ = np.vstack((J_, curr_prior_J.toarray() if sp.issparse(curr_prior_J) else curr_prior_J))
        else:
            J = _rot_pt_jac(x0, pose0, fixed_pt3d, n_cams, n_pts, n_dist, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K, px_err_sd,
                            meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef)
            J_ = numerical_jacobian(lambda x: _rotated_points(x, pose0, fixed_pt3d, n_cams, n_pts, n_dist, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K, px_err_sd,
                                      meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef), x0, 1e-4)

        if 1:
            J = J[:100, -100:]
            J_ = J_[:100, -100:]

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        axs[0].imshow(np.sign(J) * np.abs(J) ** (1/8))
        axs[0].set_title('analytical')
        axs[1].imshow(np.sign(J_) * np.abs(J_) ** (1/8))
        axs[1].set_title('numerical')
        plt.show()

    if just_return_r_J:
        return cfun(x0, True), jac(x0, True)

    # print('initial mean residual: %.5f' % (np.mean(cfun(x0)[:-m3]),))

    tmp = sys.stdout
    sys.stdout = log_writer or LogWriter()
    res = least_squares(cfun, x0, verbose=2, ftol=1e-5, xtol=1e-5, gtol=1e-8, method='trf',
                        jac_sparsity=A if not ANALYTICAL_JACOBIAN else None,
                        x_scale='jac', jac='2-point' if not ANALYTICAL_JACOBIAN else jac,
                        # for some reason doesnt work as well as own huber loss
                        # tr_solver='lsmr',
                        loss='linear' if USE_OWN_PSEUDO_HUBER_LOSS else 'huber',  # f_scale=1.0,  #huber_coef,
                        # args=(pose0, fixed_pt3d, n_cams, n_pts, pose_idxs, pt3d_idxs, pts2d, v_pts2d, K, px_err_sd,
                        #       meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef),
                        max_nfev=max_nfev)
    sys.stdout = tmp

    assert isinstance(res.x, Manifold), 'somehow manifold lost during optimization'
    new_poses, new_pts3d, new_dist, new_cam_intr, new_t_off = _unpack(res.x.to_array(), n_cams - skip_pose_n,
                                                        0 if poses_only else n_pts, n_dist, n_cam_intr, meas_idxs.size,
                                                        meas_r0=None if len(meas_idxs) < 2 else meas_r[0])

    new_prior, prior_len = (None,) * 4, 0 if prior_r is None else len(prior_r)
    if marginalize_pose_idxs is not None and (len(marginalize_pose_idxs) + len(marginalize_pt3d_idxs) + prior_len > 0):
        new_prior = marginalize(res.x, res.fun, res.jac, marginalize_pose_idxs, marginalize_pt3d_idxs,
                                n_cams, n_pts, n_dist, n_cam_intr, meas_idxs, prior_len, prior_weight)

    if prior_r is not None and len(prior_r) > 0:
        res.fun = res.fun[:-len(prior_r)]

    # print('finished mean residual: %.5f' % (np.mean(res.fun[:-m3]),))
    # return also per frame errors (median repr err, meas loc and ori errs)
    # so that can follow the errors and notice/debug problems
    res.fun = res.fun / np.sqrt(weight)
    repr_err = np.linalg.norm(res.fun[:cam_idxs.size * 2].reshape((-1, 2)), axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        errs = np.array([[np.median(repr_err[cam_idxs == i])] for i in range(n_cams)])

    if len(meas_idxs) > 0:
        loc_err = res.fun[-6*len(meas_idxs):-3*len(meas_idxs)].reshape((-1, 3))
        ori_err = res.fun[-3*len(meas_idxs):].reshape((-1, 3))
        errs = np.concatenate((errs, np.ones((len(errs), 6), dtype=errs.dtype) * np.nan), axis=1)
        errs[meas_idxs, 1:] = np.concatenate((loc_err, ori_err), axis=1)

    if new_poses.dtype != orig_dtype:
        new_poses = new_poses.astype(orig_dtype)
        new_pts3d = new_pts3d.astype(orig_dtype)
        new_t_off = new_t_off.astype(orig_dtype)
        errs = errs.astype(orig_dtype)

    return new_poses, new_pts3d, new_dist, new_cam_intr, new_t_off, new_prior, errs


def _costfun(params, pose0, fixed_pt3d, n_cams, n_pts, n_dist, n_cam_intr, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K, px_err_sd,
             meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef):
    """
    Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    if isinstance(params, (tuple, list)):
        params = np.array(params)

    params = np.hstack((pose0, params))
    poses, pts3d, dist_coefs, cam_intr, t_off = _unpack(params, n_cams, n_pts if len(fixed_pt3d) == 0 else 0, n_dist,
                                                        n_cam_intr, meas_idxs.size,
                                                        meas_r0=None if len(meas_idxs) < 2 else meas_r[0])

    if cam_intr is not None:
        if len(cam_intr) != 2:
            K[0, 0] = K[1, 1] = cam_intr[0]
        if len(cam_intr) > 1:
            K[0, 2], K[1, 2] = cam_intr[-2:]

    points_3d = fixed_pt3d if len(pts3d) == 0 else pts3d
    points_proj = _project(points_3d[pt3d_idxs], poses[cam_idxs], K, dist_coefs)

    if t_off is not None and len(t_off) > 0 and not DT_ADJ_MEAS_VEL:
        d_pts2d = v_pts2d * t_off[cam_idxs]  # for 1st order approximation of where the 2d points would when meas_r recorded
    else:
        d_pts2d = 0

    # # global rotation, translation and scale, applied to measurements
    # if len(meas_idxs) > 1 and GLOBAL_ADJ:
    #     rot_off = np.array([[0, 0, 0]]) if GLOBAL_ADJ < 3 else params[d-7:d-4].reshape((1, 3))
    #     loc_off = params[d-4:d-1]
    #     scale_off = params[d-1]
    #
    #     meas_r = tools.rotate_points_aa(meas_r - meas_r[0], rot_off) * math.exp(scale_off) + meas_r[0] + loc_off
    #     meas_aa = tools.rotate_rotations_aa(np.repeat(rot_off, len(meas_aa), axis=0), meas_aa)

    px_err = (((pts2d + d_pts2d) - points_proj) / px_err_sd[:, None]).ravel()

    if USE_WORLD_CAM_FRAME:
        loc_err = ((meas_r - poses[meas_idxs, 3:]) / loc_err_sd).ravel()
        rot_err_aa = tools.rotate_rotations_aa(meas_aa, -poses[meas_idxs, :3])
    else:
        if t_off is not None and len(t_off) > 0 and DT_ADJ_MEAS_VEL:
            d_meas_r = v_pts2d * t_off
        else:
            d_meas_r = 0

        cam_rot_wf = -poses[meas_idxs, :3]
        cam_loc_wf = tools.rotate_points_aa(-poses[meas_idxs, 3:6], cam_rot_wf)
        loc_err = ((meas_r + d_meas_r - cam_loc_wf) / loc_err_sd).ravel()
        if 0:
            # TODO: rot_err_aa can be 2*pi shifted, what to do?
            rot_err_aa = tools.rotate_rotations_aa(meas_aa, -cam_rot_wf)
        else:
            Rm = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(meas_aa))
            Rw = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(poses[meas_idxs, :3]))
            E = np.matmul(Rm, Rw)
            rot_err_aa = tools.logR(E)

    ori_err = (rot_err_aa / ori_err_sd).ravel()

    loc_err = loc_err if 1 else np.zeros((len(meas_idxs)*3,))
    ori_err = ori_err if 1 else np.zeros((len(meas_idxs)*3,))

    err = np.concatenate((px_err, loc_err, ori_err))
    return err


def _bundle_adjustment_sparsity(n_cams, n_pts, n_dist, n_cam_intr, cam_idxs, pt3d_idxs, meas_idxs, poses_only):
    # error term count  (first 2d reprojection errors, then 3d gps measurement error, then 3d orientation meas err)
    m1, m2, m3 = cam_idxs.size * 2, meas_idxs.size * 3, meas_idxs.size * 3
    m = m1 + m2 + m3

    # TODO: take into account n_dist, n_cam_intr

    # parameter count (6d poses, np x 3d keypoint locations, 1d time offset)
    n1 = n_cams * 6
    n2 = (0 if poses_only else n_pts * 3)
    n3 = meas_idxs.size * (1 if ENABLE_DT_ADJ else 0)
    n4 = 0 if meas_idxs.size < 2 or not GLOBAL_ADJ else (7 if GLOBAL_ADJ > 2 else 4)
    n = n1 + n2 + n3 + n4

    A = sp.lil_matrix((m, n), dtype=int)
    i = np.arange(cam_idxs.size)

    # poses affect reprojection error terms
    for s in range(6):
        A[2 * i, cam_idxs * 6 + s] = 1
        A[2 * i + 1, cam_idxs * 6 + s] = 1

    # keypoint locations affect reprojection error terms
    if not poses_only:
        p_offset = n1
        for s in range(3):
            A[2 * i, p_offset + pt3d_idxs * 3 + s] = 1
            A[2 * i + 1, p_offset + pt3d_idxs * 3 + s] = 1

    # time offsets affect reprojection error terms (possible to do in a better way?)
    if ENABLE_DT_ADJ:
        p_offset = n1 + n2
        cam2meas = np.ones((np.max(cam_idxs)+1,), dtype=np.int) * -1
        cam2meas[meas_idxs] = np.arange(meas_idxs.size)
        i = np.where(cam2meas[cam_idxs] >= 0)[0]
        mc_idxs = cam2meas[cam_idxs[i]]
        A[2 * i, p_offset + mc_idxs] = 1
        A[2 * i + 1, p_offset + mc_idxs] = 1

    # frame loc affect loc measurement error terms
    i = np.arange(meas_idxs.size)
    if USE_WORLD_CAM_FRAME:
        # (x~x, y~y, z~z only)
        for s in range(3):
            A[m1 + 6 * i + s, meas_idxs * 6 + 3 + s] = 1
    else:
        # all
        for s in range(3):
            for r in range(3):
                A[m1 + 6 * i + s, meas_idxs * 6 + 3 + r] = 1

    # orientation components affect ori measurement error terms
    # for s in range(3):
    #     A[m1 + m2 + 3 * i + s, meas_idxs * 6 + s] = 1
    for s in range(3):
        for r in range(3):
            A[m1 + 3 + 6 * i + s, meas_idxs * 6 + r] = 1

    if n4 > 0:
        # measurement rotation offset affects all measurement error terms
        d, p_offset = 0, n1 + n2 + n3
        if n4 > 4:
            d = 3
            for s in range(6):
                A[m1 + 6 * i + s, p_offset:p_offset + 3] = 1

        # measurement location offset affects loc measurement error terms
        for s in range(3):
            A[m1 + 6 * i + s, p_offset + d + s] = 1

        # measurement scale offset affects loc measurement error terms
        for s in range(3):
            A[m1 + 6 * i + s, p_offset + d + 3] = 1

    if 0:
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.imshow(A.toarray())
        plt.show()

    return A


def _jacobian(params, pose0, fixed_pt3d, n_cams, n_pts, n_dist, n_cam_intr, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K,
              px_err_sd, meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef):
    """
    cost function jacobian, from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6891346/
    """
    assert not GLOBAL_ADJ and (not ENABLE_DT_ADJ or DT_ADJ_MEAS_VEL), \
        'analytical jacobian does not support GLOBAL_ADJ or ENABLE_DT_ADJ with pixel velocity'

    params = np.hstack((pose0, params))
    poses, pts3d, dist_coefs, cam_intr, t_off = _unpack(params, n_cams, n_pts if len(fixed_pt3d) == 0 else 0,
                                                        n_dist, n_cam_intr, meas_idxs.size,
                                                        meas_r0=None if len(meas_idxs) < 2 else meas_r[0])

    poses_only = len(fixed_pt3d) > 0
    points_3d = fixed_pt3d if poses_only else pts3d
    points_3d_rot = tools.rotate_points_aa(points_3d[pt3d_idxs], poses[cam_idxs, :3])
    points_3d_cf = points_3d_rot + poses[cam_idxs, 3:6]

    # error term count  (first 2d reprojection errors, then 3d gps measurement error, then 3d orientation meas err)
    m1, m2, m3 = cam_idxs.size * 2, meas_idxs.size * 3, meas_idxs.size * 3
    m = m1 + m2 + m3

    # parameter count (6d poses, np x 3d keypoint locations, 1d time offset)
    n1 = n_cams * 6
    n2 = meas_idxs.size * (1 if ENABLE_DT_ADJ else 0)
    n3 = (0 if poses_only else n_pts * 3)
    n4 = (0 if dist_coefs is None else 2)
    n5 = (0 if cam_intr is None else len(cam_intr))
    n = n1 + n2 + n3 + n4 + n5
    # n3 = meas_idxs.size * (1 if ENABLE_DT_ADJ else 0)
    # n4 = 0 if meas_idxs.size < 2 or not GLOBAL_ADJ else (7 if GLOBAL_ADJ > 2 else 4)
    # np = n1 + n2 + n3 + n4

    J = sp.lil_matrix((m, n), dtype=np.float32)
    i = np.arange(cam_idxs.size)

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

    if cam_intr is None or len(cam_intr) == 2:
        fx, fy = K[0, 0], K[1, 1]
    else:
        fx, fy = [cam_intr[0]] * 2

    Xr, Yr, Zr = points_3d_rot[:, 0], points_3d_rot[:, 1], points_3d_rot[:, 2]
    iZc = 1 / points_3d_cf[:, 2]
    Xn, Yn = points_3d_cf[:, 0] * iZc, points_3d_cf[:, 1] * iZc
    iZciSD = iZc / px_err_sd.flatten()

    if dist_coefs is None:
        # camera location
        J[2 * i + 0, cam_idxs*6 + 3] = -fx*iZciSD
        J[2 * i + 0, cam_idxs*6 + 5] = fx*Xn*iZciSD
        J[2 * i + 1, cam_idxs*6 + 4] = -fy*iZciSD
        J[2 * i + 1, cam_idxs*6 + 5] = fy*Yn*iZciSD

        # camera rotation
        J[2 * i + 0, cam_idxs*6 + 0] = fx*Xn*Yr*iZciSD
        J[2 * i + 0, cam_idxs*6 + 1] = -fx*(Xn*Xr + Zr)*iZciSD
        J[2 * i + 0, cam_idxs*6 + 2] = fx*Yr*iZciSD
        J[2 * i + 1, cam_idxs*6 + 0] = fy*(Yn*Yr + Zr)*iZciSD
        J[2 * i + 1, cam_idxs*6 + 1] = -fy*Yn*Xr*iZciSD
        J[2 * i + 1, cam_idxs*6 + 2] = -fy*Xr*iZciSD
    else:
        k1, k2 = dist_coefs[:2]
        R2 = Xn**2 + Yn**2
        R2k1 = R2 * k1
        R4k2 = R2 ** 2 * k2
        alpha_xy = Xn * Yn * (4*R2*k2 + 2*k1)
        y_gamma_r = Yn * (3*R2k1 + 5*R4k2 + 1)
        x_gamma_r = Xn * (3*R2k1 + 5*R4k2 + 1)
        gamma_x = R2k1 + R4k2 + Xn**2*(4*R2*k2 + 2*k1) + 1
        gamma_y = R2k1 + R4k2 + Yn**2*(4*R2*k2 + 2*k1) + 1

        # camera location
        J[2 * i + 0, cam_idxs * 6 + 3] = -fx*iZciSD*gamma_x
        J[2 * i + 0, cam_idxs * 6 + 4] = -fx*iZciSD*alpha_xy
        J[2 * i + 0, cam_idxs * 6 + 5] =  fx*iZciSD*x_gamma_r
        J[2 * i + 1, cam_idxs * 6 + 3] = -fy*iZciSD*alpha_xy
        J[2 * i + 1, cam_idxs * 6 + 4] = -fy*iZciSD*gamma_y
        J[2 * i + 1, cam_idxs * 6 + 5] =  fy*iZciSD*y_gamma_r

        # camera rotation
        J[2 * i + 0, cam_idxs * 6 + 0] =  fx*iZciSD*(Zr*alpha_xy + Yr*x_gamma_r)
        J[2 * i + 0, cam_idxs * 6 + 1] = -fx*iZciSD*(Xr*x_gamma_r + Zr*gamma_x)
        J[2 * i + 0, cam_idxs * 6 + 2] = -fx*iZciSD*(Xr*alpha_xy - Yr*gamma_x)
        J[2 * i + 1, cam_idxs * 6 + 0] =  fy*iZciSD*(Yr*y_gamma_r + Zr*gamma_y)
        J[2 * i + 1, cam_idxs * 6 + 1] = -fy*iZciSD*(Zr*alpha_xy + Xr*y_gamma_r)
        J[2 * i + 1, cam_idxs * 6 + 2] =  fy*iZciSD*(Yr*alpha_xy - Xr*gamma_y)

    # keypoint locations affect reprojection error terms
    ####################################################
    # similar to above, first separate de/dU => de/dU' * dU'/dUc * dUc/dU,
    # first one (de/dU') is -[[fx, 0],
    #                         [0, fy]]
    # second one (dU'/dUc) is [[1/Zc, 0, -Xc/Zc**2],
    #                          [0, 1/Zc, -Yc/Zc**2]]  (or the monstrosity from above if k1 or k2 are non-zero)
    # third one  (dUc/dU => d/dU (RU + P) = R) is the camera rotation matrix R
    # => i.e. rotate (de/dU' * dU'/dUc) by R^-1
    if not poses_only:
        if dist_coefs is None:
            dEu = -np.stack((fx * np.ones((len(Xn),)), np.zeros((len(Xn),)), -fx * Xn), axis=1) * iZciSD[:, None]
            dEv = -np.stack((np.zeros((len(Xn),)), fy * np.ones((len(Xn),)), -fy * Yn), axis=1) * iZciSD[:, None]
        else:
            dEu = np.zeros((len(Xn), 3))
            dEv = np.zeros_like(dEu)
            dEu[:, 0] = -fx*iZciSD*gamma_x
            dEu[:, 1] = -fx*iZciSD*alpha_xy
            dEu[:, 2] =  fx*iZciSD*x_gamma_r
            dEv[:, 0] = -fy*iZciSD*alpha_xy
            dEv[:, 1] = -fy*iZciSD*gamma_y
            dEv[:, 2] =  fy*iZciSD*y_gamma_r

        dEuc = tools.rotate_points_aa(dEu, -poses[cam_idxs, :3])
        dEvc = tools.rotate_points_aa(dEv, -poses[cam_idxs, :3])
        J[2 * i + 0, n1 + n2 + pt3d_idxs * 3 + 0] = dEuc[:, 0]
        J[2 * i + 0, n1 + n2 + pt3d_idxs * 3 + 1] = 0 if RESTRICT_3D_POINT_Y else dEuc[:, 1]
        J[2 * i + 0, n1 + n2 + pt3d_idxs * 3 + 2] = dEuc[:, 2]
        J[2 * i + 1, n1 + n2 + pt3d_idxs * 3 + 0] = dEvc[:, 0]
        J[2 * i + 1, n1 + n2 + pt3d_idxs * 3 + 1] = 0 if RESTRICT_3D_POINT_Y else dEvc[:, 1]
        J[2 * i + 1, n1 + n2 + pt3d_idxs * 3 + 2] = dEvc[:, 2]

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
    if dist_coefs is not None:
        J[2 * i + 0, n1 + n2 + n3 + 0] = tmp = -fx*Xn*R2 / px_err_sd.flatten()
        J[2 * i + 0, n1 + n2 + n3 + 1] = tmp * R2
        J[2 * i + 1, n1 + n2 + n3 + 0] = tmp = -fy*Yn*R2 / px_err_sd.flatten()
        J[2 * i + 1, n1 + n2 + n3 + 1] = tmp * R2

    # camera intrinsics (I=[fl, cx, cy]) affect reprojection error terms
    # [[e_u], [e_v]] = [[u - fl*Xn - cx],
    #                   [v - fl*Yn - cy]]
    # de/dI = [[-Xn, -1, 0],
    #          [-Yn, 0, -1]]
    if n_cam_intr > 0:
        if n_cam_intr != 2:
            J[2 * i + 0, n1 + n2 + n3 + n4 + 0] = -Xn / px_err_sd.flatten()
            J[2 * i + 1, n1 + n2 + n3 + n4 + 0] = -Yn / px_err_sd.flatten()
        if n_cam_intr > 1:
            J[2 * i + 0, n1 + n2 + n3 + n4 + (1 if n_cam_intr != 2 else 0)] = -1 / px_err_sd.flatten()
            J[2 * i + 1, n1 + n2 + n3 + n4 + (2 if n_cam_intr != 2 else 1)] = -1 / px_err_sd.flatten()

    # # time offsets affect reprojection error terms (possible to do in a better way?)
    # if ENABLE_DT_ADJ:
    #     p_offset = n1 + n2
    #     cam2meas = np.ones((np.max(pose_idxs)+1,), dtype=np.int) * -1
    #     cam2meas[meas_idxs] = np.arange(meas_idxs.size)
    #     i = np.where(cam2meas[pose_idxs] >= 0)[0]
    #     mc_idxs = cam2meas[pose_idxs[i]]
    #     A[2 * i, p_offset + mc_idxs] = 1
    #     A[2 * i + 1, p_offset + mc_idxs] = 1

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

    i = np.arange(meas_idxs.size)

    # cam locations affecting location measurement error
    iR = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(-poses[meas_idxs, :3]))
    iRs = iR / loc_err_sd
    for s in range(3):
        for r in range(3):
            J[m1 + 3 * i + s, meas_idxs * 6 + 3 + r] = iRs[:, s, r]

    # cam orientations affecting location measurement error
    R = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(poses[meas_idxs, :3]))
    Rs = R / loc_err_sd
    for s in range(3):
        for r in range(3):
            a, b = [1, 0, 0][r], [2, 2, 1][r]
            sign = -1 if r == 1 else 1
            J[m1 + 3 * i + s, meas_idxs * 6 + r] = sign * (Rs[:, a, s] * poses[meas_idxs, 3 + b]
                                                               - Rs[:, b, s] * poses[meas_idxs, 3 + a])

    # time offset affecting location measurement error
    if t_off is not None and len(t_off) > 0 and DT_ADJ_MEAS_VEL:
        v_pts3d = (v_pts2d / loc_err_sd).flatten()
        for s in range(3):
            J[m1 + 3 * i + s, n1 + i] = v_pts3d[3 * i + s]

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
    Rm = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(meas_aa))
    Rw0 = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(poses[meas_idxs, :3]))
    for u in range(len(meas_idxs)):
        D1 = tools.dlogR_dR(Rm[u].dot(Rw0[u]))
        D2 = np.vstack((-Rm[u].dot(tools.wedge(Rw0[u, :, 0])),
                        -Rm[u].dot(tools.wedge(Rw0[u, :, 1])),
                        -Rm[u].dot(tools.wedge(Rw0[u, :, 2]))))
        e_off = m1 + m2 + 3 * u
        p_off = meas_idxs[u] * 6
        J[e_off:e_off + 3, p_off:p_off + 3] = D1.dot(D2) / ori_err_sd

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

    if FIXED_PITCH_AND_ROLL:
        J[:, :2] = 0

    # maybe skip first poses
    if pose0.size > 0:
        J = J[:, pose0.size:]

    # for debugging
    if 0:
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.imshow((J.toarray() != 0).astype(int))
        plt.show()

    return J.tocsr()


def numerical_jacobian(fun, x0, eps=1e-4):
    if 1:
        from tqdm import tqdm
        y0 = fun(x0)
        J = np.zeros((len(y0), len(x0)))
        for j in tqdm(range(len(x0))):
            d = np.zeros(x0.shape)
            d[j] = eps
            x1 = x0 + d
            y1 = fun(x1)
            yd = (y1 - y0) / eps
            J[:, j] = yd
    elif 0:
        from scipy.optimize._numdiff import approx_derivative
        J = approx_derivative(fun, x0, method='2-point')
    else:
        from scipy.optimize import approx_fprime
        y0 = fun(x0)
        J = np.zeros((len(y0), len(x0)))
        for i in tqdm(range(10)):  #len(y0))):
            J[i, :] = approx_fprime(x0, lambda x: fun(x)[i], epsilon=eps)
    return J


def _solve_de_dksi():
    from sympy import symbols, Matrix, diff

    fx, fy, k1, k2, Xc, Yc, Zc, Xr, Yr, Zr, R = symbols('fx fy k1 k2 Xc Yc Zc Xr Yr Zr R')
    R = (Xc**2/Zc**2 + Yc**2/Zc**2) ** (1/2)

    de_dU = -Matrix([[fx, 0],
                     [0, fy]])
    # dU_dUc = diff(Matrix([[Xc/Zc], [Yc/Zc]]) * (1 + k1 * R**2 + k2 * R**4), Matrix([[Xc, Yc, Zc]]))
    dU_dUc = Matrix([[1/Zc * (2*Xc**2/Zc**2 * (k1 + 2*k2*R**2) + 1 + k1 * R**2 + k2 * R**4),
                      2*Xc/Zc**3 * (k1*Yc + 2*k2*Yc*R**2),
                      -Xc/Zc**2 * (1 + 3*k1*R**2 + 5*k2*R**4)],
                     [1/Zc * (2*Yc**2/Zc**2 * (k1 + 2*k2*R**2) + 1 + k1 * R**2 + k2 * R**4),
                      2*Yc/Zc**3 * (k1*Xc + 2*k2*Xc*R**2),
                      -Yc/Zc**2 * (1 + 3*k1*R**2 + 5*k2*R**4)]])
    dUc_dksi = Matrix([[1, 0, 0, 0, Zr, -Yr],
                       [0, 1, 0, -Zr, 0, Xr],
                       [0, 0, 1, Yr, -Xr, 0]])
    de_dksi = de_dU * dU_dUc * dUc_dksi
    return de_dksi


def _rotated_points(params, pose0, fixed_pt3d, n_cams, n_pts, n_dist, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K, px_err_sd,
                    meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef):
    """
    Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    if isinstance(params, (tuple, list)):
        params = np.array(params)

    params = np.hstack((pose0, params))
    poses, pts3d, dist_coefs, t_off = _unpack(params, n_cams, n_pts if len(fixed_pt3d) == 0 else 0, n_dist,
                                              meas_idxs.size, meas_r0=None if len(meas_idxs) < 2 else meas_r[0])

    points_3d = fixed_pt3d if len(pts3d) == 0 else pts3d
    points_3d_cf = tools.rotate_points_aa(points_3d[pt3d_idxs], poses[cam_idxs, :3]) + poses[cam_idxs, 3:6]
    return points_3d_cf.flatten()


def _rot_pt_jac(params, pose0, fixed_pt3d, n_cams, n_pts, n_dist, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K, px_err_sd,
                meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef):

    params = np.hstack((pose0, params))
    poses, pts3d, dist_coefs, t_off = _unpack(params, n_cams, n_pts if len(fixed_pt3d) == 0 else 0, n_dist,
                                              meas_idxs.size, meas_r0=None if len(meas_idxs) < 2 else meas_r[0])

    # TODO: take into account dist_coefs? or is this function deprecated?

    poses_only = len(fixed_pt3d) > 0
    points_3d = fixed_pt3d if poses_only else pts3d
    points_3d_cf = tools.rotate_points_aa(points_3d[pt3d_idxs], poses[cam_idxs, :3]) #+ poses[pose_idxs, 3:6]

    # output term count (rotated coords)
    m = cam_idxs.size * 3

    # parameter count (6d poses)
    n = n_cams * 6

    J = np.zeros((m, n))
    i = np.arange(cam_idxs.size)

    # poses affect reprojection error terms
    ########################################
    #  - NOTE: for the following to work, need the poses in cam-to-world, i.e. rotation of world origin first,
    #    then translation in rotated coordinates
    #  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6891346/  equation (11)
    #  - also https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf, section 10.3.5, eq 10.23
    #  - ksi: first three location, second three rotation
    #  - de/dksi = de/dU' * dU'/dksi
    #  - de/dU' = -[[fx/Zc, 0, -fx*Xc/Zc**2],
    #               [0, fy/Zc, -fy*Yc/Zc**2]]
    #  - dU'/dw = [I3 | -[U']^] = [[1, 0, 0, 0, Zc, -Yc],
    #                              [0, 1, 0, -Zc, 0, Xc],
    #                              [0, 0, 1, Yc, -Xc, 0]]
    #  - -[[fx/Zc, 0, -Xc*fx/Zc**2, | -Xc*Yc*fx/Zc**2, Xc**2*fx/Zc**2 + fx, -Yc*fx/Zc],
    #      [0, fy/Zc, -Yc*fy/Zc**2, | -Yc**2*fy/Zc**2 - fy, Xc*Yc*fy/Zc**2, Xc*fy/Zc]]    / px_err_sd

    Xc, Yc, Zc = points_3d_cf[:, 0], points_3d_cf[:, 1], points_3d_cf[:, 2]

    # camera location
    J[3 * i + 0, cam_idxs*6 + 3] = 1
    J[3 * i + 0, cam_idxs*6 + 4] = 0
    J[3 * i + 0, cam_idxs*6 + 5] = 0
    J[3 * i + 1, cam_idxs*6 + 3] = 0
    J[3 * i + 1, cam_idxs*6 + 4] = 1
    J[3 * i + 1, cam_idxs*6 + 5] = 0
    J[3 * i + 2, cam_idxs*6 + 3] = 0
    J[3 * i + 2, cam_idxs*6 + 4] = 0
    J[3 * i + 2, cam_idxs*6 + 5] = 1

    # camera rotation
    J[3 * i + 0, cam_idxs*6 + 0] = 0
    J[3 * i + 0, cam_idxs*6 + 1] = Zc
    J[3 * i + 0, cam_idxs*6 + 2] = -Yc
    J[3 * i + 1, cam_idxs*6 + 0] = -Zc
    J[3 * i + 1, cam_idxs*6 + 1] = 0
    J[3 * i + 1, cam_idxs*6 + 2] = Xc
    J[3 * i + 2, cam_idxs*6 + 0] = Yc
    J[3 * i + 2, cam_idxs*6 + 1] = -Xc
    J[3 * i + 2, cam_idxs*6 + 2] = 0

    return J


def _unpack(params, n_cams, n_pts, n_dist, n_cam_intr, n_meas, meas_r0):
    """
    Retrieve camera parameters and 3-D coordinates.
    """
    a = n_cams * 6
    b = a + n_meas * (1 if ENABLE_DT_ADJ else 0)
    c = b + n_pts * 3
    d = c + (7 if GLOBAL_ADJ > 2 else 4 if GLOBAL_ADJ else 0)
    e = d + n_dist
    f = e + n_cam_intr

    poses = params[:a].reshape((n_cams, 6))

    if FIXED_PITCH_AND_ROLL:
        poses[:, 2] = np.sign(poses[:, 2]) * np.linalg.norm(poses[:, :3], axis=1)
        poses[:, :2] = 0

    t_off = params[a:b].reshape((-1, 1))
    pts3d = params[b:c].reshape((n_pts, 3))
    dist_coefs, cam_intr = None, None

    if n_meas > 1 and GLOBAL_ADJ:
        rot_off = np.array([[0, 0, 0]]) if GLOBAL_ADJ < 3 else params[d-7:d-4].reshape((1, 3))
        loc_off = params[d-4:d-1]
        scale_off = params[d-1]

        # use rot_off, loc_off and scale_off to transform cam_params and pts3d instead of the measurements
        if USE_WORLD_CAM_FRAME:
            poses[:, :3] = tools.rotate_rotations_aa(np.repeat(-rot_off, len(poses), axis=0), poses[:, :3])
            poses[:, 3:] = tools.rotate_points_aa(poses[:, 3:] - meas_r0 - loc_off, -rot_off) * math.exp(-scale_off) + meas_r0
            pts3d = tools.rotate_points_aa(pts3d - meas_r0 - loc_off, -rot_off) * math.exp(-scale_off) + meas_r0
        else:
            # TODO: debug following, probably wrong
            cam_loc_wf = tools.rotate_points_aa(-poses[:, 3:], -poses[:, :3])
            cam_loc_wf_adj = tools.rotate_points_aa(cam_loc_wf - meas_r0 - loc_off, -rot_off) * math.exp(-scale_off) + meas_r0
            poses[:, 3:] = tools.rotate_points_aa(-cam_loc_wf_adj, poses[:, :3])   # can probably do with fewer calculations
            poses[:, :3] = tools.rotate_rotations_aa(np.repeat(rot_off, len(poses), axis=0), poses[:, :3])
            pts3d = tools.rotate_points_aa(pts3d - meas_r0 - loc_off, -rot_off) * math.exp(-scale_off) + meas_r0

    if n_dist > 0:
        dist_coefs = params[d:e].reshape((-1,))

    if n_cam_intr > 0:
        cam_intr = params[e:f].reshape((-1,))

    return poses, pts3d, dist_coefs, cam_intr, t_off


def _project(pts3d, poses, K, dist_coefs=None):
    """Convert 3-D points to 2-D by projecting onto images."""
    if USE_WORLD_CAM_FRAME:
        pts3d_rot = tools.rotate_points_aa(pts3d - poses[:, 3:6], -poses[:, 0:3])
    else:
        pts3d_rot = tools.rotate_points_aa(pts3d, poses[:, 0:3]) + poses[:, 3:6]

    P = pts3d_rot / pts3d_rot[:, 2:3]

    if dist_coefs is not None:
        k1, k2 = np.pad(dist_coefs[0:2], (0, 2 - len(dist_coefs[0:2])), 'constant')
        r2 = np.sum(P[:, 0:2] ** 2, axis=1)[:, None]
        P[:, 0:2] *= (1 + k1 * r2 + k2 * r2 ** 2)

    pts2d_proj = K[:2, :].dot(P.T).T
    return pts2d_proj


def project(pts3d, poses, K, dist_coefs=None):
    return _project(pts3d, poses, K, dist_coefs)


def marginalize(x, r, J, marginalize_pose_idxs, marginalize_kp3d_idxs,
                n_cams, n_pts, n_dist, n_cam_intr, measure_idxs, prior_rows, prior_weight=1.0):

    a = n_cams * 6
    b = a + len(measure_idxs) * (1 if ENABLE_DT_ADJ else 0)
    c = b + n_pts * 3
    d = c + (7 if GLOBAL_ADJ > 2 else 4 if GLOBAL_ADJ else 0)
    e = d + n_dist
    f = e + n_cam_intr

    # determine idxs of x that will be marginalized
    m_x = np.zeros((f,), dtype=bool)
    for i in marginalize_pose_idxs:
        m_x[i*6: (i+1)*6] = True
    if ENABLE_DT_ADJ:
        m_x[a + np.where(np.in1d(measure_idxs, marginalize_pose_idxs))[0]] = True
    for i in marginalize_kp3d_idxs:
        m_x[b + i*3: b + (i+1)*3] = True

    k_x = np.logical_not(m_x)   # idxs that will be kept (κ-vars + u-vars)

    # from the article:
    # "To preserve sparsity in the landmark-landmark Hessian block, we drop observations of landmarks that will stay
    #  active in frames which are about to be marginalized, before calculating H~ and ~b. This means that landmarks are
    #  never part of the κ-variables."
    k_x[b:c] = False            # κ-vars + u-vars, without landmark related cols

    if sp.issparse(J):
        nzr, nzc = J.nonzero()
        J_rows = np.zeros((J.shape[0],), dtype=bool)
        J_rows[nzr[m_x[nzc]]] = True
    else:
        J_rows = np.any(np.logical_not(np.isclose(J[:, m_x], 0)), axis=1)     # rows affected by marginalized (mu-) vars

    if prior_rows > 0:
        # always include all prior related rows
        J_rows[-prior_rows:] = True
        if prior_weight != 1.0:
            J[-prior_rows:, :] *= 1/prior_weight
            r[-prior_rows:] *= 1/prior_weight

    if sp.issparse(J):
        J_cols = np.zeros((J.shape[1],), dtype=bool)
        J_cols[nzc[J_rows[nzr]]] = True  # mu-vars + κ-vars
    else:
        J_cols = np.any(np.logical_not(np.isclose(J[J_rows, :], 0)), axis=0)  # cols affecting above rows (mu-vars + κ-vars)

    k_x = np.logical_and(J_cols, k_x)   # now only κ-vars (and vars related to previous prior)
    J_cols_m = np.where(m_x)[0]         # col indices related to mu-vars
    J_cols_k = np.where(k_x)[0]         # col indices related to κ-vars
    J_cols = np.concatenate((J_cols_m, J_cols_k), axis=0)

    Js = J[np.ix_(J_rows, J_cols)]
    rs = r[J_rows]

    # TODO: use specialized, flat QR decomposition as in "Square Root Marginalization for Sliding-Window Bundle Adjustment"
    Q, R = np.linalg.qr(Js.toarray() if sp.issparse(Js) else Js, 'complete')

    n_m = len(J_cols_m)
    R2k = R[n_m:, n_m:]
    Q2 = Q[:, n_m:]
    Q2Tr = Q2.T.dot(rs)

    n_zeros = np.where(np.flip(np.any(np.logical_not(np.isclose(R2k, 0)), axis=1)))[0]
    if len(n_zeros) > 0 and n_zeros[0] > 0:
        R2k = R2k[:-n_zeros[0], :]
        Q2Tr = Q2Tr[:-n_zeros[0]]

    new_k = k_x
    new_x = x[k_x]
    new_r = Q2Tr
    new_J = R2k
    return new_k, new_x, new_r, new_J
