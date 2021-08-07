"""
Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""
import math
import sys
import logging
from functools import lru_cache

import numpy as np
import quaternion

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from visnav.algo import tools
from visnav.algo.odo.base import LogWriter

GLOBAL_ADJ = 0      # 3: all incl rotation, 2: location and scale, 1: scale only
ONLY_YAW = 0
ANALYTICAL_JACOBIAN = 0
ENABLE_DT_ADJ = 0


def vis_gps_bundle_adj(poses: np.ndarray, pts3d: np.ndarray, pts2d: np.ndarray, v_pts2d: np.ndarray,
                       cam_idxs: np.ndarray, pt3d_idxs: np.ndarray, K: np.ndarray, px_err_sd: float,
                       meas_r: np.ndarray, meas_aa: np.ndarray, t_off: np.ndarray, meas_idxs: np.ndarray,
                       loc_err_sd: float, ori_err_sd: float,
                       log_writer=None, max_nfev=None, skip_pose_n=1, poses_only=False, huber_coef=False):
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

    cam_idxs with shape (n_observations,)
            contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.

    pt3d_idxs with shape (n_observations,)
            contains indices of points (from 0 to n_points - 1) involved in each observation.

    """
    assert len(poses.shape) == 2 and poses.shape[1] == 6, 'wrong shape poses: %s' % (poses.shape,)
    assert len(pts3d.shape) == 2 and pts3d.shape[1] == 3, 'wrong shape pts3d: %s' % (pts3d.shape,)
    assert len(pts2d.shape) == 2 and pts2d.shape[1] == 2, 'wrong shape pts2d: %s' % (pts2d.shape,)
    assert len(cam_idxs.shape) == 1, 'wrong shape cam_idxs: %s' % (cam_idxs.shape,)
    assert len(pt3d_idxs.shape) == 1, 'wrong shape pt3d_idxs: %s' % (pt3d_idxs.shape,)
    assert K.shape == (3, 3), 'wrong shape K: %s' % (K.shape,)

    #assert not skip_pose0, 'some bug with skipping first pose optimization => for some reason cost stays high, maybe problem with A?'

    n_cams = poses.shape[0]
    n_pts = pts3d.shape[0]
    A = _bundle_adjustment_sparsity(n_cams, n_pts, cam_idxs, pt3d_idxs, meas_idxs, poses_only)   # n_cams-a or n_cams?
    if skip_pose_n > 0:
        A = A[:, skip_pose_n * 6:]

    # poses[:, :3] = poses[:, :3] / np.pi * 180

    pose0 = poses[0:skip_pose_n].ravel()
    x0 = [poses[skip_pose_n:].ravel()]

    if poses_only:
        fixed_pt3d = pts3d
    else:
        fixed_pt3d = np.zeros((0, 3))
        x0.append(pts3d.ravel())

    if ENABLE_DT_ADJ:
        x0.append(t_off.ravel())

    if len(meas_idxs) > 1 and GLOBAL_ADJ:
        x0.append(np.array([0] * (7 if GLOBAL_ADJ > 2 else 4)))

    x0 = np.hstack(x0)

    if False:
        err = _costfun(x0, pose0, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K)
        print('ERR: %.4e' % (np.sum(err**2)/2))

    @lru_cache(1)
    def _cfun(x):
        return _costfun(x, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K, px_err_sd,
                       meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef)

    def cfun(x):
        return _cfun(tuple(x))

    def jac(x):
        return _jacobian(x, cfun(x), pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K, px_err_sd,
                         meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef)

    # if isinstance(huber_coef, (tuple, list)):
    #     huber_coef = np.array([huber_coef[0]] * len(px_err)
    #                           + [huber_coef[1]] * len(loc_err)
    #                           + [huber_coef[2]] * len(ori_err))

    tmp = sys.stdout
    sys.stdout = log_writer or LogWriter()
    res = least_squares(cfun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, xtol=1e-5, method='trf',
                        jac='3-point' if not ANALYTICAL_JACOBIAN else jac,
                        # for some reason doesnt work as well as own huber loss
                        # tr_solver='lsmr',
                        loss='huber', f_scale=3.0,  #huber_coef,
                        # args=(pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K, px_err_sd,
                        #       meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef),
                        max_nfev=max_nfev)
    sys.stdout = tmp

    # TODO (1): return res.fun somewhat processed (per frame: mean px err, meas errs),
    #           so that can follow the errors and notice/debug problems

    new_poses, new_pts3d, new_t_off = _unpack(res.x, n_cams - skip_pose_n,
                                              0 if poses_only else n_pts, meas_idxs.size,
                                              meas_r0=None if len(meas_idxs) < 2 else meas_r[0])
    return new_poses, new_pts3d, new_t_off


def _costfun(params, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K, px_err_sd,
             meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef):
    """
    Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """

    params = np.hstack((pose0, params))
    poses, pts3d, t_off = _unpack(params, n_cams, n_pts if len(fixed_pt3d) == 0 else 0, meas_idxs.size,
                                  meas_r0=None if len(meas_idxs) < 2 else meas_r[0])

    points_3d = fixed_pt3d if len(pts3d) == 0 else pts3d
    points_proj = _project(points_3d[pt3d_idxs], poses[cam_idxs], K)

    if t_off is not None and len(t_off) > 0:
        d_pts2d = v_pts2d * t_off[cam_idxs]  # for 1st order approximation of where the 2d points would when meas_r recorded
    else:
        d_pts2d = 0

    # # global rotation, translation and scale, applied to measurements
    # if len(meas_idxs) > 1 and GLOBAL_ADJ:
    #     rot_off = np.array([[0, 0, 0]]) if GLOBAL_ADJ < 3 else params[d-7:d-4].reshape((1, 3))
    #     loc_off = params[d-4:d-1]
    #     scale_off = params[d-1]
    #
    #     meas_r = _rotate(meas_r - meas_r[0], rot_off) * math.exp(scale_off) + meas_r[0] + loc_off
    #     meas_aa = _rotate_rotations(np.repeat(rot_off, len(meas_aa), axis=0), meas_aa)

    px_err = (((pts2d + d_pts2d) - points_proj) / px_err_sd).ravel()
    loc_err = ((meas_r - poses[meas_idxs, 3:]) / loc_err_sd).ravel()

    # rotation error in aa, might be better to use quaternions everywhere
    rot_err_aa = _rotate_rotations(meas_aa, -poses[meas_idxs, :3])

    # convert to yaw-pitch-roll-angles as aa cant have negative norm, small rotation can be 2*pi - 0.001
    # - note that gimbal lock is not a problem if error stays reasonably small
    rot_err_q = quaternion.as_float_array(quaternion.from_rotation_vector(rot_err_aa))

    rot_err_ypr = tools.wrap_rads(tools.qarr_to_ypr(rot_err_q))
    ori_err = (rot_err_ypr / ori_err_sd).ravel()

    err = np.concatenate((px_err, loc_err, ori_err))
    if isinstance(huber_coef, (tuple, list)):
        huber_coef = np.array([huber_coef[0]] * len(px_err)
                              + [huber_coef[1]] * len(loc_err)
                              + [huber_coef[2]] * len(ori_err))

    return err  #tools.pseudo_huber_loss(err, huber_coef) if huber_coef is not False else (err ** 2)


def _bundle_adjustment_sparsity(n_cams, n_pts, cam_idxs, pt3d_idxs, meas_idxs, poses_only):
    # error term count  (first 2d reprojection errors, then 3d gps measurement error, then 3d orientation meas err)
    m1, m2, m3 = cam_idxs.size * 2, meas_idxs.size * 3, meas_idxs.size * 3
    m = m1 + m2 + m3

    # parameter count (6d poses, n x 3d keypoint locations, 1d time offset)
    n1 = n_cams * 6
    n2 = (0 if poses_only else n_pts * 3)
    n3 = meas_idxs.size * (1 if ENABLE_DT_ADJ else 0)
    n4 = 0 if meas_idxs.size < 2 or not GLOBAL_ADJ else (7 if GLOBAL_ADJ > 2 else 4)
    n = n1 + n2 + n3 + n4

    A = lil_matrix((m, n), dtype=int)
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

    # frame loc affect loc measurement error terms (x~x, y~y, z~z only)
    i = np.arange(meas_idxs.size)
    for s in range(3):
        A[m1 + 6 * i + s, meas_idxs * 6 + 3 + s] = 1

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


def _jacobian(params, err, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K, px_err_sd,
              meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef):
    """
    cost function jacobian, from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6891346/
    """
    assert not GLOBAL_ADJ and not ONLY_YAW and not ENABLE_DT_ADJ, \
        'analytical jacobian does not support GLOBAL_ADJ, ONLY_YAW or ENABLE_DT_ADJ'

    params = np.hstack((pose0, params))
    poses, pts3d, t_off = _unpack(params, n_cams, n_pts if len(fixed_pt3d) == 0 else 0, meas_idxs.size,
                                  meas_r0=None if len(meas_idxs) < 2 else meas_r[0])

    poses_only = len(fixed_pt3d) > 0
    points_3d = fixed_pt3d if poses_only else pts3d
    points_3d_cf = _rotate(points_3d[pt3d_idxs] - poses[cam_idxs, 3:6], -poses[cam_idxs, :3])

    # error term count  (first 2d reprojection errors, then 3d gps measurement error, then 3d orientation meas err)
    m1, m2, m3 = cam_idxs.size * 2, meas_idxs.size * 3, meas_idxs.size * 3
    m = m1 + m2 + m3

    # parameter count (6d poses, n x 3d keypoint locations, 1d time offset)
    n1 = n_cams * 6
    n2 = (0 if poses_only else n_pts * 3)
    n = n1 + n2
    # n3 = meas_idxs.size * (1 if ENABLE_DT_ADJ else 0)
    # n4 = 0 if meas_idxs.size < 2 or not GLOBAL_ADJ else (7 if GLOBAL_ADJ > 2 else 4)
    # n = n1 + n2 + n3 + n4

    J = lil_matrix((m, n), dtype=int)
    i = np.arange(cam_idxs.size)

    ## derivative of pseudo huber loss can be applied afterwards by multiplying with the err derivative
    # NOTE: no need as least_squares already does this
    # - because d phl(e)/dksi = d phl(e)/de * de/dksi
    # - derivative d phl(e)/de = e/sqrt(1+e**2/delta**2)
    # if isinstance(huber_coef, (tuple, list)):
    #     huber_coef = np.array([huber_coef[0]] * m1 + [huber_coef[1]] * m2 + [huber_coef[2]] * m3)
    # dhuber = err / np.sqrt(1 + err**2/huber_coef**2)

    # poses affect reprojection error terms
    ########################################
    #  - TODO: for the following to work, need the poses in cam-to-world, i.e. rotation of world origin first,
    #    then translation in rotated coordinates
    #  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6891346/  equation (11), first three location, second three rotation
    #  - [[rho*fx, 0, -rho**2*fx*X, rho**2*fx*X*Y, -fx-rho**2*fx*X**2, rho*fx*Y],
    #     [0, rho*fy, -rho**2*fy*Y, fy+rho**2*fy*Y**2, -rho**2*fy*X*Y, -rho*fy*X]] / px_err_sd
    #  - d phl(e)/dksi = d phl(e)/de * de/dU' * dU'/dksi

    fx, fy = K[0, 0], K[1, 1]
    Xc, Yc, rho = points_3d_cf[:, 0], points_3d_cf[:, 1], 1 / points_3d_cf[:, 2]

    # camera location
    J[i*2 + 0, cam_idxs*6 + 3] = - (rho * fx) / px_err_sd
    J[i*2 + 0, cam_idxs*6 + 5] = - (-rho**2 * fx * Xc) / px_err_sd
    J[i*2 + 1, cam_idxs*6 + 4] = - (rho * fy) / px_err_sd
    J[i*2 + 1, cam_idxs*6 + 5] = - (-rho**2 * fy * Yc) / px_err_sd

    # camera rotation
    J[i*2 + 0, cam_idxs*6 + 0] = - (rho**2 * fx * Xc * Yc) / px_err_sd
    J[i*2 + 0, cam_idxs*6 + 1] = - (-fx - rho**2 * fx * Xc**2) / px_err_sd
    J[i*2 + 0, cam_idxs*6 + 2] = - (rho * fx * Yc) / px_err_sd
    J[i*2 + 1, cam_idxs*6 + 0] = - (fy + rho**2 * fy * Yc**2) / px_err_sd
    J[i*2 + 1, cam_idxs*6 + 1] = - (-rho**2 * fy * Xc * Yc) / px_err_sd
    J[i*2 + 1, cam_idxs*6 + 2] = - (-rho * fy * Xc) / px_err_sd

    # keypoint locations affect reprojection error terms
    ####################################################
    # similar to above, first separate de/dU => de/dU' * dU'/dU,
    # first one (de/dU') is [[fx/Z', 0, -fx*X'/Z'**2],
    #                        [0, fy/Z', -fy*Y'/Z'**2]]
    # second one (dU'/dU => d/dU (RU + P) = R) is the camera rotation matrix R
    # => i.e. rotate de/dU' by R^-1
    if not poses_only:
        dEu = - np.hstack((fx * rho, np.zeros((len(rho),)), -fx * Xc * rho**2)) / px_err_sd
        dEv = - np.hstack((np.zeros((len(rho),)), fy * rho, -fy * Yc * rho**2)) / px_err_sd
        dEuc = _rotate(dEu, -poses[cam_idxs, :3])
        dEvc = _rotate(dEv, -poses[cam_idxs, :3])
        J[2 * i + 0, n1 + pt3d_idxs * 3 + 0] = dEuc[:, 0]
        J[2 * i + 0, n1 + pt3d_idxs * 3 + 1] = dEuc[:, 1]
        J[2 * i + 0, n1 + pt3d_idxs * 3 + 2] = dEuc[:, 2]
        J[2 * i + 1, n1 + pt3d_idxs * 3 + 0] = dEvc[:, 0]
        J[2 * i + 1, n1 + pt3d_idxs * 3 + 1] = dEvc[:, 1]
        J[2 * i + 1, n1 + pt3d_idxs * 3 + 2] = dEvc[:, 2]

    # # time offsets affect reprojection error terms (possible to do in a better way?)
    # if ENABLE_DT_ADJ:
    #     p_offset = n1 + n2
    #     cam2meas = np.ones((np.max(cam_idxs)+1,), dtype=np.int) * -1
    #     cam2meas[meas_idxs] = np.arange(meas_idxs.size)
    #     i = np.where(cam2meas[cam_idxs] >= 0)[0]
    #     mc_idxs = cam2meas[cam_idxs[i]]
    #     A[2 * i, p_offset + mc_idxs] = 1
    #     A[2 * i + 1, p_offset + mc_idxs] = 1

    # frame loc affect loc measurement error terms (x~x, y~y, z~z only)
    ###################################################################
    # need to convert cam-to-world pose into world-cam location so that can compare
    # err = ((R^-1 * -P) - (Rm^-1 * -Pm)) / loc_err_sd
    # derr/dP = -R^-1 / loc_err_sd
    #?derr/dR = u^ * -P,   where R^-1 ~=> u, rotation axis; u^ = [[0, -uz, uy],
    #                                                             [uz, 0, -ux],
    #                                                             [-uy, ux, 0]]
    # derr/dw = -R'[-P]^ * (ww' + (R-I)[-w]^) / ||w||^2    from https://arxiv.org/pdf/1312.0788.pdf, eq (8)
    #                                                   / loc_err_sd
    # or https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf, section 10.3.5, eq 10.20:
    # derr/dw = d/dw [R(w0)R(w)]^-1 * (-P/loc_err_sd) = d/dR(w0)R(w) [R(w0)R(w)]' * (-P/loc_err_sd)
    #                                                   * d/dw R(w0)R(w)
    #  => -[R(w0)R(w)]' * (-P/loc_err_sd) * R(w0) * np.vstack([-e1^, -e2^, -e3^])
    # TODO: how to implement? need to unstack rotation matrices from np.vstack([-e1^, -e2^, -e3^]) ??
    i = np.arange(meas_idxs.size)
    for s in range(3):
        A[m1 + 6 * i + s, meas_idxs * 6 + 3 + s] = 1

    # TODO:
    #   - check that all (also non-active) 3d points included in ba
    #   - check jacobian with numerically computed jacobian
    #   - toy problem

    # orientation components affect ori measurement error terms
    ###########################################################
    # err = th_e / loc_err_sd * w_e
    # exp(err^) = exp(th_e / loc_err_sd * w_e^) = (R * Rm^-1)
    # exp(w_e) = Rm^-1 * R(w)
    # dw_e/dw = d/dw log(Rm^-1 * R) = d/dw (th/2 sin th)*(R(w)' * Rm - Rm' * R(w)), where th = acos((tr(Rm' * R)-1)/2)
    #
    # based on https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
    # by chain rule:
    # dw_e/dw = d/dw log(Rm' * R(w)) = d/d(Rm'*R(w)) log(Rm'*R(w))  *  d/dR(w) Rm'*R(w)  *  d/dw R(w0)*exp(w)
    # 1) d/dR log(R) = ?  (section 10.3.2, eq 10.11)
    # 2) d/dR Rm'*R = Rm'  (section 7.3.1, eq 7.14)
    # 3) d/dw R(w0) * exp(w) = R(w0) * np.vstack([-e1^, -e2^, -e3^]) (vector stacked view of rot mx, sections 10.3.1, 10.3.4, eq 10.8, 10.18)
    # TODO: how to implement??
    for s in range(3):
        for r in range(3):
            A[m1 + 3 + 6 * i + s, meas_idxs * 6 + r] = 1

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

    # apply pseudo huber loss derivative (no need as least_squares already does this)
    # J = J * dhuber

    # maybe skip first poses
    if pose0.size > 0:
        J = J[:, pose0.size:]

    # for debugging
    if 0:
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.imshow((J.toarray() != 0).astype(int))
        plt.show()

    return J


def _unpack(params, n_cams, n_pts, n_meas, meas_r0):
    """
    Retrieve camera parameters and 3-D coordinates.
    """
    a = n_cams * 6
    b = a + n_pts * 3
    c = b + n_meas * (1 if ENABLE_DT_ADJ else 0)
    d = c + (7 if GLOBAL_ADJ > 2 else 4)

    poses = params[:a].reshape((n_cams, 6))
    # poses[:, :3] = poses[:, :3]/180*np.pi

    if ONLY_YAW:
        poses[:, 2] = np.sign(poses[:, 2]) * np.linalg.norm(poses[:, :3], axis=1)
        poses[:, :2] = 0

    pts3d = params[a:b].reshape((n_pts, 3))
    t_off = params[b:c].reshape((-1,))

    if n_meas > 1 and GLOBAL_ADJ:
        rot_off = np.array([[0, 0, 0]]) if GLOBAL_ADJ < 3 else params[d-7:d-4].reshape((1, 3))
        loc_off = params[d-4:d-1]
        scale_off = params[d-1]

        # use rot_off, loc_off and scale_off to transform cam_params and pts3d instead of the measurements
        poses[:, :3] = _rotate_rotations(np.repeat(-rot_off, len(poses), axis=0), poses[:, :3])
        poses[:, 3:] = _rotate(poses[:, 3:] - meas_r0 - loc_off, -rot_off) * math.exp(-scale_off) + meas_r0
        pts3d = _rotate(pts3d - meas_r0 - loc_off, -rot_off) * math.exp(-scale_off) + meas_r0

    return poses, pts3d, t_off


def _rotate(points, rot_vecs):
    return _cached_rotate(tuple(points.flatten()), tuple(rot_vecs.flatten()))


@lru_cache(maxsize=1)
def _cached_rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    points = np.array(points).reshape((-1, 3))
    rot_vecs = np.array(rot_vecs).reshape((-1, 3))

    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rot_points = cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
    return rot_points


def _rotate_rotations(rot_vecs1, rot_vecs2):
    """Rotate rotation vectors by given rotation vectors.

    Rodrigues' rotation formula is used, from
      https://math.stackexchange.com/questions/382760/composition-of-two-axis-angle-rotations
    """

    a = np.linalg.norm(rot_vecs1, axis=1)[:, np.newaxis]
    b = np.linalg.norm(rot_vecs2, axis=1)[:, np.newaxis]
    sin_a, cos_a = np.sin(0.5 * a), np.cos(0.5 * a)
    sin_b, cos_b = np.sin(0.5 * b), np.cos(0.5 * b)

    with np.errstate(invalid='ignore'):
        va = rot_vecs1 / a
        vb = rot_vecs2 / b
        va = np.nan_to_num(va)
        vb = np.nan_to_num(vb)

    c = 2 * np.arccos(np.clip(cos_a * cos_b - sin_a * sin_b * np.sum(va * vb, axis=1)[:, None], -1, 1))
    sin_c = np.sin(0.5 * c)
    I = (c != 0).flatten()

    res = np.zeros(rot_vecs1.shape)
    res[I] = c[I] * (1 / sin_c[I]) * (sin_a[I] * cos_b[I] * va[I] +
                                      cos_a[I] * sin_b[I] * vb[I] +
                                      sin_a[I] * sin_b[I] * np.cross(va[I], vb[I]))

    return res


def _project(pts3d, poses, K):
    """Convert 3-D points to 2-D by projecting onto images."""
    pts3d_rot = _rotate(pts3d - poses[:, 3:6], -poses[:, :3])
    # pts2d_proj = _rotate(pts3d, poses[:, :3])
    # pts2d_proj += poses[:, 3:6]

    # pts2d_proj = -pts2d_proj[:, :2] / pts2d_proj[:, 2, np.newaxis]
    # f = poses[:, 6]
    # k1 = poses[:, 7]
    # k2 = poses[:, 8]
    # n = np.sum(pts2d_proj ** 2, axis=1)
    # r = 1 + k1 * n + k2 * n ** 2
    # pts2d_proj *= (r * f)[:, np.newaxis]

    pts2d_proj = K.dot(pts3d_rot.T).T                    # own addition
    pts2d_proj = pts2d_proj[:, :2] / pts2d_proj[:, 2:]   # own addition

    return pts2d_proj


def project(pts3d, poses, K):
    return _project(pts3d, poses, K)

