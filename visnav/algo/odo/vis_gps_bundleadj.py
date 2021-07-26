"""
Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""
import sys
import logging

import numpy as np
import quaternion

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from visnav.algo import tools
from visnav.algo.odo.base import LogWriter


def vis_gps_bundle_adj(poses: np.ndarray, pts3d: np.ndarray, pts2d: np.ndarray, v_pts2d: np.ndarray,
                       cam_idxs: np.ndarray, pt3d_idxs: np.ndarray, K: np.ndarray, px_err_sd: float,
                       meas_r: np.ndarray, meas_q: np.ndarray, t_off: np.ndarray, meas_idxs: np.ndarray,
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

    if poses_only:
        x0 = poses[skip_pose_n:].ravel()
        fixed_pt3d = pts3d
    else:
        x0 = np.hstack((poses[skip_pose_n:].ravel(), pts3d.ravel()))
        fixed_pt3d = np.zeros((0, 3))

    x0 = np.hstack((x0.ravel(), t_off.ravel()))
    pose0 = poses[0:skip_pose_n].ravel()

    if False:
        err = _costfun(x0, pose0, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K)
        print('ERR: %.4e' % (np.sum(err**2)/2))

    tmp = sys.stdout
    sys.stdout = log_writer or LogWriter()
    res = least_squares(_costfun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, xtol=1e-5, method='trf',
                        # for some reason doesnt work as well as own huber loss
                        # tr_solver='lsmr', # loss='huber', f_scale=huber_coef
                        args=(pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K, px_err_sd,
                              meas_r, meas_q, meas_idxs, loc_err_sd, ori_err_sd, huber_coef),
                        max_nfev=max_nfev)
    sys.stdout = tmp

    # TODO (1): return res.fun somewhat processed (per frame: mean px err, meas errs),
    #           so that can follow the errors and notice/debug problems

    new_poses, new_pts3d, new_t_off = _optimized_params(res.x, n_cams - skip_pose_n,
                                                        0 if poses_only else n_pts, meas_idxs.size)
    return new_poses, new_pts3d, new_t_off


def _rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
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


def _costfun(params, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, v_pts2d, K, px_err_sd,
             meas_r, meas_aa, meas_idxs, loc_err_sd, ori_err_sd, huber_coef):
    """
    Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    enable_dt = False

    params = np.hstack((pose0, params))
    a = n_cams * 6
    b = a + (n_pts * 3 if len(fixed_pt3d) == 0 else 0)
    c = b + meas_idxs.size * 1

    poses = params[:a].reshape((n_cams, 6))
    points_3d = fixed_pt3d if len(fixed_pt3d) > 0 else params[a:b].reshape((n_pts, 3))
    points_proj = _project(points_3d[pt3d_idxs], poses[cam_idxs], K)

    t_off = np.zeros((n_cams, 1))
    t_off[meas_idxs] = params[b:c].reshape((meas_idxs.size, 1))
    d_pts2d = v_pts2d * t_off[cam_idxs]  # for 1st order approximation of where the 2d points would when meas_r recorded

    px_err = ((points_proj - (pts2d + d_pts2d * (1 if enable_dt else 0))) / px_err_sd).ravel()
    loc_err = ((poses[meas_idxs, 3:] - meas_r) / loc_err_sd).ravel()

    # rotation error in aa, might be better to use quaternions everywhere
    rot_err_aa = _rotate_rotations(poses[meas_idxs, :3], -meas_aa)

    # convert to yaw-pitch-roll-angles as aa cant have negative norm, small rotation can be 2*pi - 0.001
    # - note that gimbal lock is not a problem if error stays reasonably small
    rot_err_q = quaternion.as_float_array(quaternion.from_rotation_vector(rot_err_aa))

    # TODO: check for frame.id==23 where yaw skips from +180 to -180
    rot_err_ypr = tools.wrap_rads(tools.qarr_to_ypr(rot_err_q))
    ori_err = (rot_err_ypr / ori_err_sd).ravel()

    err = np.concatenate((px_err, loc_err, ori_err))
    if isinstance(huber_coef, (tuple, list)):
        huber_coef = np.array([huber_coef[0]] * len(px_err)
                              + [huber_coef[1]] * len(loc_err)
                              + [huber_coef[2]] * len(ori_err))

    return tools.pseudo_huber_loss(huber_coef, err) if huber_coef is not False else (err ** 2)


def _bundle_adjustment_sparsity(n_cams, n_pts, cam_idxs, pt3d_idxs, meas_idxs, poses_only):
    # error term count  (first 2d reprojection errors, then 3d gps measurement error, then 3d orientation meas err)
    m1, m2, m3 = cam_idxs.size * 2, meas_idxs.size * 3, meas_idxs.size * 3
    m = m1 + m2 + m3

    # parameter count (6d poses, n x 3d keypoint locations, 1d time offset)
    n1 = n_cams * 6
    n2 = (0 if poses_only else n_pts * 3)
    n3 = meas_idxs.size * 1
    n = n1 + n2 + n3

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

    # orientation components affect all ori measurement error terms
    # for s in range(3):
    #     A[m1 + m2 + 3 * i + s, meas_idxs * 6 + s] = 1
    for s in range(3):
        for r in range(3):
            A[m1 + 3 + 6 * i + s, meas_idxs * 6 + r] = 1

    if 0:
        import matplotlib.pyplot as plt
        plt.imshow(A.toarray())
        plt.show()

    return A


def _optimized_params(params, n_cams, n_pts, n_meas):
    """
    Retrieve camera parameters and 3-D coordinates.
    """
    a = n_cams * 6
    b = a + n_pts * 3
    c = b + n_meas * 1

    cam_params = params[:a].reshape((n_cams, 6))
    pts3d = params[a:b].reshape((n_pts, 3))
    t_off = params[b:c].reshape((n_meas,))

    return cam_params, pts3d, t_off
