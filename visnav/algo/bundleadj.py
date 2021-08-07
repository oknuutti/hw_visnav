"""
Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""
import sys
import logging

import numpy as np

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from visnav.algo import tools


def mp_bundle_adj(arg_queue, log_queue, res_queue):
    while True:
        cmd, args, kwargs = arg_queue.get()
        if cmd == 'ba':
            res_queue.put(vis_bundle_adj(*args, log_writer=LogWriter(log_queue), **kwargs))
        else:
            break


def vis_bundle_adj(poses: np.ndarray, pts3d: np.ndarray, pts2d: np.ndarray,
                   cam_idxs: np.ndarray, pt3d_idxs: np.ndarray, K: np.ndarray, log_writer=None,
                   max_nfev=None, skip_pose0=False, poses_only=False, huber_coef=False):
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

    a = 1 if skip_pose0 else 0
    #assert not skip_pose0, 'some bug with skipping first pose optimization => for some reason cost stays high, maybe problem with A?'

    n_cams = poses.shape[0]
    n_pts = pts3d.shape[0]
    A = _bundle_adjustment_sparsity(n_cams-a, n_pts, cam_idxs, pt3d_idxs, poses_only)   # n_cams-a or n_cams?
    if poses_only:
        x0 = poses[a:].ravel()
        fixed_pt3d = pts3d
    else:
        x0 = np.hstack((poses[a:].ravel(), pts3d.ravel()))
        fixed_pt3d = np.zeros((0, 3))

    pose0 = poses[0:a].ravel()

    if False:
        err = _costfun(x0, pose0, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K)
        print('ERR: %.4e' % (np.sum(err**2)/2))

    tmp = sys.stdout
    sys.stdout = log_writer or LogWriter()
    res = least_squares(_costfun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, xtol=1e-5, method='trf',
                        # for some reason doesnt work as well as own huber loss
                        # tr_solver='lsmr', # loss='huber', f_scale=huber_coef
                        args=(pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K, huber_coef),
                        max_nfev=max_nfev)
    sys.stdout = tmp

    new_poses, new_pts3d = _optimized_params(res.x, n_cams-a, 0 if poses_only else n_pts)
    return new_poses, new_pts3d


class LogWriter:
    def __init__(self, log_queue=None):
        self.log_queue = log_queue

    def write(self, msg):
        if msg.strip() != '':
            if self.log_queue is None:
                logging.info(msg)
            else:
                self.log_queue.put(('info', msg))


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

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def _project(pts3d, poses, K):
    """Convert 3-D points to 2-D by projecting onto images."""
    pts2d_proj = _rotate(pts3d, poses[:, :3])
    pts2d_proj += poses[:, 3:6]

    # pts2d_proj = -pts2d_proj[:, :2] / pts2d_proj[:, 2, np.newaxis]
    # f = poses[:, 6]
    # k1 = poses[:, 7]
    # k2 = poses[:, 8]
    # n = np.sum(pts2d_proj ** 2, axis=1)
    # r = 1 + k1 * n + k2 * n ** 2
    # pts2d_proj *= (r * f)[:, np.newaxis]

    pts2d_proj = K.dot(pts2d_proj.T).T                              # own addition
    pts2d_proj = pts2d_proj[:, :2] / pts2d_proj[:, 2, np.newaxis]   # own addition

    return pts2d_proj


def _costfun(params, pose0, fixed_pt3d, n_cams, n_pts, cam_idxs, pt3d_idxs, pts2d, K, huber_coef):
    """
    Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    params = np.hstack((pose0, params))
    poses = params[:n_cams * 6].reshape((n_cams, 6))
    points_3d = fixed_pt3d if len(fixed_pt3d)>0 else params[n_cams * 6:].reshape((n_pts, 3))
    points_proj = _project(points_3d[pt3d_idxs], poses[cam_idxs], K)
    err = (points_proj - pts2d).ravel()

    return tools.pseudo_huber_loss(err, huber_coef) if huber_coef else (err ** 2)


def _bundle_adjustment_sparsity(n_cams, n_pts, cam_idxs, pt3d_idxs, poses_only):
    m = cam_idxs.size * 2
    n = n_cams * 6 + (0 if poses_only else n_pts * 3)
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(cam_idxs.size)
    for s in range(6):
        A[2 * i, cam_idxs * 6 + s] = 1
        A[2 * i + 1, cam_idxs * 6 + s] = 1

    if not poses_only:
        for s in range(3):
            A[2 * i, n_cams * 6 + pt3d_idxs * 3 + s] = 1
            A[2 * i + 1, n_cams * 6 + pt3d_idxs * 3 + s] = 1

    return A


def _optimized_params(params, n_cams, n_pts):
    """
    Retrieve camera parameters and 3-D coordinates.
    """
    cam_params = params[:n_cams * 6].reshape((n_cams, 6))
    pts3d = params[n_cams * 6:].reshape((n_pts, 3))

    return cam_params, pts3d
