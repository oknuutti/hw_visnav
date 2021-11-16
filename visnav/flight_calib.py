import argparse
import pickle
import os

import numpy as np
import quaternion
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from kapture.io.csv import kapture_from_dir
from kapture.io.features import image_keypoints_from_file

from visnav.algo import tools
from visnav.algo.model import Camera
from visnav.algo.odo.vis_gps_bundleadj import vis_gps_bundle_adj
from visnav.algo.tools import Pose
from visnav.depthmaps import get_cam_params
from visnav.missions.nokia import NokiaSensor


DEBUG = 0
PX_ERR_SD = 1.0
LOC_ERR_SD = (2.0, 3.0, 2.0)
SENSOR_NAME = 'cam'
FEATURE_NAME = 'gftt'
TAKEOFF_LAWN_ALT = 0.6  # in meters

(
    ERROR_TYPE_ALTITUDE,
    ERROR_TYPE_CURVATURE,
    ERROR_TYPE_REPROJECTION,
    ERROR_TYPE_DISPERSION,
    ERROR_TYPE_PITCH,
) = ERROR_TYPES = range(5)

ERROR_COEFS = dict(zip(ERROR_TYPES, (
    1,
    100,
    1,
    1,
    1,
)))


def main():
    parser = argparse.ArgumentParser(description='Estimate focal length and simple radial distortion based on '
                                                 'flat ground at the starting location')
    parser.add_argument('--path', nargs='+', required=True, help='path to folder with result.pickle and kapture-folder')
    parser.add_argument('--takeoff', nargs='+', type=int, required=True, help='take-off trajectory in question?')
    parser.add_argument('--first-frame', '-f', nargs='+', type=int, required=True, help='first frame')
    parser.add_argument('--last-frame', '-l', nargs='+', type=int, required=True, help='last frame')
    parser.add_argument('--img-sc', default=0.5, type=float, help='image scale')
    parser.add_argument('--ini-fl', type=float, help='initial value for focal length')
    parser.add_argument('--ini-k1', type=float, help='initial value for radial distortion coef k1')
    parser.add_argument('--fix-fl', action='store_true', help='do not optimize focal length')
    parser.add_argument('--fix-k1', action='store_true', help='do not optimize radial distortion coef k1')
    parser.add_argument('--opt-disp', action='store_true', help='use dispersion around estimated surface as a measure')
    parser.add_argument('--opt-pitch', action='store_true', help='expect pitch to not change')
    args = parser.parse_args()

    cf_args = []
    for i, path in enumerate(args.path):
        with open(os.path.join(path, 'result.pickle'), 'rb') as fh:
            results, *_ = pickle.load(fh)

        kapt = kapture_from_dir(os.path.join(path, 'kapture'))
        sensor_id, width, height, fl_x, fl_y, pp_x, pp_y, *dist_coefs = get_cam_params(kapt, SENSOR_NAME)
        x0 = (args.ini_fl * args.img_sc if args.ini_fl else (fl_x + fl_y) / 2, args.ini_k1 or dist_coefs[0])

        print('loading poses and keypoints...')
        poses, pts3d, pts2d, cam_idxs, pt3d_idxs, meas_r, meas_aa, meas_idxs, obs_kp = \
            get_ba_params(path, args.first_frame[i], args.last_frame[i], results, kapt, sensor_id)

        error_types = {ERROR_TYPE_REPROJECTION}
        if args.takeoff[i]:
            error_types.update({ERROR_TYPE_ALTITUDE, ERROR_TYPE_CURVATURE})
            if args.opt_disp:
                error_types.add(ERROR_TYPE_DISPERSION)
        elif args.opt_pitch:
            error_types.add(ERROR_TYPE_PITCH)

        cf_args.append((width, height, pp_x, pp_y, poses, pts3d, pts2d, cam_idxs, pt3d_idxs,
                        meas_r, meas_aa, meas_idxs, obs_kp, error_types,
                        x0[0] if args.fix_fl else None, x0[1] if args.fix_k1 else None))

    print('optimizing focal length and radial distortion coef k1...')
    if 0:
        # optimize focal length, simple radial distortion using gaussian process hyperparameter search
        # TODO
        pass
    elif not args.fix_fl and not args.fix_k1:
        res = least_squares(costfun, x0, verbose=2, ftol=1e-5, xtol=1e-5, gtol=1e-8, max_nfev=1000,
                            bounds=((0.7 * x0[0], -0.12), (1.4 * x0[0], 0.12)), diff_step=(0.0003, 0.001),  # x_scale=(1, 0.002),
                            args=(cf_args,))
    elif args.fix_fl:
        res = least_squares(costfun, x0[1:], verbose=2, ftol=1e-5, xtol=1e-5, gtol=1e-8, max_nfev=1000,
                            bounds=(-0.12, 0.12), diff_step=0.001,  # x_scale=(1, 0.002),
                            args=(cf_args,))
    elif args.fix_k1:
        res = least_squares(costfun, x0[0:1], verbose=2, ftol=1e-5, xtol=1e-5, gtol=1e-8, max_nfev=1000,
                            bounds=(0.7 * x0[0], 1.4 * x0[0]), diff_step=0.0003,  # x_scale=(1, 0.002),
                            args=(cf_args,))

    costfun(res.x, cf_args, plot=True)


def costfun(x, cf_args, plot=False):
    errs = []
    for cf_arg in cf_args:
        errs.append(traj_costfun(x, *cf_arg, plot=plot))
    return np.concatenate(errs)


def traj_costfun(x, width, height, pp_x, pp_y, poses, pts3d, pts2d, cam_idxs, pt3d_idxs,
                 meas_r, meas_aa, meas_idxs, obs_kp, error_types, def_fl=None, def_k1=None, plot=False):
    fl, k1 = (abs(x[0]), x[1]) if len(x) > 1 else ((def_fl, x[0]) if def_fl else (x[0], def_k1))

    cam = Camera(width, height, cam_mx=np.array([[fl, 0, pp_x], [0, fl, pp_y], [0, 0, 1]]),
                 dist_coefs=np.array([k1, 0, 0, 0, 0], dtype=np.float32))
    norm_pts2d = cam.undistort(pts2d).squeeze()

    # run ba
    new_poses, new_pts3d, _, ba_errs = vis_gps_bundle_adj(
        poses, pts3d, norm_pts2d, np.zeros((0, 1)), cam_idxs, pt3d_idxs, cam.intrinsic_camera_mx(), np.array([PX_ERR_SD]),
        meas_r, meas_aa, np.zeros((0, 1)), meas_idxs, np.array([LOC_ERR_SD]), np.array([np.inf]), px_err_weight=np.array([1]),
        log_writer=None, max_nfev=1000, skip_pose_n=1, poses_only=False, huber_coef=(1, 5, 0.5))

    new_poses = np.concatenate((poses[:1, :], new_poses), axis=0)
    errs = []

    if ERROR_TYPE_REPROJECTION in error_types:
        err_repr = ERROR_COEFS[ERROR_TYPE_REPROJECTION] * np.nanmean(ba_errs[:, 0])
        errs.append(err_repr)

    if {ERROR_TYPE_ALTITUDE, ERROR_TYPE_CURVATURE, ERROR_TYPE_DISPERSION}.intersection(error_types):
        new_pts3d = new_pts3d[obs_kp, :]

        # fit a plane, based on https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
        centroid = np.median(new_pts3d.T, axis=1, keepdims=True)
        svd = np.linalg.svd(new_pts3d.T - centroid)
        normal = svd[0][:, -1:].T

        # fit a parabolic surface to keypoints
        x0 = 0, *centroid.flatten(), *normal.flatten()
        res = least_squares(paraboloid_costfun, x0, verbose=0, ftol=1e-5, xtol=1e-5, gtol=1e-8, max_nfev=1000,
                            loss='huber', f_scale=1.0, args=(new_pts3d,))

        # extract a plane that corresponds to the fitted parabolic surface (should be more accurate than the median based)
        c, x0, y0, z0, xn, yn, zn = res.x

        if ERROR_TYPE_CURVATURE in error_types:
            errs.append(ERROR_COEFS[ERROR_TYPE_CURVATURE] * c)

        if ERROR_TYPE_DISPERSION in error_types:
            displ = np.abs(res.fun)
            lim = np.quantile(displ, 0.8)   # how well the keypoints lie on the estimated surface
            err_disp = np.mean(displ[displ < lim])
            errs.append(ERROR_COEFS[ERROR_TYPE_DISPERSION] * err_disp)

        if ERROR_TYPE_ALTITUDE in error_types:
            centroid = np.array([x0, y0, z0]).reshape((1, 3))
            normal = np.array([xn, yn, zn]).reshape((1, 3))
            normal /= np.linalg.norm(normal)

            pose_cf = new_poses[meas_idxs[-1], :]
            loc_wf = (-Pose(pose_cf[3:], tools.angleaxis_to_q(pose_cf[:3]))).loc

            end_distance = normal.dot(loc_wf[:, None] - centroid)
            end_meas_alt = -meas_r[-1, 1] - TAKEOFF_LAWN_ALT  # neg y-axis is altitude
            err_alt = (end_distance - end_meas_alt)[0, 0]
            errs.append(ERROR_COEFS[ERROR_TYPE_ALTITUDE] * err_alt / end_meas_alt)

    if ERROR_TYPE_PITCH in error_types:
        ypr = np.array([tools.q_to_ypr(tools.angleaxis_to_q(p[:3])) for p in new_poses]) / np.pi * 180
        err_pitch = ypr[-1, 1] - ypr[0, 1]
        errs.append(ERROR_COEFS[ERROR_TYPE_PITCH] * err_pitch)

    print('=== fl: %s, k1: %s, cost: %s ===' % (fl / 0.5, k1, errs))

    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(new_pts3d[:, 0], new_pts3d[:, 1], new_pts3d[:, 2], 'C0.')
        plt.show()

    return np.array(errs)


def paraboloid_costfun(x, points):
    c, x0, y0, z0, xn, yn, zn = x
    center = np.array([x0, y0, z0]).reshape((1, 3))
    normal = np.array([xn, yn, zn]).reshape((1, 3))
    normal /= np.linalg.norm(normal)

    # calculate altitude relative to plane
    points_alt = normal.dot(points.T - center.T).T

    # project to plane
    points_proj = points - center - points_alt * normal

    # calc paraboloid altitude at each point
    parab_alt = c * np.sum(points_proj ** 2, axis=1)

    if DEBUG:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        parab_surf = points_proj + parab_alt[:, None] * normal
        ax.plot(parab_surf[:, 0], parab_surf[:, 1], parab_surf[:, 2], 'C3.')
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'C0.')
        plt.show()

    return parab_alt - points_alt.flatten()


def get_ba_params(path, ff, lf, results, kapt, sensor_id):
    frames = [(id, fname[sensor_id]) for id, fname in kapt.records_camera.items()
              if id >= ff and (lf is None or id < lf)]
    fname2id = {fname: id for id, fname in frames}

    poses = np.array([[*tools.q_to_angleaxis(kapt.trajectories[id][sensor_id].r, True),
                       *kapt.trajectories[id][sensor_id].t] for id, fname in frames]).astype(float)

    pts3d = kapt.points3d[:, :3]

    feat = kapt.keypoints[FEATURE_NAME]
    uv_map = {}
    for id_f, fname in frames:
        uvs = image_keypoints_from_file(
            os.path.join(path, 'kapture', 'reconstruction', 'keypoints', FEATURE_NAME, fname + '.kpt'),
            feat.dtype, feat.dsize)
        uv_map[id_f] = uvs

    f_uv = {}
    for id3, r in kapt.observations.items():
        for fname, id2 in r[FEATURE_NAME]:
            if fname in fname2id:
                id_f = fname2id[fname]
                f_uv.setdefault(id_f, {})[id3] = uv_map[id_f][id2, :]

    # f_uv = {id_f: {id3: uv_map[id_f][id2, :]
    #                     for id3 in range(len(pts3d))
    #                         for id2 in range(len(uv_map[id_f]))
    #                             if (fname, id2) in kapt.observations.get(id3, {}).get(FEATURE_NAME, {})}
    #         for id_f, fname in frames}

    obs_kp = list(set.union(*[set(m.keys()) for m in f_uv.values()]))

    cam_idxs, pt3d_idxs, pts2d = list(map(np.array, zip(*[
        (i, id3, uv.flatten())
        for i, (id_f, kps_uv) in enumerate(f_uv.items())
            for id3, uv in kps_uv.items()
    ])))

    meas_idxs = np.array([i for i, r in enumerate(results) if r[1] is not None and i < len(poses)], dtype=int)
    meas_q = {i: results[i][0].prior.quat.conj() for i in meas_idxs}
    meas_r = np.array([tools.q_times_v(meas_q[i], -results[i][0].prior.loc) for i in meas_idxs], dtype=np.float32)
    meas_aa = np.array([tools.q_to_angleaxis(meas_q[i], compact=True) for i in meas_idxs], dtype=np.float32)

    return poses, pts3d, pts2d, cam_idxs, pt3d_idxs, meas_r, meas_aa, meas_idxs, obs_kp


if __name__ == '__main__':
    main()
