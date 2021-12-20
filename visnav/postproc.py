import argparse
import pickle
import os
import math

import numpy as np
import quaternion
import matplotlib.pyplot as plt

from kapture.io.csv import kapture_from_dir
from kapture.io.features import image_keypoints_from_file

from visnav.algo import tools
from visnav.algo.model import Camera
from visnav.algo.odo.base import Keypoint
from visnav.algo.odo.problem import Problem
from visnav.algo.odo.rootba import RootBundleAdjuster, LinearizerQR
from visnav.algo.odo.vis_gps_bundleadj import vis_gps_bundle_adj
from visnav.algo.tools import Pose
from visnav.depthmaps import get_cam_params
from visnav.missions.nokia import NokiaSensor
from visnav.run import plot_results

DEBUG = 0
PX_ERR_SD = 1.0
LOC_ERR_SD = (3.0, 10.0, 3.0) if 1 else (np.inf,)
ORI_ERR_SD = (math.radians(1.0) if 0 else np.inf,)
HUBER_COEFS = (1.0, 5.0, 0.5)
SENSOR_NAME = 'cam'
FEATURE_NAME = 'gftt'


# TODO: the following:
#  - fix JxJa
#  - use drone velocity measurement to optimize imu-cam delta-time
#  - how to freeze np first poses? or, how to add a prior?
#  - try https://pythonhosted.org/sppy/ to speed up sparse matrix manipulations,
#    maybe https://pypi.org/project/minieigen/ helps also?


def main():
    parser = argparse.ArgumentParser(description='Run global BA on a flight')
    parser.add_argument('--path', nargs='+', required=True, help='path to folder with result.pickle and kapture-folder')
    parser.add_argument('--img-sc', default=0.5, type=float, help='image scale')
    parser.add_argument('--nadir-looking', action='store_true', help='is cam looking down? used for plots only')
    parser.add_argument('--plot', action='store_true', help='plot result')
    parser.add_argument('--ini-fl', type=float, help='initial value for focal length')
    parser.add_argument('--ini-cx', type=float, help='initial value for x principal point')
    parser.add_argument('--ini-cy', type=float, help='initial value for y principal point')
    parser.add_argument('--ini-dist', nargs='+', default=[], type=float, help='initial values for radial distortion coefs')
    parser.add_argument('--fix-fl', action='store_true', help='do not optimize focal length')
    parser.add_argument('--fix-pp', action='store_true', help='do not optimize principal point')
    parser.add_argument('--fix-dist', action='store_true', help='do not optimize distortion coefs')
    parser.add_argument('--abs-loc-r', action='store_true', help='use absolute location measures in BA')
    parser.add_argument('--abs-ori-r', action='store_true', help='use absolute orientation measure in BA')
    args = parser.parse_args()

    assert args.fix_fl and args.fix_pp and args.fix_dist and not args.abs_loc_r and not args.abs_ori_r, \
        'not implemented'

    # check https://github.com/NikolausDemmel/rootba/blob/master/src/rootba/bal/solver_options.hpp
    solver = RootBundleAdjuster(
        ini_tr_rad=1e3,
        min_tr_rad=1e-32,
        max_tr_rad=1e16,
        ini_vee=2.0,
        vee_factor=2.0,
        thread_n=1,
        max_iters=30,
        max_time=300,   # in sec
        min_step_quality=0,
        xtol=0,
        rtol=0,
        ftol=1e-5,

        jacobi_scaling_eps=0,
        lin_cg_maxiter=500,
        lin_cg_tol=1e-5,
        preconditioner_type=LinearizerQR.PRECONDITIONER_TYPE_SCHUR_JACOBI,

        huber_coefs=HUBER_COEFS,
        use_weighted_residuals=False,
    )

    for i, path in enumerate(args.path):
        with open(os.path.join(path, 'result.pickle'), 'rb') as fh:
            results, map3d, frame_names, meta_names, gt, ba_errs = pickle.load(fh)

        kapt = kapture_from_dir(os.path.join(path, 'kapture'))
        sensor_id, width, height, fl_x, fl_y, pp_x, pp_y, *dist_coefs = get_cam_params(kapt, SENSOR_NAME)

        if args.ini_fl:
            fl_x = fl_y = args.ini_fl * args.img_sc
        else:
            fl_x = fl_y = (fl_x + fl_y) / 2

        if args.ini_cx:
            pp_x = args.ini_cx * args.img_sc

        if args.ini_cy:
            pp_y = args.ini_cy * args.img_sc

        if len(args.ini_dist):
            dist_coefs = args.ini_dist
        dist_n = np.where(np.array(dist_coefs) != 0)[0][-1] + 1
        dist_coefs = dist_coefs[:dist_n]

        cam_params = (fl_x, fl_y, pp_x, pp_y, *dist_coefs)
        cam_param_idxs = np.array([0, 2, 3, 4, 5], dtype=int)    # what params to optimize

        print('loading poses and keypoints...')
        poses, pts3d, pts2d, pose_idxs, pt3d_idxs, meas_r, meas_aa, meas_idxs, _ = \
            get_ba_params(path, results, kapt, sensor_id)

        _meas_r = meas_r if 1 else None   # enable/disable loc measurements
        _meas_aa = meas_aa if 0 else None  # enable/disable ori measurements
        _meas_idxs = None if meas_r is None and meas_aa is None else meas_idxs

        problem = Problem(pts2d, cam_params, cam_param_idxs, poses, pose_idxs, pts3d, pt3d_idxs, _meas_r, _meas_aa,
                          _meas_idxs, PX_ERR_SD, LOC_ERR_SD, ORI_ERR_SD, dtype=np.float64)

        if 0:
            J = problem.jacobian()
            r = problem.residual()

            a0, K, loc_err_sd = np.array([]), np.array([[fl_x, 0, pp_x], [0, fl_y, pp_y], [0, 0, 1]]), np.array(LOC_ERR_SD)
            rref, Jref = vis_gps_bundle_adj(poses, pts3d, pts2d, a0, pose_idxs, pt3d_idxs, K, dist_coefs,
                                            np.array([PX_ERR_SD]), meas_r, meas_aa, a0, meas_idxs, loc_err_sd, np.inf,
                                            max_nfev=10, skip_pose_n=0, huber_coef=HUBER_COEFS, poses_only=False,
                                            just_return_r_J=True)

            J = J.toarray()
            Jref = Jref.toarray()

            plt.figure(1)
            plt.plot(r, label='new')
            plt.plot(rref, label='ref')
            plt.legend()

            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            axs[0].imshow(np.sign(J) * np.abs(J) ** (1 / 8))
            axs[0].set_title('new')
            axs[1].imshow(np.sign(Jref) * np.abs(Jref) ** (1 / 8))
            axs[1].set_title('ref')
            plt.show()

            exit()

        solver.solve(problem)

        cam_params, poses, pts3d = map(lambda x: getattr(problem, x), ('cam_params', 'poses', 'pts3d'))

        results = results[:len(poses)]
        for j in range(len(results)):
            results[j][0].post = Pose(poses[j, 3:], tools.angleaxis_to_q(poses[j, :3]))
        map3d = [Keypoint(pt3d=pts3d[j, :]) for j in range(len(pts3d))]

        with open(os.path.join(path, 'global-ba-result.pickle'), 'wb') as fh:
            pickle.dump((results, map3d, frame_names, meta_names, gt, ba_errs), fh)

        fl_x, fl_y, pp_x, pp_y, *dist_coefs = cam_params
        print('new fl: %s, cx: %s, cy: %s, k1, k2: %s' % (
            fl_x / args.img_sc, pp_x / args.img_sc, pp_y / args.img_sc, dist_coefs))

        if args.plot:
            plot_results(results, map3d, frame_names, meta_names, nadir_looking=args.nadir_looking)


def get_ba_params(path, results, kapt, sensor_id):
    frames = [(id, fname[sensor_id]) for id, fname in kapt.records_camera.items()]
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
