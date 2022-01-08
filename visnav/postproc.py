import argparse
import pickle
import os
import math

import cv2
import numpy as np
import quaternion
from scipy.spatial.ckdtree import cKDTree

import matplotlib.pyplot as plt
from tqdm import tqdm

import kapture as kt
from kapture.io.csv import kapture_from_dir, kapture_to_dir
from kapture.io.features import get_keypoints_fullpath, image_keypoints_from_file, image_keypoints_to_file, \
    get_descriptors_fullpath, image_descriptors_from_file, image_descriptors_to_file

from visnav.algo import tools
from visnav.algo.featdet import detect_gridded
from visnav.algo.model import Camera
from visnav.algo.odo.base import Keypoint
from visnav.algo.odo.problem import Problem
from visnav.algo.odo.rootba import RootBundleAdjuster, LinearizerQR
from visnav.algo.odo.vis_gps_bundleadj import vis_gps_bundle_adj
from visnav.algo.tools import Pose
from visnav.depthmaps import get_cam_params, set_cam_params
from visnav.missions.nokia import NokiaSensor
from visnav.run import plot_results

DEBUG = 0
PX_ERR_SD = 1.0
LOC_ERR_SD = (3.0, 10.0, 3.0) if 1 else (np.inf,)
ORI_ERR_SD = (math.radians(1.0) if 0 else np.inf,)
HUBER_COEFS = (1.0, 5.0, 0.5)
SENSOR_NAME = 'cam'
VO_FEATURE_NAME = 'gftt'
EXTRACTED_FEATURE_NAME = 'akaze'
EXTRACTED_FEATURE_COUNT = 2000
EXTRACTED_FEATURE_GRID = (3, 3)
EXTRACTED_FEATURE_PARAMS = {'akaze': {
    'descriptor_type': cv2.AKAZE_DESCRIPTOR_MLDB,  # default: cv2.AKAZE_DESCRIPTOR_MLDB
    'descriptor_channels': 3,  # default: 3
    'descriptor_size': 0,  # default: 0
    'diffusivity': cv2.KAZE_DIFF_CHARBONNIER,  # default: cv2.KAZE_DIFF_PM_G2
    'threshold': 0.00005,  # default: 0.001
    'nOctaves': 4,  # default: 4
    'nOctaveLayers': 4,  # default: 4
}}
MAX_REPR_ERROR = 8
MIN_INLIERS = 15


logger = tools.get_logger("main")


# TODO: the following:
#  - debug feature extraction
#  - debug feature matching across flights
#  - (5) implement truly global ba by joining given flights
#  - debug global ba
#  - (6) implement image normalization and cam idealization for depth map estimation (here or at depthmaps.py?)
#  /- how to freeze np first poses? or, how to add a prior?
#  /- try https://pythonhosted.org/sppy/ to speed up sparse matrix manipulations,
#  /  maybe https://pypi.org/project/minieigen/ helps also?


def main():
    parser = argparse.ArgumentParser(description='Run global BA on a flight')
    parser.add_argument('--path', nargs='+', required=True, help='path to folder with result.pickle and kapture-folder')
    parser.add_argument('--matches-path', type=str, help='path to across-flight feature match file')
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
    parser.add_argument('--skip-fe', action='store_true', help='Skip AKAZE feature extraction')
    parser.add_argument('--skip-fm', action='store_true', help='Skip feature matching across flights')
    parser.add_argument('--skip-ba', action='store_true', help='Skip global BA')
    parser.add_argument('--normalize-images', action='store_true',
                        help='After all is done, normalize images based on current camera params')
    parser.add_argument('--fe-n', type=int, default=10, help='Extract AKAZE features every n frames')
    args = parser.parse_args()

    assert args.fix_fl and args.fix_pp and args.fix_dist and not args.abs_loc_r and not args.abs_ori_r, \
        'not implemented'

    if not args.skip_fe:
        run_fe(args)

    if not args.skip_fm:
        assert args.matches_path, 'no feature match file path given'
        run_fm(args)

    if not args.skip_ba:
        run_ba(args)

    if args.normalize_images:
        pass
        # TODO: (6) normalize images (and idealize cam params) so that can continue with depth estimation using depthmaps.py


def run_fe(args):
    for i, path in enumerate(args.path):
        kapt_path = os.path.join(path, 'kapture')
        kapt = kapture_from_dir(kapt_path)
        sensor_id, width, height, fl_x, fl_y, pp_x, pp_y, *dist_coefs = cam_params = get_cam_params(kapt, SENSOR_NAME)

        if EXTRACTED_FEATURE_NAME in kapt.keypoints:
            # remove all previous features of same type
            for img_file in kapt.keypoints[EXTRACTED_FEATURE_NAME]:
                os.unlink(get_keypoints_fullpath(EXTRACTED_FEATURE_NAME, kapt_path, img_file))
                os.unlink(get_descriptors_fullpath(EXTRACTED_FEATURE_NAME, kapt_path, img_file))
        kapt.keypoints[EXTRACTED_FEATURE_NAME] = kt.Keypoints(EXTRACTED_FEATURE_NAME, np.float32, 2)

        logger.info('Extracting %s-features from every %dth frame' % (EXTRACTED_FEATURE_NAME, args.fe_n))
        records = list(kapt.records_camera.values())
        for j in tqdm(range(0, len(records) - args.fe_triang_int, args.fe_n)):
            d = []
            for k in (0, args.fe_triang_int):
                img_file = records[j+k][sensor_id]
                img_path = os.path.join(kapt_path, 'sensors', 'records_data', img_file)
                kps, descs = extract_features(img_path, args)
                d.append((img_file, kps, descs))

            # match features based on descriptors, triangulate matches, filter out bad ones
            pts3d, idx1, idx2 = triangulate(kapt, cam_params, *d[0], *d[1])

            if pts3d is not None:
                # write 3d points to kapture
                pt3d_id_start = len(kapt.points3d)
                kapt.points3d = kt.Points3d(np.concatenate((kapt.points3d,
                    np.concatenate((pts3d, np.ones_like(pts3d) * 128), axis=1)
                ), axis=0))

                for (img_file, kps, descs), idx in zip(d, (idx1, idx2)):
                    # write keypoints and descriptors to kapture
                    kapt.keypoints[EXTRACTED_FEATURE_NAME].add(img_file)
                    image_keypoints_to_file(get_keypoints_fullpath(EXTRACTED_FEATURE_NAME, kapt_path, img_file),
                                            kps[idx, :])
                    image_descriptors_to_file(get_descriptors_fullpath(EXTRACTED_FEATURE_NAME, kapt_path, img_file),
                                              descs[idx, :])

                    # write observations to kapture
                    for kp_id, uv in enumerate(kps[idx, :]):
                        k.observations.add(pt3d_id_start + kp_id, EXTRACTED_FEATURE_NAME, img_file, idx)

        kapture_to_dir(kapt_path, kapt)


def run_fm(args):
    n = len(args.path)
    if n <= 1:
        return

    # Example values:
    #   flight-id = 'data/output/27'
    #   frame-id = 6   (for image cam/frame000006.jpg)
    #   feature-id: 4  (4th feature extracted for given frame)
    cam_params = {}  # {flight-id: cam_params, ...}
    poses = {}  # {(flight-id, frame-id): pose, ...}
    pts2d = {}  # {(flight-id, frame-id): [2d-point, ...], ...}       # feature-id is the index of the inner list
    descr = {}  # {(flight-id, frame-id): [descr, ...], ...}          # feature-id is the index of the inner list
    obser = {}  # {(flight-id, frame-id, feature-id): 3d-point, ...}

    res_pts3d = []  # [[(flight-id, pt3d-id), ...], [3d-point, [(frame-id, feature-id), ...]]...}
    res_obser = []  # [(flight-id, frame-id, feature-id, pt3d-id), ...]  # pt3d-id is the index of list res_pts3d

    k = 0
    for i, path1 in enumerate(args.path):
        for j, path2 in zip(range(i + 1, len(args.path)), args.path[i+1:]):
            k += 1
            logger.info('Finding feature matches and doing initial triangulation for %s and %s (%d/%d)' % (
                path1, path2, k, n*(n-1)/2))
            find_matches(path1, path2, cam_params, poses, pts2d, descr, obser, res_pts3d, res_obser)

    # calculate mean of each entry in res_pts3d
    res_pts3d = [np.mean(np.array(pts3d), axis=0) for pts3d in res_pts3d]

    # save res_pts3d, res_obser so that can use for global ba
    with open(args.matches_path, 'wb') as fh:
        pickle.dump((res_pts3d, res_obser), fh)


def find_matches(path1, path2, cam_params, poses, pts2d, descr, obser, res_pts3d, res_obser):
    # find all frames with akaze features, load poses and observations
    for path in (path1, path2):
        if path in cam_params:
            continue

        kapt = kapture_from_dir(os.path.join(path, 'kapture'))
        cam_params[path] = get_cam_params(kapt, SENSOR_NAME)

        sensor_id, cam = cam_params[path][0], cam_obj(cam_params[path])
        img2fid = dict([(fname[sensor_id], id) for id, fname in kapt.records_camera.items()])
        ftype = kapt.keypoints[EXTRACTED_FEATURE_NAME]

        for img_file in kapt.keypoints[EXTRACTED_FEATURE_NAME]:
            frame_id = img2fid[img_file]
            traj = kapt.trajectories[(frame_id, sensor_id)]
            poses[(path, frame_id)] = Pose(traj.t, traj.r)
            pts2d[(path, frame_id)] = cam.undistort(image_keypoints_from_file(
                get_keypoints_fullpath(EXTRACTED_FEATURE_NAME, path, img_file), ftype.dtype, ftype.dsize))
            descr[(path, frame_id)] = image_descriptors_from_file(
                get_descriptors_fullpath(EXTRACTED_FEATURE_NAME, path, img_file), ftype.dtype, ftype.dsize)

            # {feat_id: pt3d_id, ...}
            feat_to_pt3d = {t[EXTRACTED_FEATURE_NAME][1]: pt3d_id
                            for pt3d_id, t in kapt.observations.items()
                            if EXTRACTED_FEATURE_NAME in t}
            obser.update({(path, frame_id, feat_id): kapt.points3d[pt3d_id]
                          for feat_id, pt3d_id in feat_to_pt3d.items()})

    # TODO: in some distant future, find closest pose based on GFTT/AKAZE locations instead of drone location
    #       (similar to what is done in navex/datasets/preproc)
    # for each pose in flight1, find closest pose from flight2 (based on location only)
    fids1, locs1 = zip(*[(fid, pose.loc) for (path, fid), pose in poses.items() if path == path1])
    fids2, locs2 = zip(*[(fid, pose.loc) for (path, fid), pose in poses.items() if path == path2])
    tree = cKDTree(np.array(locs2))
    d, idxs2 = tree.query(locs1)

    # loop through matched frames
    for fid1, idx2 in zip(fids1, idxs2):
        if idx2 < len(fids2):
            # match features, verify with 3d-2d-ransac both ways, add 3d points to list
            d = []
            for path, fid in zip((path1, path2), (fid1, fids2[idx2])):
                cam = cam_obj(cam_params[path])
                pts3d = [obser[(path, fid, i)] for i in range(len(pts2d[(path, fid)]))]
                d.append((cam.cam_mx, pts2d[(path, fid)], descr[(path, fid)], pts3d))

            matches1, matches2 = match_and_validate(*d[0], *d[1])

            if matches1 is not None:
                for path, fid, feat_ids in zip((path1, path2), (fid1, fids2[idx2]), (matches1, matches2)):
                    for feat_id in feat_ids:
                        key = (path, fid, feat_id)
                        if key in res_obser:
                            pt3d_id = res_obser[key]
                            res_pts3d[pt3d_id].append(obser[key])
                        else:
                            pt3d_id = len(res_pts3d)
                            res_pts3d.append([obser[key]])
                        res_obser[key] = pt3d_id


def run_ba(args):
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

    # TODO: (5) update so that also loads AKAZE features, joins separate flights before running truly global BA
    for i, path in enumerate(args.path):
        with open(os.path.join(path, 'result.pickle'), 'rb') as fh:
            results, map3d, frame_names, meta_names, gt, ba_errs = pickle.load(fh)

        kapt_path = os.path.join(path, 'kapture')
        kapt = kapture_from_dir(kapt_path)
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

        logger.info('loading poses and keypoints...')
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
        poses = [Pose(poses[j, 3:], tools.angleaxis_to_q(poses[j, :3])) for j in range(len(results))]
        for j, pose in enumerate(poses):
            results[j][0].post = pose
        map3d = [Keypoint(pt3d=pts3d[j, :]) for j in range(len(pts3d))]

        with open(os.path.join(path, 'global-ba-result.pickle'), 'wb') as fh:
            pickle.dump((results, map3d, frame_names, meta_names, gt, ba_errs), fh)

        fl_x, fl_y, pp_x, pp_y, *dist_coefs = cam_params
        logger.info('new fl: %s, cx: %s, cy: %s, k1, k2: %s' % (
            fl_x / args.img_sc, pp_x / args.img_sc, pp_y / args.img_sc, dist_coefs))

        update_kapture(kapt_path, kapt, [width, height] + list(cam_params), poses, pts3d)

        if args.plot:
            plot_results(results, map3d, frame_names, meta_names, nadir_looking=args.nadir_looking)


def get_ba_params(path, results, kapt, sensor_id):
    frames = [(id, fname[sensor_id]) for id, fname in kapt.records_camera.items()]
    fname2id = {fname: id for id, fname in frames}

    poses = np.array([[*tools.q_to_angleaxis(kapt.trajectories[id][sensor_id].r, True),
                       *kapt.trajectories[id][sensor_id].t] for id, fname in frames]).astype(float)

    pts3d = kapt.points3d[:, :3]
    feat = kapt.keypoints[VO_FEATURE_NAME]
    uv_map = {}
    for id_f, fname in frames:
        uvs = image_keypoints_from_file(get_keypoints_fullpath(VO_FEATURE_NAME, path, fname), feat.dtype, feat.dsize)
        uv_map[id_f] = uvs

    f_uv = {}
    for id3, r in kapt.observations.items():
        for fname, id2 in r[VO_FEATURE_NAME]:
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


def update_kapture(kapt_path, kapt, cam_params, poses, pts3d):
    cam_id = set_cam_params(kapt, SENSOR_NAME, cam_params)

    for frame_num, pose in enumerate(poses):
        kapt.trajectories[(frame_num, cam_id)] = kt.PoseTransform(r=pose.quat.components, t=pose.loc)

    kapt.points3d = kt.Points3d(np.concatenate((pts3d, np.ones_like(pts3d)*128), axis=1))

    kapture_to_dir(kapt_path, kapt)


def extract_features(img_path, args):
    if EXTRACTED_FEATURE_NAME == 'akaze':
        det = cv2.AKAZE_create(**EXTRACTED_FEATURE_PARAMS[EXTRACTED_FEATURE_NAME])
    else:
        assert False, 'feature extractor "%s" not implemented' % EXTRACTED_FEATURE_NAME

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kps = detect_gridded(det, img, None, *EXTRACTED_FEATURE_GRID, EXTRACTED_FEATURE_COUNT)
    kps, descs = det.compute(img, kps)
    kps = np.array([k.pt for k in kps], dtype='f4').reshape((-1, 2))
    return kps, descs


def cam_obj(cam_params):
    sensor_id, width, height, fl_x, fl_y, pp_x, pp_y, *dist_coefs = cam_params
    cam_mx = np.array([[fl_x, 0, pp_x], [0, fl_y, pp_y], [0, 0, 1]])
    cam = Camera(width, height, cam_mx=cam_mx, dist_coefs=dist_coefs)
    return cam


def triangulate(kapt, cam_params, img_file1, kps1, descs1, img_file2, kps2, descs2):

    # match features based on descriptors, cross check for validity, sort keypoints so that indices indicate matches
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, True)
    matches = matcher.match(descs1, descs2)
    kps1 = kps1[[m.queryIdx for m in matches], :]
    kps2 = kps2[[m.trainIdx for m in matches], :]

    # get camera parameters, undistort keypoints
    sensor_id, cam = cam_params[0], cam_obj(cam_params)
    kps1, kps2 = map(lambda x: cam.undistort(x), (kps1, kps2))

    # get camera frame pose matrices
    img2fid = dict([(fname[sensor_id], id) for id, fname in kapt.records_camera.items()])
    pose1, pose2 = map(lambda x: kapt.trajectories[(img2fid[x], sensor_id)], (img_file1, img_file2))
    pose1, pose2 = map(lambda x: Pose(x.t, x.r).to_mx(), (pose1, pose2))

    # triangulate matched keypoints
    kps4d = cv2.triangulatePoints(cam.cam_mx.dot(pose1), cam.cam_mx.dot(pose2),
                                  kps1.reshape((-1, 1, 2)), kps2.reshape((-1, 1, 2)))
    pts3d = (kps4d.T[:, :3] / kps4d.T[:, 3:])[0]

    # filter out bad triangulation results
    mask = np.ones((len(matches),), dtype=bool)
    for pose, kps in zip((pose1, pose2), (kps1, kps2)):

        # check that points are in front of the cam
        kps4d_l = pose.dot(kps4d)
        kps3d_l = kps4d_l[:, :3] / kps4d_l[:, 3:]
        mask = np.logical_and(mask, kps3d_l[:, 2] < 0)

        # check that reprojection error is not too large
        repr_kps = cam.cam_mx.dot(pose).dot(kps4d_l.T).T
        repr_kps = repr_kps[:, :2] / repr_kps[:, 2:]
        repr_err = np.linalg.norm(kps - repr_kps, axis=1)
        mask = np.logical_and(mask, repr_err < MAX_REPR_ERROR)

    if np.sum(mask) < MIN_INLIERS:
        return None, None

    return (pts3d[mask, :], *zip(*[[m.queryIdx, m.trainIdx] for m in matches[mask]]))


def match_and_validate(cam_mx1, kps1, descr1, pts3d1, cam_mx2, kps2, descr2, pts3d2):
    # match features based on descriptors, cross check for validity, sort keypoints so that indices indicate matches
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, True)
    matches = matcher.match(descr1, descr2)
    kps1, pts3d1 = map(lambda x: x[[m.queryIdx for m in matches], :], (kps1, pts3d1))
    kps2, pts3d2 = map(lambda x: x[[m.trainIdx for m in matches], :], (kps2, pts3d2))

    # kps1 and kps2 already undistorted

    # 3d-2d ransac on both, need same features to be inliers
    inliers = set(range(len(kps1)))
    for pts3d, cam_mx, kps in ((pts3d1, pts3d2), (cam_mx2, cam_mx1), (kps2, kps1)):
        ok, rv, r, inl = cv2.solvePnPRansac(pts3d, kps, cam_mx, None, iterationsCount=10000,
                                            reprojectionError=MAX_REPR_ERROR, flags=cv2.SOLVEPNP_AP3P)
        inliers = inliers.intersection(inl)

    if len(inliers) < MIN_INLIERS:
        return None, None

    return (*zip(*[[m.queryIdx, m.trainIdx] for m in matches[inliers]]),)


if __name__ == '__main__':
    main()