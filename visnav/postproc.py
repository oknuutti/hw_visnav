import argparse
import pickle
import os
import math
from typing import List

import cv2
import numpy as np
import quaternion
import scipy
from kapture import Kapture
from kapture.io.records import get_record_fullpath
from scipy.spatial import cKDTree

from tqdm import tqdm
#from memory_profiler import profile

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
from visnav.algo.odo.vis_gps_bundleadj import vis_gps_bundle_adj, numerical_jacobian
from visnav.algo.tools import Pose
from visnav.depthmaps import get_cam_params, set_cam_params
from visnav.missions.nokia import NokiaSensor
from visnav.run import plot_results

DEBUG = 0
PX_ERR_SD = 2  # 0.55   #  0.55    # |(960, 540)| * 0.0005 => 1101 * 0.0005 => 0.55
LOC_ERR_SD = (6.0,) if 1 else (np.inf,)
ORI_ERR_SD = (math.radians(30.0) if 0 else np.inf,)
HUBER_COEFS = None  # (1.0, 5.0, 0.5)
SENSOR_NAME = 'cam'
CNN_MODEL_PATH = os.path.join('data', 'models')
VO_FEATURE_NAME = 'gftt'
EXTRACTED_FEATURE_NAME = 'akaze'
EXTRACTED_FEATURE_COUNT = 3000
EXTRACTED_FEATURE_GRID = (1, 1)
EXTRACTED_FEATURE_PARAMS = {
    'akaze': {
        'descriptor_type': cv2.AKAZE_DESCRIPTOR_MLDB,  # default: cv2.AKAZE_DESCRIPTOR_MLDB
        'descriptor_channels': 3,  # default: 3
        'descriptor_size': 0,   # default: 0
        'diffusivity': cv2.KAZE_DIFF_CHARBONNIER,  # default: cv2.KAZE_DIFF_PM_G2
        'threshold': 0.00005,   # default: 0.001
        'nOctaves': 4,          # default: 4
        'nOctaveLayers': 8,     # default: 4
    },
    'sift': {
        'nfeatures': EXTRACTED_FEATURE_COUNT,
        'nOctaveLayers': 6,         # default: 3
        'contrastThreshold': 0.01,  # default: 0.04
        'edgeThreshold': 25,        # default: 10
        'sigma': 1.6,               # default: 1.6
    },
}
EXTRACTED_FEATURE_PARAMS['rsift'] = EXTRACTED_FEATURE_PARAMS['sift']
MAX_REPR_ERROR = 8
MIN_INLIERS = 15

logger = tools.get_logger("main")


# TODO: the following:
#  - ? find intra batch global keyframe matches
#  /- try https://pythonhosted.org/sppy/ to speed up sparse matrix manipulations,
#  /  maybe https://pypi.org/project/minieigen/ helps also?


def main():
    parser = argparse.ArgumentParser(description='Run global BA on a batch or a set of related batches')
    parser.add_argument('--path', nargs='+', required=True, help='path to folder with result.pickle and kapture-folder')
    parser.add_argument('--matches-path', type=str, help='path to across-batch feature match file')
    parser.add_argument('--frozen-batches', nargs='+', help='path to kapture-containing folder of batches that want to freeze')
    parser.add_argument('--img-sc', default=0.5, type=float, help='image scale')
    parser.add_argument('--nadir-looking', action='store_true', help='is cam looking down? used for plots only')
    parser.add_argument('--plot', action='store_true', help='plot result')
    parser.add_argument('--plot-only', action='store_true', help='plot initial result, exit')
    parser.add_argument('--ref-model-dem', type=str, help='plot reference model along initial result, geotiff file')
    parser.add_argument('--ref-model-orto', type=str, help='plot reference model along initial result, orto image')
    parser.add_argument('--plot-matches-only', action='store_true', help='plot akaze feature matches, exit')
    parser.add_argument('--ini-fl', type=float, help='initial value for focal length')
    parser.add_argument('--ini-cx', type=float, help='initial value for x principal point')
    parser.add_argument('--ini-cy', type=float, help='initial value for y principal point')
    parser.add_argument('--ini-dist', nargs='+', default=[], type=float, help='initial values for radial distortion coefs')

    parser.add_argument('--ini-tr', type=float, default=1000, help='initial ba trust region size')
    parser.add_argument('--max-iters', type=int, default=30, help='max ba iterations')
    parser.add_argument('--max-time', type=int, default=0, help='max ba duration in seconds')
    parser.add_argument('--ftol', type=float, default=1e-5, help='stop ba if cost decreases less than this')

    parser.add_argument('--repr-sd', type=float, default=PX_ERR_SD, help='reprojectrion error standard deviation [px]')
    parser.add_argument('--loc-sd', nargs='+', type=float, default=LOC_ERR_SD, help='location error standard deviation [m]')
    parser.add_argument('--ori-sd', nargs='+', type=float, default=ORI_ERR_SD, help='orientation error standard deviation [deg]')
    parser.add_argument('--inter-batch-repr-weight', type=float, default=1.0,
                        help='weight of inter-batch reprojection errors (default=1.0, '
                             'in addition to error count balancing between regular reprojection errors)')
    parser.add_argument('--fix-fl', action='store_true', help='do not optimize focal length')
    parser.add_argument('--fix-pp', action='store_true', help='do not optimize principal point')
    parser.add_argument('--fix-dist', action='store_true', help='do not optimize distortion coefs')
    parser.add_argument('--abs-loc-r', action='store_true', help='use absolute location measures in BA')
    parser.add_argument('--abs-ori-r', action='store_true', help='use absolute orientation measure in BA')
    parser.add_argument('--skip-fe', action='store_true', help='Skip AKAZE feature extraction')
    parser.add_argument('--skip-fm', action='store_true', help='Skip feature matching across batches')
    parser.add_argument('--skip-ba', action='store_true', help='Skip global BA')
    parser.add_argument('--float32', action='store_true', help='use 32-bit floats instead of 64-bits')
    parser.add_argument('--filter', type=float, help='filter out observations with repr err larger than this')
    parser.add_argument('--workers', type=int, default=0, help='how many worker processes to allow')
    parser.add_argument('--normalize-images', action='store_true',
                        help='After all is done, normalize images based on current camera params')
    parser.add_argument('--fe-n', type=int, default=10, help='Extract AKAZE features every n frames')
    parser.add_argument('--fe-triang-int', type=int, default=5, help='AKAZE feature triangulation interval in keyframes')
    parser.add_argument('--feature-name', default=EXTRACTED_FEATURE_NAME,
                        help='what features to extract (akaze, sift, r2d2, or base-filename of own cnn model)')
    args = parser.parse_args()

    if not args.skip_fe and not args.plot_only:
        run_fe(args)

    if not args.skip_fm and not args.plot_only or args.plot_matches_only:
        assert args.matches_path, 'no feature match file path given'
        run_fm(args)

    if not args.skip_ba or args.plot_only:
        run_ba(args)

    if args.normalize_images and not args.plot_only:
        run_ni(args)


def run_ni(args):
    # normalizes images (and idealizes cam params) so that can continue with depth estimation using depthmaps.py
    for i, path in enumerate(args.path):
        kapt_path = os.path.join(path, 'kapture')
        kapt = kapture_from_dir(kapt_path)
        sensor_id, width, height, fl_x, fl_y, pp_x, pp_y, *dist_coefs = cam_params = get_cam_params(kapt, SENSOR_NAME)
        if np.isclose(np.sum(np.abs(dist_coefs)), 0.0):
            logger.warning('at %s already an idealized camera without any distortion coefs!' % path)
            continue

        cam = cam_obj(cam_params)
        map_u, map_v = cv2.initUndistortRectifyMap(cam.cam_mx, np.array(cam.dist_coefs), None,
                                                   cam.cam_mx, (cam.width, cam.height), cv2.CV_16SC2)

        # undistort images and keypoints
        for fid, img_files in tqdm(kapt.records_camera.items(), desc='Undistorting images'):
            img_file = img_files[sensor_id]
            img_path = get_record_fullpath(kapt_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            undist_img = cv2.remap(img, map_u, map_v, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(img_path, undist_img)

            for feature_name, kp_type in kapt.keypoints.items():
                if img_file in kp_type:
                    kp_file_path = get_keypoints_fullpath(feature_name, kapt_path, img_file)
                    pts2d = cam.undistort(image_keypoints_from_file(kp_file_path, kp_type.dtype, kp_type.dsize))
                    image_keypoints_to_file(kp_file_path, pts2d)

        # idealize cam distortion
        cam_params = np.array(cam_params)
        cam_params[-len(dist_coefs):] = 0.0
        set_cam_params(kapt, SENSOR_NAME, cam_params)
        kapture_to_dir(kapt_path, kapt)


def run_fe(args):
    for i, path in enumerate(args.path):
        kapt_path = os.path.join(path, 'kapture')
        kapt = kapture_from_dir(kapt_path)
        sensor_id, width, height, fl_x, fl_y, pp_x, pp_y, *dist_coefs = cam_params = get_cam_params(kapt, SENSOR_NAME)

        if kapt.keypoints is None:
            kapt.keypoints = {}
        if kapt.descriptors is None:
            kapt.descriptors = {}

        # remove all previous extra features
        rem_pt3d = []
        for feature_name in kapt.keypoints.keys():
            if feature_name != VO_FEATURE_NAME:
                for img_file in kapt.keypoints[feature_name]:
                    os.unlink(get_keypoints_fullpath(feature_name, kapt_path, img_file))
                    os.unlink(get_descriptors_fullpath(feature_name, kapt_path, img_file))

                # remove also related 3d-points and observations
                for pt3d_id, obs in kapt.observations.items():
                    if feature_name in obs:
                        obs.pop(feature_name)
                        if len(obs) == 0:
                            rem_pt3d.append(pt3d_id)
        if len(rem_pt3d) > 0:
            for pt3d_id in rem_pt3d:
                kapt.observations.pop(pt3d_id)
            min_id, max_id, tot_ids = np.min(rem_pt3d), np.max(rem_pt3d), len(rem_pt3d)
            assert max_id - min_id + 1 == tot_ids and len(kapt.points3d) - 1 == max_id, \
                   'can only remove all the 3d-points after a certain index'
            kapt.points3d = kapt.points3d[:min_id, :]

        # NOTE: size changed 2=>3, breaks compatibility with previous versions, for scale restricted matching
        kapt.keypoints[args.feature_name] = kt.Keypoints(args.feature_name, np.float32, 3)

        if args.feature_name == 'akaze':
            kapt.descriptors[args.feature_name] = kt.Descriptors(args.feature_name, np.uint8, 61,
                                                                 args.feature_name, 'hamming')
        elif args.feature_name in ('sift', 'rsift'):
            kapt.descriptors[args.feature_name] = kt.Descriptors(args.feature_name, np.float32, 128,
                                                                 args.feature_name, 'L2')
        else:
            assert args.feature_name == 'r2d2' or os.path.exists(os.path.join(CNN_MODEL_PATH, args.feature_name + '.ckpt')), \
                'invalid feature %s' % (args.feature_name,)
            kapt.descriptors[args.feature_name] = kt.Descriptors(args.feature_name, np.float32, 128,
                                                                 args.feature_name, 'L2')

        logger.info('Extracting %s-features from every %dth frame' % (args.feature_name, args.fe_n))
        records = list(kapt.records_camera.items())
        for j in tqdm(range(0, len(records) - args.fe_triang_int, args.fe_n)):
            d = []
            for k in (0, args.fe_triang_int):
                fid, img_files = records[j+k]
                sc_q = kapt.trajectories[(fid, sensor_id)].r
                img_path = get_record_fullpath(kapt_path, img_files[sensor_id])
                kps, descs = extract_features(img_path, sc_q, args)
                d.append((img_files[sensor_id], kps, descs))

            # match features based on descriptors, triangulate matches, filter out bad ones
            pts3d, idx1, idx2 = triangulate(kapt, cam_params, *d[0], *d[1], args.feature_name)

            if pts3d is not None:
                # write 3d points to kapture
                pt3d_id_start = len(kapt.points3d)
                kapt.points3d = kt.Points3d(np.concatenate((kapt.points3d,
                    np.concatenate((pts3d, np.ones_like(pts3d) * 128), axis=1)
                ), axis=0))

                for (img_file, kps, descs), idx in zip(d, (idx1, idx2)):
                    # write keypoints and descriptors to kapture
                    kapt.keypoints[args.feature_name].add(img_file)
                    kapt.descriptors[args.feature_name].add(img_file)
                    image_keypoints_to_file(get_keypoints_fullpath(args.feature_name, kapt_path, img_file),
                                            kps[idx, :])
                    image_descriptors_to_file(get_descriptors_fullpath(args.feature_name, kapt_path, img_file),
                                              descs[idx, :])

                    # write observations to kapture
                    for id in range(len(idx)):
                        kapt.observations.add(pt3d_id_start + id, args.feature_name, img_file, id)

        kapture_to_dir(kapt_path, kapt)


def run_fm(args):
    n = len(args.path)
    if n <= 1:
        return

    # Example values:
    #   path = 'data/output/27-kapt'
    #   batch-id = 27
    #   frame-id = 6   (for image cam/frame000006.jpg)
    #   feature-id: 4  (4th feature extracted for given frame)
    cam_params = {}  # {batch-id: cam_params, ...}
    poses = {}  # {(batch-id, frame-id): pose, ...}
    pts2d = {}  # {(batch-id, frame-id): [2d-point, ...], ...}       # feature-id is the index of the inner list
    descr = {}  # {(batch-id, frame-id): [descr, ...], ...}          # feature-id is the index of the inner list
    obser = {}  # {(batch-id, frame-id, feature-id): 3d-point, ...}

    res_pts3d = []  # [[3d-point, ...], ...]
    res_obser_map = {}  # {(batch-id, frame-id, feature-id): pt3d-id, ...}  # pt3d-id is the index of list res_pts3d
    processed_pairs = set()
    frozen_pts3d = set()
    if os.path.exists(args.matches_path):
        with open(args.matches_path, 'rb') as fh:
            res_pts3d, res_obser_map, processed_pairs = pickle.load(fh)
        frozen_bids = list(map(path2batchid, args.frozen_batches or []))
        frozen_pts3d = {pt3d_id for (bid, fid, kid), pt3d_id in res_obser_map.items() if bid in frozen_bids}

    img_files = {}  # for plotting only

    k = 0
    for i, path1 in enumerate(args.path):
        for j, path2 in zip(range(i + 1, len(args.path)), args.path[i+1:]):
            k += 1
            b1, b2 = map(path2batchid, (path1, path2))
            if not args.plot_matches_only and ((b1, b2) in processed_pairs or (b2, b1) in processed_pairs):
                logger.info('Already processed %s and %s (%d/%d)' % (b1, b2, k, n*(n-1)/2))
            else:
                find_matches(path1, path2, cam_params, poses, pts2d, descr, obser, res_pts3d, res_obser_map,
                             args.feature_name, args.plot, img_files, plot_only=args.plot_matches_only,
                             desc='Processing %s and %s (%d/%d)' % (b1, b2, k, n*(n-1)/2))
                processed_pairs.add((b1, b2))

    if args.plot_matches_only:
        return

    # calculate mean of each entry in res_pts3d, except if point observed by a frozen batch
    res_pts3d = np.array([np.atleast_2d(np.array(pts3d))[0, :] if i in frozen_pts3d
                          else np.mean(np.atleast_2d(np.array(pts3d)), axis=0)
                          for i, pts3d in enumerate(res_pts3d)])

    # save res_pts3d, res_obser so that can use for global ba
    with open(args.matches_path, 'wb') as fh:
        pickle.dump((res_pts3d, res_obser_map, processed_pairs), fh)


def path2batchid(path):
    return path.split('/')[-1].split('-')[0]


def find_matches(path1, path2, cam_params, poses, kps, descr, obser, res_pts3d, res_obser, feature_name,
                 plot=False, img_files=None, plot_only=False, desc=None):
    # find all frames with akaze features, load poses and observations
    bid1, bid2 = map(path2batchid, (path1, path2))

    for path in (path1, path2):
        bid = path2batchid(path)
        if bid in cam_params:
            continue

        kapt_path = os.path.join(path, 'kapture')
        kapt = kapture_from_dir(kapt_path)
        cam_params[bid] = get_cam_params(kapt, SENSOR_NAME)

        sensor_id, cam = cam_params[bid][0], cam_obj(cam_params[bid])
        img2fid = dict([(fname[sensor_id], id) for id, fname in kapt.records_camera.items()])
        kp_type = kapt.keypoints[feature_name]
        ds_type = kapt.descriptors[feature_name]

        for img_file in kapt.keypoints[feature_name]:
            frame_id = img2fid[img_file]
            traj = kapt.trajectories[(frame_id, sensor_id)]
            poses[(bid, frame_id)] = Pose(traj.t, traj.r)
            _kps = image_keypoints_from_file(
                get_keypoints_fullpath(feature_name, kapt_path, img_file), kp_type.dtype, kp_type.dsize)
            _kps[:, :2] = cam.undistort(_kps[:, :2]).squeeze()
            kps[(bid, frame_id)] = _kps
            descr[(bid, frame_id)] = image_descriptors_from_file(
                get_descriptors_fullpath(feature_name, kapt_path, img_file), ds_type.dtype, ds_type.dsize)

            # {feat_id: pt3d_id, ...}
            feat_to_pt3d = {feat_id: pt3d_id
                            for pt3d_id, t in kapt.observations.items() if feature_name in t
                            for f, feat_id in t[feature_name] if f == img_file}
            obser.update({(bid, frame_id, feat_id): kapt.points3d[pt3d_id][:3]
                          for feat_id, pt3d_id in feat_to_pt3d.items()})

            if plot or plot_only:
                img_files[(bid, frame_id)] = get_record_fullpath(kapt_path, img_file)

    if plot_only:
        fids1, kids1, fids2, kids2, pts3d = get_pairs(res_obser, res_pts3d, bid1, bid2)
        for fid1, kid1, fid2, kid2, pt3d in tqdm(zip(fids1, kids1, fids2, kids2, pts3d),
                                                 total=len(fids1), desc='%s - %s pairs' % (bid1, bid2)):
            repr_kps = []
            cam1, cam2 = map(lambda bid: cam_obj(cam_params[bid]), (bid1, bid2))
            for bid, fid, cam in [(bid1, fid1, cam1), (bid2, fid2, cam2)]:
                pose = poses[(bid, fid)]
                pts3d_cf = tools.q_times_mx(pose.quat, pt3d) + pose.loc
                repr_kps.append(np.atleast_2d(cam.project(pts3d_cf.astype(np.float32)) + 0.5).astype(int))

            plot_matches(img_files[(bid1, fid1)], img_files[(bid2, fid2)],
                         cam1.distort(kps[(bid1, fid1)][kid1.astype(int)].squeeze()[:, :2]),
                         cam2.distort(kps[(bid2, fid2)][kid2.astype(int)].squeeze()[:, :2]),
                         repr_kps1=repr_kps[0], repr_kps2=repr_kps[1])
        return

    # TODO: in some distant future, find closest pose based on GFTT/AKAZE locations instead of drone location
    #       (similar to what is done in navex/datasets/preproc)
    # for each pose in batch1, find closest pose from batch2 (based on location only)
    fids1, locs1 = zip(*[(fid, pose.loc) for (bid, fid), pose in poses.items() if bid == bid1])
    fids2, locs2 = zip(*[(fid, pose.loc) for (bid, fid), pose in poses.items() if bid == bid2])
    tree = cKDTree(np.array(locs2))
    d, idxs2 = tree.query(locs1)

    # loop through matched frames
    successes = 0
    pbar = tqdm(zip(fids1, idxs2), total=len(fids1), desc=desc)
    for fid1, idx2 in pbar:
        if idx2 < len(fids2):
            # match features, verify with 3d-2d-ransac both ways, add 3d points to list
            d, cams = [], []
            for path, fid in zip((path1, path2), (fid1, fids2[idx2])):
                bid = path2batchid(path)
                cams.append(cam_obj(cam_params[bid]))
                pts3d = np.array([obser[(bid, fid, i)] for i in range(len(kps[(bid, fid)]))])
                d.append((cams[-1].cam_mx, kps[(bid, fid)], descr[(bid, fid)], pts3d))

            if plot:
                plot_matches(img_files[(bid1, fid1)], img_files[(bid2, fids2[idx2])],
                             cams[0].distort(kps[(bid1, fid1)].squeeze()[:, :2]),
                             cams[1].distort(kps[(bid2, fids2[idx2])].squeeze()[:, :2]), kps_only=True)

            matches1, matches2 = match_and_validate(*d[0], *d[1], feature_name)

            if matches1 is not None:
                for feat_id1, feat_id2 in zip(matches1, matches2):
                    key1 = (bid1, fid1, feat_id1)
                    key2 = (bid2, fids2[idx2], feat_id2)
                    pt3d_id = None
                    if key1 in res_obser:
                        pt3d_id = res_obser[key1]
                    if pt3d_id is None and key2 in res_obser:
                        pt3d_id = res_obser[key2]
                    if pt3d_id is None:
                        pt3d_id = len(res_pts3d)
                        res_pts3d.append([])
                    res_pts3d[pt3d_id].append(obser[key1])
                    res_pts3d[pt3d_id].append(obser[key2])
                    res_obser[key1] = pt3d_id
                    res_obser[key2] = pt3d_id

                successes += 1
                pbar.set_postfix({'successful matches': successes})
                if plot:
                    cam1, cam2 = map(lambda bid: cam_obj(cam_params[bid]), (bid1, bid2))
                    plot_matches(img_files[(bid1, fid1)], img_files[(bid2, fids2[idx2])],
                                 cam1.distort(d[0][1][matches1, ...].squeeze()[:, :2]),
                                 cam2.distort(d[1][1][matches2, ...].squeeze()[:, :2]))


#@profile
def run_ba(args):
    # check https://github.com/NikolausDemmel/rootba/blob/master/src/rootba/bal/solver_options.hpp
    solver = RootBundleAdjuster(
        ini_tr_rad=args.ini_tr,
        min_tr_rad=1e-15 if args.float32 else 1e-32,
        max_tr_rad=3e6 if args.float32 else 1e16,
        ini_vee=2.0,
        vee_factor=2.0,
        n_workers=args.workers,
        max_iters=args.max_iters,
        max_time=args.max_time,   # in sec
        min_step_quality=0,
        xtol=0,
        rtol=0,
        ftol=args.ftol,  #3e-4,

        # filter out observations with large reprojection errors at these iterations
        max_repr_err={0: args.filter} if args.filter else {},  #0: 320, 1: 64, 2: 32, 3: 16, 4: 12, 6: 8},

        jacobi_scaling_eps=0,
        lin_cg_maxiter=500,
        lin_cg_tol=1e-5,
        preconditioner_type=LinearizerQR.PRECONDITIONER_TYPE_SCHUR_JACOBI,

        huber_coefs=HUBER_COEFS,
        use_weighted_residuals=True,
        inter_batch_repr_weight=args.inter_batch_repr_weight,
    )

    akaze_pts3d, akaze_obser_map = [None] * 2
    if args.matches_path:
        # load AKAZE features so that can join separate batches before running BA
        with open(args.matches_path, 'rb') as fh:
            # [3d-point, ...]
            # {(batch-id, frame-id, feature-id): pt3d-id, ...}  # pt3d-id is the index to above list
            akaze_pts3d, akaze_obser_map, *_ = pickle.load(fh)
        akaze_obser_map = {(path2batchid(bid), fid, kid): pt3d_id for (bid, fid, kid), pt3d_id in akaze_obser_map.items()}
    else:
        assert len(args.path) == 1, 'optimizing multiple batches separately not supported, please specify ' \
                                    '--matches-path for optimizing the given batches together'

    arr_pts2d, arr_cam_params, arr_cam_param_idxs, arr_poses, arr_pose_idxs = [], [], [], [], []
    arr_pts3d, arr_pt3d_idxs, arr_meas_r, arr_meas_aa, arr_meas_idxs = [], [], [], [], []
    arr_akaze_obser, arr_frames, batch_ids, arr_kapt = [], [], [], []

    ref_model = None
    if args.plot_only and args.ref_model_dem:
        assert args.ref_model_orto, 'give both dem and orto'
        from visnav.iotools.terrainmodel import TerrainModel
        ref_model = TerrainModel(args.ref_model_dem, args.ref_model_orto, NokiaSensor.COORD0,
                                 NokiaSensor.b2c.to_global(NokiaSensor.w2b, False),
                                 bounds=(0.250, 0.458, 0.417, 0.667))  # (t, l, b, r)

    for i, path in enumerate(args.path):
        with open(os.path.join(path, 'result.pickle'), 'rb') as fh:
            orig_keyframes, map3d, frame_names, meta_names, *_ = pickle.load(fh)
        del _
        if not args.plot_only:
            del map3d, frame_names, meta_names

        kapt_path = os.path.join(path, 'kapture')
        kapt = kapture_from_dir(kapt_path)
        sensor_id, width, height, fl_x, fl_y, pp_x, pp_y, *dist_coefs = get_cam_params(kapt, SENSOR_NAME)
        arr_kapt.append((kapt, width, height))

        if args.plot_only:
            plot_results(orig_keyframes, map3d, frame_names, meta_names, nadir_looking=args.nadir_looking)
            replay_kapt(kapt_path, kapt, ref_model=ref_model)
            continue

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

        # what params to optimize
        cam_param_idxs = [0, 2, 3, 4, 5]
        if args.fix_fl:
            cam_param_idxs.remove(0)
        if args.fix_pp:
            cam_param_idxs.remove(2)
            cam_param_idxs.remove(3)
        if args.fix_dist:
            cam_param_idxs.remove(4)
            cam_param_idxs.remove(5)
        cam_param_idxs = np.array(cam_param_idxs, dtype=int)

        logger.info('loading poses and keypoints...')
        frames, poses, pts3d, pts2d, pose_idxs, pt3d_idxs, meas_r, meas_aa, meas_idxs, akaze_obser, _ = \
            get_ba_params(kapt_path, orig_keyframes, kapt, sensor_id, args.feature_name)
        del _, orig_keyframes

        _meas_r = meas_r if args.abs_loc_r else None   # enable/disable loc measurements
        _meas_aa = meas_aa if args.abs_ori_r else None  # enable/disable ori measurements
        _meas_idxs = None if meas_r is None and meas_aa is None else meas_idxs

        arr_pts2d.append(pts2d)
        arr_cam_params.append(cam_params)
        arr_cam_param_idxs.append(cam_param_idxs)
        arr_poses.append(poses)
        arr_pose_idxs.append(pose_idxs)
        arr_pts3d.append(pts3d)
        arr_pt3d_idxs.append(pt3d_idxs)
        arr_meas_r.append(_meas_r)
        arr_meas_aa.append(_meas_aa)
        arr_meas_idxs.append(_meas_idxs)
        arr_akaze_obser.append(akaze_obser)
        arr_frames.append(frames)
        batch_ids.append(path2batchid(path))

    if args.plot_only:
        exit()

    frozen_batches = [path2batchid(path) for path in (args.frozen_batches or [])]

    pts2d, batch_idxs, cam_params, cam_param_idxs, poses, pose_idxs, pose_batch, pts3d, pt3d_idxs, pt3d_batch, \
            frozen_points, pt3d_gftt_n, meas_r, meas_aa, meas_idxs, akaze_repr_err_count = \
                    join_batches(arr_pts2d, arr_cam_params, arr_cam_param_idxs, arr_poses,
                                 arr_pose_idxs, arr_pts3d, arr_pt3d_idxs, arr_meas_r, arr_meas_aa,
                                 arr_meas_idxs, arr_akaze_obser, arr_frames, batch_ids,
                                 akaze_pts3d, akaze_obser_map, frozen_batches)

    del arr_pts2d, arr_cam_params, arr_cam_param_idxs, arr_poses, arr_pose_idxs, arr_pts3d, arr_pt3d_idxs, arr_meas_r, \
        arr_meas_aa, arr_meas_idxs, arr_akaze_obser, arr_frames, batch_ids, akaze_pts3d, akaze_obser_map

    problem = Problem(pts2d, batch_idxs, cam_params, cam_param_idxs, poses, pose_idxs, pose_batch,
                      pts3d, pt3d_idxs, pt3d_batch, frozen_points, akaze_repr_err_count, meas_r, meas_aa, meas_idxs,
                      args.repr_sd, args.loc_sd, args.ori_sd, dtype=np.float32 if args.float32 else np.float64)

    del pts2d, batch_idxs, cam_params, cam_param_idxs, poses, pose_idxs, pts3d, pt3d_idxs, meas_r, meas_aa, meas_idxs

    if 1:
        solver.solve(problem, callback=lambda x: save_and_plot(x, args, arr_kapt, pt3d_gftt_n, plot=True, save=True)
                                       if DEBUG else None)

        save_and_plot(problem, args, arr_kapt, pt3d_gftt_n, log=True, save=True, plot=args.plot)

        if args.plot:
            for i, (path, kapt) in enumerate(zip(args.path, arr_kapt)):
                replay_kapt(os.path.join(path, 'kapture'), kapt[0])
    elif 1:
        N, M = 1000, 100
        problem.maybe_populate_cache()

        if 1:
            x0 = problem.xb[:M]
            J = problem.jacobian_repr_batch(fmt='dense')
            J = J[:N, :M]

            def costfun(x):
                problem.xb = np.concatenate((x, problem.xb[M:]), axis=0)
                return problem.residual_repr()[:N].flatten()
        elif 0:
            x0 = problem.xp[:M]
            J = problem.jacobian_repr_frame(fmt='lil')
            J = J[:N, :M].toarray()

            def costfun(x):
                problem.xp = np.concatenate((x, problem.xp[M:]), axis=0)
                return problem.residual_repr()[:N].flatten()
        else:
            x0 = problem.xl[:M]
            J = problem.jacobian_repr_landmark(fmt='lil')
            J = J[:N, :M].toarray()

            def costfun(x):
                problem.xl = np.concatenate((x, problem.xl[M:]), axis=0)
                return problem.residual_repr()[:N].flatten()

        J_ = numerical_jacobian(costfun, x0, 1e-6)

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        axs[0].imshow(np.sign(J) * np.abs(J) ** (1/8))
        axs[0].set_title('analytical')
        axs[1].imshow(np.sign(J_) * np.abs(J_) ** (1/8))
        axs[1].set_title('numerical')
        plt.show()

    else:
        J = problem.jacobian()
        r = problem.residual()

        a0, K, loc_err_sd = np.array([]), np.array([[fl_x, 0, pp_x], [0, fl_y, pp_y], [0, 0, 1]]), np.array(args.loc_sd)
        rref, Jref = vis_gps_bundle_adj(poses, pts3d, pts2d, a0, pose_idxs, pt3d_idxs, K, dist_coefs,
                                        np.array([args.repr_sd]), meas_r, meas_aa, a0, meas_idxs, loc_err_sd, np.inf,
                                        max_nfev=10, skip_pose_n=0, huber_coef=HUBER_COEFS, poses_only=False,
                                        just_return_r_J=True)

        J = J.toarray()
        Jref = Jref.toarray()

        import matplotlib.pyplot as plt
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


def save_and_plot(problem, args, arr_kapt, pt3d_gftt_n, log=False, save=False, plot=False):
    pose_batch, pt3d_batch, all_cam_params, all_poses, all_pts3d = \
        map(lambda x: getattr(problem, x), ('pose_batch', 'pt3d_batch', 'cam_params', 'poses', 'pts3d'))

    for i, path in enumerate(args.path):
        pose_I = np.where(pose_batch == i)[0]
        cam_params = all_cam_params[i]

        if log:
            fl_x, fl_y, pp_x, pp_y, *dist_coefs = cam_params
            logger.info('batch %d, new fl: %s, cx: %s, cy: %s, k1, k2: %s' % (
                i + 1, fl_x / args.img_sc, pp_x / args.img_sc, pp_y / args.img_sc, dist_coefs))

        if save or plot:
            with open(os.path.join(path, 'result.pickle'), 'rb') as fh:
                keyframes, map3d, frame_names, meta_names, gt, ba_errs = pickle.load(fh)

            poses = [Pose(all_poses[j, 3:], tools.angleaxis_to_q(all_poses[j, :3])) for j in pose_I]

            # includes also the akaze keypoints
            pts3d = all_pts3d[pt3d_batch[:, i], :]

            keyframes = keyframes[:len(poses)]
            for j, pose in enumerate(poses):
                keyframes[j]['pose'].post = pose
            map3d = [(setattr(map3d[j], 'pt3d', pts3d[j, :]) or map3d[j]) if len(map3d) > j
                     else Keypoint(pt3d=pts3d[j, :]) for j in range(len(pts3d))]

        if save:
            with open(os.path.join(path, 'result.pickle'), 'wb') as fh:
                pickle.dump((keyframes, map3d, frame_names, meta_names, gt, ba_errs), fh)

            kapt, width, height = arr_kapt[i]
            update_kapture(os.path.join(path, 'kapture'), kapt, [width, height] + list(cam_params), poses, pts3d)

        if plot:
            plot_results(keyframes, map3d, frame_names, meta_names, nadir_looking=args.nadir_looking)

    if save and args.matches_path:
        # update joint-obs.pickle with new 3d-points
        with open(args.matches_path, 'rb') as fh:
            _, akaze_obser_map, processed_pairs = pickle.load(fh)
        with open(args.matches_path, 'wb') as fh:
            pickle.dump((all_pts3d[pt3d_gftt_n:], akaze_obser_map, processed_pairs), fh)


#@profile
def get_ba_params(kapt_path, keyframes, kapt, sensor_id, feature_name=EXTRACTED_FEATURE_NAME):
    frames = [(id, fname[sensor_id]) for id, fname in kapt.records_camera.items()]
    frames = sorted(frames, key=lambda x: x[0])
    fname2id = {fname: id for id, fname in frames}

    feat = kapt.keypoints[VO_FEATURE_NAME]
    uv_map = {}
    for id_f, fname in frames:
        uvs = image_keypoints_from_file(get_keypoints_fullpath(VO_FEATURE_NAME, kapt_path, fname), feat.dtype, feat.dsize)
        uv_map[id_f] = uvs

    # load also observations of akaze features
    if feature_name in kapt.keypoints:
        ef_frames = {fname for r in kapt.observations.values() if feature_name in r
                           for fname, id2 in r[feature_name]}
        feat = kapt.keypoints[feature_name]
        ef_uv_map = {}
        for fname in ef_frames:
            uvs = image_keypoints_from_file(get_keypoints_fullpath(feature_name, kapt_path, fname), feat.dtype, feat.dsize)
            ef_uv_map[fname2id[fname]] = uvs

    max_pt3d_id = -1
    f_uv, ef_uv = {}, {}
    for id3, r in kapt.observations.items():
        if VO_FEATURE_NAME in r:
            for fname, id2 in r[VO_FEATURE_NAME]:
                id_f = fname2id[fname]
                f_uv.setdefault(id_f, {})[id3] = uv_map[id_f][id2, :]
                max_pt3d_id = max(max_pt3d_id, id3)
        elif feature_name in r:
            for fname, id2 in r[feature_name]:
                id_f = fname2id[fname]
                ef_uv.setdefault(id_f, {})[id2] = tuple(ef_uv_map[id_f][id2, :])

    obs_kp = list(set.union(*[set(m.keys()) for m in f_uv.values()]))
    pts3d = kapt.points3d[:max_pt3d_id+1, :3]

    cam_idxs, pt3d_idxs, pts2d = list(map(np.array, zip(*[
        (i, id3, uv.flatten())
        for i, (id_f, fname) in enumerate(frames) if id_f in f_uv
            for id3, uv in f_uv[id_f].items()
    ])))

    # NOTE: there's poses that have no observations! i.e. id_f not always in f_uv, all poses needed for cam_idxs to work
    # TODO: should remove unobservable poses (and 3d-points), maybe after initial ba?

    poses = np.array([[*tools.q_to_angleaxis(kapt.trajectories[id][sensor_id].r, True),
                       *kapt.trajectories[id][sensor_id].t.flatten()] for id, fname in frames], dtype=float)

    if keyframes is not None:
        meas_idxs = np.array([i for i, kf in enumerate(keyframes) if kf['meas'] is not None and i < len(poses)], dtype=int)
        meas_q = {i: keyframes[i]['pose'].prior.quat.conj() for i in meas_idxs}
        meas_r = np.array([tools.q_times_v(meas_q[i], -keyframes[i]['pose'].prior.loc) for i in meas_idxs], dtype=np.float32)
        meas_aa = np.array([tools.q_to_angleaxis(meas_q[i], compact=True) for i in meas_idxs], dtype=np.float32)
    else:
        meas_r, meas_aa, meas_idxs = [None] * 3

    return frames, poses, pts3d, pts2d, cam_idxs, pt3d_idxs, meas_r, meas_aa, meas_idxs, ef_uv, obs_kp


def update_kapture(kapt_path, kapt, cam_params, poses, pts3d):
    cam_id = set_cam_params(kapt, SENSOR_NAME, cam_params)

    frame_ids = sorted(list(kapt.records_camera.keys()))
    for frame_id, pose in zip(frame_ids, poses):
        kapt.trajectories[(frame_id, cam_id)] = kt.PoseTransform(r=pose.quat.components, t=pose.loc)

    kapt.points3d[:len(pts3d), :] = np.concatenate((pts3d, np.ones_like(pts3d)*128), axis=1)

    kapture_to_dir(kapt_path, kapt)


def extract_features(img_path, sc_q, args):
    if args.feature_name == 'akaze':
        if extract_features.detector is None:
            extract_features.detector = cv2.AKAZE_create(**EXTRACTED_FEATURE_PARAMS[args.feature_name])
        det = extract_features.detector
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        kps = detect_gridded(det, img, None, *EXTRACTED_FEATURE_GRID, EXTRACTED_FEATURE_COUNT)
        kps, descs = det.compute(img, kps)
    elif args.feature_name in ('rsift', 'sift'):
        if extract_features.detector is None:
            extract_features.detector = cv2.SIFT_create(**EXTRACTED_FEATURE_PARAMS[args.feature_name])
        det = extract_features.detector
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        kps = detect_gridded(det, img, None, *EXTRACTED_FEATURE_GRID, EXTRACTED_FEATURE_COUNT)
        kps, descs = det.compute(img, kps)
        if args.feature_name == 'rsift' and descs is not None and np.array(descs).size > 0:
            descs = np.array(descs)
            descs = np.sqrt(descs / np.sum(descs, axis=1, keepdims=True))
    else:
        model_path = os.path.join(CNN_MODEL_PATH, args.feature_name + '.ckpt')
        type = 'r2d2' if args.feature_name == 'r2d2' else 'own'

        assert type == 'r2d2' or os.path.exists(model_path), \
            'feature extractor "%s" not implemented' % args.feature_name

        if extract_features.detector is None:
            from visnav.algo.cnndet import CNN_Detector
            extract_features.detector = CNN_Detector(type, model_path, gpu=None)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        kps, descs = extract_features.detector.normalizeDetectAndCompute(img, sc_q)

    kps = np.array([(*k.pt, k.size) for k in kps], dtype='f4').reshape((-1, 3))
    return kps, descs


extract_features.detector = None


def cam_obj(cam_params):
    sensor_id, width, height, fl_x, fl_y, pp_x, pp_y, *dist_coefs = cam_params
    cam_mx = np.array([[fl_x, 0, pp_x], [0, fl_y, pp_y], [0, 0, 1]])
    cam = Camera(width, height, cam_mx=cam_mx, dist_coefs=dist_coefs)
    return cam


def triangulate(kapt, cam_params, img_file1, kps1, descs1, img_file2, kps2, descs2, feature_name):
    if descs1 is None or descs2 is None or len(descs1) < MIN_INLIERS or len(descs2) < MIN_INLIERS:
        return None, None, None

    # match features based on descriptors, cross check for validity, sort keypoints so that indices indicate matches
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING if feature_name == 'akaze' else cv2.NORM_L2, True)
    matches = matcher.match(descs1, descs2)
    if len(matches) < MIN_INLIERS:
        return None, None, None

    matches = np.array(matches)
    kps1 = kps1[[m.queryIdx for m in matches], :2]
    kps2 = kps2[[m.trainIdx for m in matches], :2]

    # get camera parameters, undistort keypoints
    sensor_id, cam = cam_params[0], cam_obj(cam_params)
    kps1, kps2 = map(lambda x: cam.undistort(x).squeeze(), (kps1, kps2))

    # get camera frame pose matrices
    img2fid = dict([(fname[sensor_id], id) for id, fname in kapt.records_camera.items()])
    pose1, pose2 = map(lambda x: kapt.trajectories[(img2fid[x], sensor_id)], (img_file1, img_file2))
    pose1, pose2 = map(lambda x: Pose(x.t, x.r).to_mx(), (pose1, pose2))

    # triangulate matched keypoints
    kps4d = cv2.triangulatePoints(cam.cam_mx.dot(pose1), cam.cam_mx.dot(pose2),
                                  kps1.reshape((-1, 1, 2)), kps2.reshape((-1, 1, 2)))
    pts3d = (kps4d.T[:, :3] / kps4d.T[:, 3:])

    # filter out bad triangulation results
    mask = np.ones((len(matches),), dtype=bool)
    for pose, kps in zip((pose1, pose2), (kps1, kps2)):
        pose44 = np.eye(4)
        pose44[:3, :] = pose

        # check that points are in front of the cam
        kps4d_l = pose44.dot(kps4d)
        kps3d_l = kps4d_l.T[:, :3] / kps4d_l.T[:, 3:]
        mask = np.logical_and(mask, kps3d_l[:, 2] > 0)  # camera borehole towards +z axis

        # check that reprojection error is not too large
        repr_kps = cam.cam_mx.dot(kps3d_l.T).T
        repr_kps = repr_kps[:, :2] / repr_kps[:, 2:]
        repr_err = np.linalg.norm(kps - repr_kps, axis=1)
        mask = np.logical_and(mask, repr_err < MAX_REPR_ERROR)

    if np.sum(mask) < MIN_INLIERS:
        return None, None, None

    return (pts3d[mask, :], *zip(*[[m.queryIdx, m.trainIdx] for m in matches[mask]]))


def match_and_validate(cam_mx1, kps1, descr1, pts3d1, cam_mx2, kps2, descr2, pts3d2, feature_name):
    pts2d1, scale1 = kps1[:, :2], kps1[:, 2]
    pts2d2, scale2 = kps2[:, :2], kps2[:, 2]

    if descr1 is None or descr2 is None or len(descr1) < MIN_INLIERS or len(descr2) < MIN_INLIERS:
        return None, None

    # scale restricted matching of features based on descriptors,
    # mutual matching for validity
    matches = scale_restricted_match(scale1, descr1, scale2, descr2,
                                     norm=cv2.NORM_HAMMING if feature_name == 'akaze' else cv2.NORM_L2)

    if matches is None or len(matches) < MIN_INLIERS:
        return None, None

    matches = np.array(matches)
    pts2d1, pts3d1 = map(lambda x: x[[m.queryIdx for m in matches], :], (pts2d1, pts3d1))
    pts2d2, pts3d2 = map(lambda x: x[[m.trainIdx for m in matches], :], (pts2d2, pts3d2))

    # kps1 and kps2 already undistorted

    # 3d-2d ransac on both, need same features to be inliers
    inliers = set(range(len(kps1)))
    for pts3d, cam_mx, kps in zip((pts3d1, pts3d2), (cam_mx2, cam_mx1), (pts2d2, pts2d1)):
        ok, rv, r, inl = cv2.solvePnPRansac(pts3d, kps, cam_mx, None, iterationsCount=10000,
                                            reprojectionError=MAX_REPR_ERROR, flags=cv2.SOLVEPNP_AP3P)
        inliers = inliers.intersection(inl.flatten() if inl is not None else [])

    if len(inliers) < MIN_INLIERS:
        return None, None

    return (*zip(*[[m.queryIdx, m.trainIdx] for m in matches[list(inliers)]]),)


def match(des1, des2, norm, mask=None):
    matcher = cv2.BFMatcher(norm, True)
    matches = matcher.match(des1, des2, mask=mask)
    return matches


def scale_restricted_match(sc1, des1, sc2, des2, norm, octave_levels=4):
    K1, K2 = len(sc1), len(sc2)

    # log scales used
    s1, s2 = np.log(sc1.squeeze()), np.log(sc2.squeeze())

    # initial matching for scale difference estimation
    matches = match(des1, des2, norm)

    # one level scale difference
    lvl_sc = np.log(2) / octave_levels
    match_levels = 1

    def group_sd(sd):
        x, y = np.unique(sd, return_counts=True)
        arr = []    # [sum(x*y), sum(y)]
        for i in range(len(x)):
            if i > 0 and abs(x[i] - arr[-1][0] / arr[-1][1]) < lvl_sc * 0.1:
                arr[-1][0] += x[i]*y[i]
                arr[-1][1] += y[i]
            else:
                arr.append([x[i]*y[i], y[i]])
        return np.array([[sx/sy, sy] for sx, sy in arr]).reshape((-1, 2)).T

    # get scale difference
    I1, I2 = [m.queryIdx for m in matches], [m.trainIdx for m in matches]
    sd = s2[I2] - s1[I1]

    # gaussian kernel density estimate
    try:
        kde = scipy.stats.gaussian_kde(sd, bw_method=3 * lvl_sc)
    except np.linalg.LinAlgError as e:
        print('Gaussian KDE failed')
        return None
    sd_mean = np.mean(sd)

    # get mode, start from the mean
    sd_mode = scipy.optimize.minimize_scalar(lambda x: -kde(x), method='bounded',
                                             bounds=(sd_mean - 1.5 * lvl_sc, sd_mean + 1.5 * lvl_sc)).x

    match_mask = (s1.view((1, K1, 1)).expand((1, K1, K2)) + sd_mode
                  - s2.view((1, 1, K2)).expand((1, K1, K2))).abs() < lvl_sc * (match_levels - 1 + 0.6)
    mask1 = match_mask.any(dim=2).view(-1)
    mask2 = match_mask.any(dim=1).view(-1)
    match_mask = match_mask[:, mask1, :][:, :, mask2]

    # scale restricted matching
    matches = match(des1[I1[mask1], :], des2[I2[mask2], :], norm, mask=match_mask)

    if mask1.sum() > 0 and mask2.sum() > 0:
        for m in matches:
            m.queryIdx = I1[mask1][m.queryIdx]
            m.trainIdx = I2[mask2][m.trainIdx]
    else:
        s, n = group_sd(sd)
        print(f'No features pass scale restriction, details:'
              f' n1: {mask1.sum()}, n2: {mask2.sum()}, sd_mode: {sd_mode}, sd_mean: {sd_mean},'
              f' s: {s.tolist()}, n: {n.tolist()}')
        return None

    return matches


def join_batches(arr_pts2d, arr_cam_params, arr_cam_param_idxs, arr_poses, arr_pose_idxs, arr_pts3d, arr_pt3d_idxs,
                 arr_meas_r, arr_meas_aa, arr_meas_idxs, arr_akaze_obser, arr_frames, batch_ids,
                 akaze_pts3d=None, akaze_obser_map=None, frozen_batches=None):

    assert len(arr_pts2d) == len(arr_cam_params) == len(arr_cam_param_idxs) == len(arr_poses) == len(arr_pose_idxs) \
           == len(arr_pts3d) == len(arr_pt3d_idxs) == len(arr_meas_r) == len(arr_meas_aa) == len(arr_meas_idxs), \
           'not all parameters are of same length'

    batch_idxs, pose_idxs, pose_batch, pt3d_idxs, pt3d_batch, meas_idxs = [], [], [], [], [], []
    frozen_points = None

    pt3d_count = 0
    pose_count, pose_counts = 0, []
    bid2idx = {bid: bi for bi, bid in enumerate(batch_ids)}
    fid2idx = {bi: {fid: fi for fi, (fid, fname) in enumerate(frames)} for bi, frames in enumerate(arr_frames)}

    for _poses, _pose_idxs, _pts3d, _pt3d_idxs, _meas_idxs in \
        zip(arr_poses, arr_pose_idxs, arr_pts3d, arr_pt3d_idxs, arr_meas_idxs):

        batch_idx = len(pose_counts)
        batch_idxs.append(np.ones((len(_pose_idxs),), dtype=int) * batch_idx)
        pose_idxs.append(_pose_idxs + pose_count)
        pose_batch.append(np.ones((len(_poses),), dtype=int) * batch_idx)
        meas_idxs.append(_meas_idxs + pose_count)
        pt3d_idxs.append(_pt3d_idxs + pt3d_count)

        _pt3d_batch = np.zeros((len(_pts3d), len(batch_ids)), dtype=bool)
        _pt3d_batch[:, batch_idx] = True
        pt3d_batch.append(_pt3d_batch)

        pose_counts.append(pose_count)
        pose_count += len(_poses)
        pt3d_count += len(_pts3d)

    i = 0
    if akaze_pts3d is not None:
        _pts2d = np.zeros((len(akaze_obser_map), 2))
        _batch_idxs = np.zeros((len(akaze_obser_map),), dtype=int)
        _pose_idxs = np.zeros((len(akaze_obser_map),), dtype=int)
        _pt3d_idxs = np.zeros((len(akaze_obser_map),), dtype=int)
        _pt3d_batch = np.zeros((len(akaze_pts3d), len(batch_ids)), dtype=bool)
        frozen_points = np.zeros((pt3d_count + len(akaze_pts3d),), dtype=bool) if frozen_batches else None
        for (batch_id, frame_id, feature_id), id_pt3d in akaze_obser_map.items():
            if batch_id in bid2idx:
                bi = bid2idx[batch_id]
                _pts2d[i, :] = arr_akaze_obser[bi][frame_id][feature_id]
                _batch_idxs[i] = bi
                _pose_idxs[i] = fid2idx[bi][frame_id] + pose_counts[bi]
                _pt3d_idxs[i] = id_pt3d + pt3d_count
                _pt3d_batch[id_pt3d, bi] = True
                i += 1
            if frozen_batches and batch_id in frozen_batches:
                frozen_points[id_pt3d + pt3d_count] = True

        arr_pts2d.append(_pts2d[:i, :])
        batch_idxs.append(_batch_idxs[:i])
        pose_idxs.append(_pose_idxs[:i])
        pt3d_idxs.append(_pt3d_idxs[:i])
        pt3d_batch.append(_pt3d_batch)
        arr_pts3d = arr_pts3d + [akaze_pts3d]
    akaze_repr_err_count = i

    pts2d, batch_idxs, poses, pose_idxs, pose_batch, pts3d, pt3d_idxs, pt3d_batch, meas_r, meas_aa, meas_idxs = map(
        lambda x: None if np.any([k is None for k in x]) else np.concatenate(x, axis=0),
        (arr_pts2d, batch_idxs, arr_poses, pose_idxs, pose_batch, arr_pts3d, pt3d_idxs, pt3d_batch,
         arr_meas_r, arr_meas_aa, meas_idxs))

    return pts2d, batch_idxs, arr_cam_params, arr_cam_param_idxs, poses, pose_idxs, pose_batch, \
           pts3d, pt3d_idxs, pt3d_batch, frozen_points, pt3d_count, meas_r, meas_aa, meas_idxs, akaze_repr_err_count


def get_pairs(all_obser, all_kps3d, bid1, bid2):
    # obser: {(bid, fid, kid): kp3d_id, ....}

    all_fids = {bid1: {}, bid2: {}}
    for (bid, fid, kid), kp3d_id in all_obser.items():
        if bid in all_fids:
            all_fids[bid].setdefault(fid, np.zeros((len(all_kps3d),), dtype=np.uint32))[kp3d_id] = kid + 1

    all_fids1, all_obs1 = zip(*[(fid, obs) for fid, obs in all_fids[bid1].items()])
    all_fids1, all_obs1 = map(lambda x: np.array(x, dtype=np.uint32), (all_fids1, all_obs1))

    all_fids2, all_obs2 = zip(*[(fid, obs) for fid, obs in all_fids[bid2].items()])
    all_fids2, all_obs2 = map(lambda x: np.array(x, dtype=np.uint32), (all_fids2, all_obs2))

    tree = cKDTree(all_obs2 > 0)
    d, idxs2 = tree.query(all_obs1 > 0, p=1)

    common = np.logical_and(all_obs1, all_obs2[idxs2, :])
    I = np.sum(common, axis=1) >= (MIN_INLIERS if 1 else 1)
    idx = np.where(I)[0]
    kps3d = np.array([all_kps3d[row, :] for row in common[I, :]], dtype=object)

    fids1 = all_fids1[I]
    kids1 = np.array([all_obs1[i, common[i, :]] - 1 for i in idx], dtype=object)

    fids2 = all_fids2[idxs2[I]]
    kids2 = np.array([all_obs2[idxs2[i], common[i, :]] - 1 for i in idx], dtype=object)

    I = np.argsort(fids1)
    return fids1[I], kids1[I], fids2[I], kids2[I], kps3d[I]


def plot_matches(img_path1, img_path2, kps1, kps2, repr_kps1=None, repr_kps2=None, kps_only=False):
    def arr2kp(arr, size=7):
        return [cv2.KeyPoint(p[0], p[1], size) for p in arr.squeeze()]

    img1, img2 = map(lambda x: cv2.imread(x), (img_path1, img_path2))

    if repr_kps1 is not None:
        assert repr_kps2 is not None, 'need also repr_kps2 to plot reprojections'
        img = np.concatenate([draw_keypoints(img1, kps1, repr_kps1),
                              draw_keypoints(img2, kps2, repr_kps2)], axis=1)
    elif kps_only:
        img = np.concatenate((cv2.drawKeypoints(img1, arr2kp(kps1), None),
                              cv2.drawKeypoints(img2, arr2kp(kps2), None)), axis=1)
    else:
        matches = [cv2.DMatch(i, i, 0, 0) for i in range(len(kps1))]
        img = cv2.drawMatches(img1, arr2kp(kps1), img2, arr2kp(kps2), matches, None)

    cv2.imshow('keypoints' if kps_only else 'matches', img)
    if not kps_only:
        cv2.setWindowTitle('matches', '%d matches' % len(kps1))
    cv2.waitKey(1 if kps_only else 0)


def draw_keypoints(image, kps, repr_kps, c_diam=7, r_diam=5, c_col=(255, 255, 0), r_col=(255, 255, 0)):
    kps, repr_kps = np.atleast_2d(kps.squeeze()), np.atleast_2d(repr_kps.squeeze())
    for (x, y), (xp, yp) in zip(kps, repr_kps):
        x, y, xp, yp = map(int, (x, y, xp, yp))
        image = cv2.circle(image, (x, y), c_diam, c_col, 1)  # negative thickness => filled circle
        image = cv2.rectangle(image, (xp - r_diam//2, yp - r_diam//2), (xp + r_diam//2, yp + r_diam//2), r_col, 1)
        image = cv2.line(image, (xp, yp), (x, y), r_col, 1)
    return image


def replay_probem(img_paths: List[str], p: Problem):
    replay(img_paths, p.pts2d, p.cam_params, p.poses, p.pose_idxs, p.pts3d, p.pt3d_idxs)


def replay_kapt(kapt_path: str, kapt: Kapture = None, frame_ids=None, ref_model=None):
    if kapt is None:
        kapt = kapture_from_dir(kapt_path)
    cam_params = get_cam_params(kapt, SENSOR_NAME)
    frames, poses, pts3d, pts2d, pose_idxs, pt3d_idxs, *_ = get_ba_params(kapt_path, None, kapt, cam_params[0])
    img_paths = [get_record_fullpath(kapt_path, f[1]) for f in frames]
    replay(img_paths, pts2d, cam_params, poses, pose_idxs, pts3d, pt3d_idxs, ref_model=ref_model)


def replay(img_paths, pts2d, cam_params, poses, pose_idxs, pts3d, pt3d_idxs, frame_ids=None, ref_model=None):
    assert len(img_paths) == len(poses)
    assert len(pts2d) == len(pose_idxs)
    assert len(pts2d) == len(pt3d_idxs)
    cam = cam_params if isinstance(cam_params, Camera) else cam_obj(cam_params)
    kp_size, kp_color = 5, (200, 0, 0)

    for i, (img_path, pose) in enumerate(zip(img_paths, poses)):
        I = pose_idxs == i
        if np.sum(I) == 0:
            continue

        image = img_path if isinstance(img_path, np.ndarray) else cv2.imread(img_path)
        image = cv2.resize(image, (cam.width, cam.height))
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        p_pts2d = (pts2d[I, :] + 0.5).astype(int)
        p_pts3d = pts3d[pt3d_idxs[I], :]

        q_cf = tools.angleaxis_to_q(pose[:3])
        pts3d_cf = tools.q_times_mx(q_cf, p_pts3d) + pose[3:]
        pts2d_proj = np.atleast_2d(cam.project(pts3d_cf.astype(np.float32)) + 0.5).astype(int)

        for (x, y), (xp, yp) in zip(p_pts2d, pts2d_proj):
            image = cv2.circle(image, (x, y), kp_size, kp_color, 1)   # negative thickness => filled circle
            image = cv2.rectangle(image, (xp-2, yp-2), (xp+2, yp+2), kp_color, 1)
            image = cv2.line(image, (xp, yp), (x, y), kp_color, 1)

        if ref_model is not None:
            ref_image = ref_model.project(cam, pose)
            image = np.concatenate((image, ref_image), axis=1)
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            # TODO: try to find correct altitude and focal length for #31?
            #  - estimate them somehow by matching with ortho image?

        cv2.imshow('keypoint reprojection', image)

        frame_id = None
        if frame_ids is not None:
            frame_id = frame_ids[i]
        elif isinstance(img_path, str):
            frame_id = img_path.split('/')[-1].split('.')[0]

        if frame_id is not None:
            cv2.setWindowTitle('keypoint reprojection', 'frame %s' % frame_id)
        if cv2.waitKey() == 27:   # if esc, stop replay, otherwise new frame
            break


if __name__ == '__main__':
    if 0:
        from visnav.algo.linalg import DictArray2D as darr
        a = darr((200, 1000), np.float64)
        a[33, 43] = np.inf
        a[33, [44, 66]] = 22.
        a[[54, 66], [877, 199]] = 99.
        print(str(a[54, 877]))
        print(str(a[66, 199]))
        print(str(a[33, 43]))

#        a.mult_with(np.ones((200, 1), dtype=np.float64) * 1/11)
        a.imul_arr(np.ones((1, 1)) * 1 / 11)

        b = np.empty(3)
        a.copyto(([54, 66, 67], [877, 199, 202]), b)
        print(str(b))
        print(str(a.isfinite()))
        exit()

    main()
