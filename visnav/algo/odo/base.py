# Based on DT-SLAM article:
#  - "DT-SLAM: Deferred Triangulation for Robust SLAM", Herrera, Kim, Kannala, Pulli, Heikkila
#    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7035876
#
# code also available, it wasn't used for any reference though:
#  - https://github.com/plumonito/dtslam/tree/master/code/dtslam
#

# TODO IDEAS:
#   - use a prior in bundle adjustment that is updated with discarded keyframes
#   - use s/c state (pose, dot-pose, dot-dot-pose?) and covariance in ba optimization cost function
#   - option to use reduced frame ba for pose estimation instead of 3d-2d ransac from previous keyframe
#   - use miniSAM or GTSAM for optimization?
#       - https://github.com/dongjing3309/minisam
#       - https://gtsam.org/docs/

import os
import copy
import time
from datetime import datetime
import logging
import math
import threading
import multiprocessing as mp
from functools import lru_cache

import numpy as np
import quaternion   # adds to numpy  # noqa # pylint: disable=unused-import
import cv2

from visnav.algo import tools
from visnav.algo.bundleadj import vis_bundle_adj
from visnav.algo.featdet import detect_gridded
from visnav.algo.image import ImageProc


class Pose:
    def __init__(self, loc, quat: quaternion, loc_s2=None, so3_s2=None):
        self.loc = np.array(loc)
        self.quat = quat
        self.loc_s2 = np.array(loc_s2) if loc_s2 is not None else None
        self.so3_s2 = np.array(so3_s2) if so3_s2 is not None else None

    def __add__(self, dpose):
        assert isinstance(dpose, DeltaPose), 'Can only add DeltaPose to this'
        return Pose(
            self.loc + dpose.loc,
            dpose.quat * self.quat,
            self.loc_s2 + dpose.loc_s2,
            self.so3_s2 + dpose.so3_s2
        )

    def __sub__(self, pose):
        return DeltaPose(
            self.loc - pose.loc,
            pose.quat.conj() * self.quat,
            self.loc_s2 - pose.loc_s2,
            self.so3_s2 - pose.so3_s2
        )

    @staticmethod
    def initial():
        return Pose(np.array([0, 0, 0]), quaternion.one, np.ones((3,)) * 0.1, np.ones((3,)) * 0.01)


class DeltaPose(Pose):
    pass


class PoseEstimate:
    def __init__(self, prior: Pose, post: Pose, method):
        self.prior = prior
        self.post = post
        self.method = method


class Measure:
    def __init__(self, data, time_off, time_adj=0):
        self.data = data
        self.time_off = time_off
        self.time_adj = time_adj

class Frame:
    _NEXT_ID = 1

    def __init__(self, time, image, img_sc, pose: PoseEstimate, measure: Measure=None,
                 kps_uv: dict=None, kps_uv_norm: dict=None, kps_uv_vel: dict=None, id=None):
        self._id = id
        if id is not None:
            Frame._NEXT_ID = max(id + 1, Frame._NEXT_ID)
        self.time = time
        self.image = image
        self.img_sc = img_sc
        self.pose = pose
        self.measure = measure
        self.kps_uv = kps_uv or {}              # dict of keypoints with keypoint img coords in this frame
        self.kps_uv_norm = kps_uv_norm or {}    # dict of keypoints with undistorted img coords in this frame
        self.kps_uv_vel = kps_uv_vel or {}
        self.ini_kp_count = len(self.kps_uv)

    def set_id(self):
        assert self._id is None, 'id already given, cant set it twice'
        self._id = Frame._NEXT_ID
        Frame._NEXT_ID += 1

    def to_rel(self, pt3d, post=True):
        pose = getattr(self.pose, 'post' if post else 'prior')
        fun = tools.q_times_v if len(pt3d.shape) == 1 else tools.q_times_mx
        return fun(pose.quat, pt3d) + pose.loc.reshape((1, 3))

    def to_mx(self, post=True):
        pose = getattr(self.pose, 'post' if post else 'prior')
        return self._to_mx(tuple(pose.quat.components), tuple(pose.loc))

    @staticmethod
    @lru_cache(maxsize=4)
    def _to_mx(q, v):
        return np.hstack((quaternion.as_rotation_matrix(np.quaternion(*q)), np.array(v).reshape((-1, 1))))

    @property
    def id(self):
        return self._id

    def __hash__(self):
        return self.id


class Keypoint:
    _NEXT_ID = 1

    def __init__(self, id=None):
        self._id = Keypoint._NEXT_ID if id is None else id
        Keypoint._NEXT_ID = max(self._id + 1, Keypoint._NEXT_ID)
        self.pt3d = None
        self.pt3d_added_frame_id = None
        self.total_count = 0
        self.inlier_count = 0
        self.inlier_time = None
        self.active = True

    @property
    def id(self):
        return self._id

    def __hash__(self):
        return self._id


class State:
    def __init__(self):
        self.initialized = False
        self.keyframes = []
        self.map2d = {}  # id => Keypoint, all keypoints with only uv coordinates (still no 3d coords)
        self.map3d = {}
        self.last_frame = None
        self.last_success_time = None
        self.first_result_given = False
        self.first_mm_done = None
        self.tracking_failed = True


class VisualOdometry:
    # visual odometry/slam parts:
    #  - feature keypoint detector
    #  - feature matcher
    #  - pose estimator
    #  - triangulator
    #  - keyframe acceptance decider
    #  - keyframe addition logic
    #  - map maintainer (remove and/or adjust keyframes and/or features)
    #  - loop closer or absolute pose estimator
    POSE_2D2D, POSE_2D3D = range(2)

    DEF_MIN_KEYPOINT_DIST = 15
    DEF_MAX_KEYPOINTS = 315
    DEF_DETECTION_GRID = (3, 3)
    DEF_MIN_TRACKING_QUALITY = 0.0001

    DEF_INIT_BIAS_SDS = np.ones(6) * 5e-3   # bias drift sds, x, y, z, then so3
    DEF_INIT_LOC_SDS = np.ones(3) * 3e-2
    DEF_INIT_ROT_SDS = np.ones(3) * 3e-2
    DEF_LOC_SDS = np.ones(3) * 5e-3
    DEF_ROT_SDS = np.ones(3) * 1e-2

    # keyframe addition
    DEF_NEW_KF_MIN_INLIER_RATIO = 0.70             # remaining inliers from previous keyframe features
    DEF_NEW_KF_MIN_DISPL_FOV_RATIO = 0.008         # displacement relative to the fov for triangulation
    DEF_NEW_KF_TRIANGULATION_TRIGGER_RATIO = 0.2   # ratio of 2d points tracked that can be triangulated
    DEF_INI_KF_TRIANGULATION_TRIGGER = 20          # need at least this qty of 2d points that can be tri. for first kf
    DEF_NEW_KF_TRANS_KP3D_ANGLE = math.radians(3)  # change in viewpoint relative to a 3d point
    DEF_NEW_KF_TRANS_KP3D_RATIO = 0.15             # ratio of 3d points with significant viewpoint change
    DEF_NEW_KF_ROT_ANGLE = math.radians(7)         # new keyframe if orientation changed by this much
    DEF_NEW_SC_ROT_ANGLE = math.radians(7)         # new keyframe if s/c orientation changed by this much
    DEF_REPR_ERR_FOV_RATIO = 0.0005                # expected reprojection error (related to DEF_NEW_KF_MIN_DISPL_FOV_RATIO)
    DEF_MAX_REPR_ERR_FOV_RATIO = 0.003             # max tolerable reprojection error (related to DEF_NEW_KF_MIN_DISPL_FOV_RATIO)
    DEF_KF_BIAS_SDS = np.ones(6) * 5e-4            # bias drift sds, x, y, z, then so3

    # map maintenance
    DEF_MAX_KEYFRAMES = 8
    DEF_MAX_MARG_RATIO = 0.90
    DEF_REMOVAL_USAGE_LIMIT = 2        # 3d keypoint valid for removal if it was available for use this many times
    DEF_REMOVAL_RATIO = 0.15           # 3d keypoint inlier participation ratio below which the keypoint is discarded
    DEF_REMOVAL_AGE = 8                # remove if last inlier was this many keyframes ago
    DEF_MM_BIAS_SDS = np.ones(6) * 2e-3  # bias drift sds, x, y, z, then so3

    DEF_INLIER_RATIO_2D2D = 0.20        # expected ratio of inliers when using the 5-point algo for pose
    DEF_MIN_2D2D_INLIERS = 20           # discard pose estimate if less inliers than this   (was 60, with 7px min dist feats)
    DEF_MIN_INLIERS = 15                # discard pose estimate if less inliers than this   (was 35, with 7px min dist feats)
    DEF_MIN_INLIER_RATIO = 0.01         # discard pose estimate if less inliers than this   (was 0.08 for asteroid)
    DEF_RESET_TIMEOUT = 6               # reinitialize if this many seconds without successful pose estimate
    DEF_MIN_FEATURE_INTENSITY = 10      # min level of intensity required near a keypoint
    DEF_MAX_FEATURE_INTENSITY = 250     # need lower than this intensity near a keypoint
    DEF_POSE_2D2D_QUALITY_LIM = 0.1     # minimum pose result quality

    DEF_EST_2D2D_PROB = 0.99           # relates to max RANSAC iterations
    DEF_EST_2D2D_METHOD = cv2.RANSAC    # cv2.LMEDS is fast but inaccurate, cv2.RANSAC is the other choice

    DEF_USE_3D2D_RANSAC = True
    DEF_EST_3D2D_ITER_COUNT = 1000      # max RANSAC iterations
    DEF_EST_3D2D_METHOD = cv2.SOLVEPNP_AP3P  # RANSAC kernel function  # SOLVEPNP_AP3P

    DEF_NEW_KEYFRAME_BA = False
    DEF_USE_BA = True
    DEF_THREADED_BA = False             # run ba in an own thread
    DEF_MAX_BA_KEYFRAMES = 8
    DEF_BA_INTERVAL = 4                 # run ba every this many keyframes
    DEF_MAX_BA_FUN_EVAL = 30            # max cost function evaluations during ba

    def __init__(self, cam, img_width=None, wf_body_q: np.quaternion = None, bf_cam_pose: Pose = None,
                 verbose=0, pause=0, **kwargs):
        self.cam = cam
        self.img_width = img_width
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        self.pause = pause
        self.cam_mx = cam.intrinsic_camera_mx(legacy=True)
        self.wf_body_q = wf_body_q or quaternion.one
        self.bf_cam_pose = bf_cam_pose or Pose(np.array([0, 0, 0]), quaternion.one)

        # set params
        for attr in dir(self.__class__):
            if attr[:4] == 'DEF_':
                key = attr[4:].lower()
                setattr(self, key, kwargs.pop(key, getattr(self.__class__, attr)))
        assert len(kwargs) == 0, 'extra keyword arguments given: %s' % (kwargs,)

        self.lk_params = {
            'winSize': (32, 32),
            'maxLevel': 4,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05),
            'minEigThreshold': self.min_tracking_quality,
        }
        self.kp_params = {
            'maxCorners': self.max_keypoints,
            'qualityLevel': 0.05,  # default around 0.05?
            'minDistance': self.min_keypoint_dist,
            'blockSize': 4,
        }

        # state
        self.state = State()

        # removed frames and keypoints
        self.removed_keyframes = []
        self.removed_keypoints = []

        # current frame specific temp value cache
        self.cache = {}
        self._map_fig = None
        self._frame_count = 0
        self._track_save_path = None  # for debug purposes
        self._track_image = None    # for debug purposes
        self._track_colors = None   # for debug purposes
        self._map_image = None    # for debug purposes
        self._map_colors = None   # for debug purposes
        self._prev_map3d = {}     # for debug purposes
        self._init_map_center = None  # for debug purposes
        self._3d_map_lock = threading.RLock()
        self._new_frame_lock = threading.RLock()
        self._ba_started = []
        self._ba_max_keyframes = None

        if self.use_ba and self.threaded_ba:
            self._ba_start_lock = threading.Lock()
            self._ba_stop_lock = threading.Lock()
            self._ba_start_lock.acquire()     # start as locked
            parent = self

            def ba_runner():
                while True:
                    parent._ba_start_lock.acquire()
                    parent._bundle_adjustment()

            self._ba_thread = threading.Thread(target=ba_runner, name="ba-thread", daemon=True)
            self._ba_thread.start()

            if mp.cpu_count() > 1:
                ctx = mp.get_context('spawn')
                self._ba_arg_queue = ctx.Queue()
                self._ba_log_queue = ctx.Queue()
                self._ba_res_queue = ctx.Queue()
                self._ba_proc = ctx.Process(target=mp_bundle_adj,
                                            args=(self._ba_arg_queue, self._ba_log_queue, self._ba_res_queue))
                self._ba_proc.start()
            else:
                self._ba_arg_queue = None

    def __reduce__(self):
        return (self.__class__, (
            # init params, state
        ))

    def __del__(self):
        self.quit()

    def quit(self):
        cv2.destroyAllWindows()
        if self.use_ba and self.threaded_ba and self._ba_arg_queue:
            self._ba_arg_queue.put(('quit', None, tuple(), dict()))
            self._ba_proc.join()
            self._ba_arg_queue = None

    def repr_err(self, frame):
        return self.repr_err_fov_ratio * np.linalg.norm(np.array(frame.image.shape) / frame.img_sc)

    def max_repr_err(self, frame):
        return self.max_repr_err_fov_ratio * np.linalg.norm(np.array(frame.image.shape) / frame.img_sc)

    def process(self, new_img, new_time, measure=None) -> (PoseEstimate, np.ndarray, float):
        with self._new_frame_lock:
            return self._process(new_img, new_time, measure)

    def _process(self, new_img, new_time, measure) -> (PoseEstimate, np.ndarray, float):
        # reset cache
        self.cache.clear()

        # initialize new frame
        new_frame = self.initialize_frame(new_time, new_img, measure)

        # maybe initialize state
        if not self.state.initialized:
            self.initialize_track(new_frame)
            return copy.deepcopy(new_frame), None

        # track/match keypoints
        self.track_keypoints(new_frame)

        # estimate pose
        self.estimate_pose(new_frame)

        # remove 3d points that haven't contributed to poses lately
        self.prune_map3d()

        # maybe do failure recovery
        if new_frame.pose.post is None:
            dt = (new_frame.time - self.state.last_success_time).total_seconds()
            if dt > self.reset_timeout:
                # if fail for too long, reinitialize (excl if only one keyframe)
                self.state.initialized = False
            elif not self.state.tracking_failed:
                # frame maybe corrupted, fail once and try again next time
                logging.info('Image maybe corrupted, failing one frame')
                self.state.tracking_failed = True
                return copy.deepcopy(new_frame), None
            elif len(new_frame.kps_uv) < self.min_2d2d_inliers / self.inlier_ratio_2d2d:
                # if too few keypoints tracked for E-mat estimation, reinitialize
                self.state.initialized = False
        else:
            self.state.tracking_failed = False

        # expected bias sds
        bias_sds = np.zeros((6,))

        # add new frame as keyframe?      # TODO: (3) run in another thread
        if self.is_new_keyframe(new_frame):
            self.add_new_keyframe(new_frame)
            bias_sds = self.kf_bias_sds
            logging.info('new keyframe (ID=%s) added' % new_frame.id)

            # maybe do map maintenance    # TODO: (3) run in yet another thread
            if self.is_maintenance_time():
                self.maintain_map()
                bias_sds = self.mm_bias_sds
                self.state.first_mm_done = self.state.first_mm_done or self.state.keyframes[-1].id

        elif new_frame.pose.method == VisualOdometry.POSE_2D2D:
            # 2d-2d match result only usable for the first added keyframe,
            # i.e. for the first frames after init that are not keyframes, dont return pose
            new_frame.pose.post = None

        if new_frame.pose.post is not None and not self.state.first_result_given:
            bias_sds = self.init_bias_sds
            self.state.first_result_given = True

        if self.use_ba and new_frame.pose.post is not None and \
                (not self.state.first_mm_done or self.state.first_mm_done == self.state.keyframes[-1].id):
            # initialized but no bundle adjustment done yet (or the first one just done)
            # => higher uncertainties
            bias_sds = self.init_bias_sds
            new_frame.pose.post.loc_s2 = self.init_loc_sds
            new_frame.pose.post.so3_s2 = self.init_rot_sds

        self.state.last_frame = new_frame
        return copy.deepcopy(new_frame), bias_sds

    def initialize_frame(self, time, image, measure):
        logging.info('new frame')
        self._frame_count += 1

        # maybe scale image
        img_sc = 1
        if self.img_width is not None:
            img_sc = self.img_width / image.shape[1]
            image = cv2.resize(image, None, fx=img_sc, fy=img_sc)

        lp = self.state.last_frame.pose.post if self.state.last_frame else Pose.initial()
        nf = Frame(time, image, img_sc, PoseEstimate(prior=lp, post=None, method=None), measure=measure)
        nf.ini_kp_count = self.state.keyframes[-1] if len(self.state.keyframes) > 0 else None
        return nf

    def flush_state(self):
        self.removed_keyframes.extend(self.state.keyframes)
        self.removed_keypoints.extend(self.state.map3d)

    def initialize_track(self, new_frame):
        if self.state is not None:
            self.flush_state()

        logging.info('initializing tracking')
        self.state = State()
        new_frame.pose.post = new_frame.pose.prior
        self.add_new_keyframe(new_frame)

        # check that init ok and enough features found
        if len(self.state.map2d) > self.min_inliers * 2:
            self.state.last_frame = new_frame
            self.state.last_success_time = new_frame.time
            self.state.initialized = True

    def check_features(self, image, in_kp2d):
        # check that not too close to image border
        w, h = self.img_width, round(image.shape[0] * self.img_width / image.shape[1])
        mask = np.logical_and.reduce((in_kp2d[:, 0, 0] >= self.min_keypoint_dist,
                                      in_kp2d[:, 0, 1] >= self.min_keypoint_dist,
                                      in_kp2d[:, 0, 0] <= w - self.min_keypoint_dist,
                                      in_kp2d[:, 0, 1] <= h - self.min_keypoint_dist))
        return mask

    def detect_features(self, new_frame):
        kp2d = self._detect_features(new_frame.image, new_frame.kps_uv.values())
        pt2d_norm = self.cam.undistort(np.array(kp2d) / new_frame.img_sc)

        for i, pt in enumerate(kp2d):
            kp = Keypoint()  # Keypoint({new_frame.id: pt})
            new_frame.kps_uv[kp.id] = pt
            new_frame.kps_uv_norm[kp.id] = pt2d_norm[i]
            self.state.map2d[kp.id] = kp
        new_frame.ini_kp_count = len(new_frame.kps_uv)

        logging.info('%d new keypoints detected' % len(kp2d))

    def _feature_detection_mask(self, image):
        return 255 * np.ones(image.shape, dtype=np.uint8)

    def _detect_features(self, image, kp2d, existing=False):
        # TODO: (3) project non-active 3d points to current image plane before detecting new features?

        # create mask defining where detection is to be done
        mask = self._feature_detection_mask(image)
        if existing:
            mask_a = mask
            mask = np.zeros(image.shape, dtype=np.uint8)

        d, (h, w) = self.min_keypoint_dist, mask.shape
        for uv in kp2d:
            if existing:
                y, x = int(uv[0, 1]), int(uv[0, 0])
                mask[y:min(h, y+1), x:min(w, x+1)] = mask_a[y:min(h, y+1), x:min(w, x+1)]
            else:
                y0, y1 = max(0, int(uv[0, 1]) - d), min(h, int(uv[0, 1]) + d)
                x0, x1 = max(0, int(uv[0, 0]) - d), min(w, int(uv[0, 0]) + d)
                if x1 > x0 and y1 > y0:
                    mask[y0:y1, x0:x1] = 0

        if np.all(mask == 0):
            logging.warning('no features detectable because of masking')
            return []

        # detect Shi-Tomasi keypoints
        det = cv2.GFTTDetector_create(**self.kp_params)

        # detect features in a grid so that features are spread out
        kp2d = detect_gridded(det, image, mask, *self.detection_grid, self.max_keypoints)
        kp2d = self.kp2arr(kp2d)

        return [] if kp2d is None else kp2d

    def is_active_kp(self, id):
        return id in self.state.map2d or id in self.state.map3d and self.state.map3d[id].active

    def track_keypoints(self, new_frame):
        lf, nf = self.state.last_frame, new_frame

        if len(lf.kps_uv) == 0:
            return

        # track keypoints using Lukas-Kanade method
        tmp = [(id, uv) for id, uv in lf.kps_uv.items() if self.is_active_kp(id)]
        ids, old_kp2d = list(map(np.array, zip(*tmp) if len(tmp) > 0 else ([], [])))
        new_kp2d, mask, err = cv2.calcOpticalFlowPyrLK(lf.image, nf.image, old_kp2d, None, **self.lk_params)

        # extra sanity check on tracked points, set mask to false if keypoint quality too poor
        mask2 = self.check_features(nf.image, new_kp2d)
        if mask2 is not None:
            mask = np.logical_and(mask.astype(np.bool).flatten(), mask2)
        else:
            mask = mask.astype(np.bool).flatten()

        new_kp2d = new_kp2d[mask, :, :]

        # mark non-tracked 3d-keypoints belonging to at least two keyframes as non-active, otherwise delete
        with self._3d_map_lock:
            for id in ids[np.logical_not(mask)]:
                self.del_keypoint(id, kf_lim=2)

        ids = ids[mask]
        dt = (nf.time - lf.time).total_seconds()

        nf.kps_uv = {id: uv for id, uv in zip(ids, new_kp2d)}
        nf.kps_uv_norm = {id: uv for id, uv in zip(ids, self.cam.undistort(np.array(new_kp2d) / nf.img_sc))}
        nf.kps_uv_vel = {id: (uv - lf.kps_uv_norm[id]) / dt for id, uv in nf.kps_uv_norm.items()}

        logging.info('Tracking: %d/%d' % (len(new_kp2d), len(old_kp2d)))

    def initialization(self, ref_frame, new_frame):
        rf, nf = ref_frame, new_frame

        # include all tracked keypoints, i.e. also 3d points
        # TODO: (3) better to compare rf post to nf prior?
        tmp = [(id, pt2d, rf.kps_uv_norm[id])
               for id, pt2d in nf.kps_uv_norm.items()
               if id in rf.kps_uv_norm]
        ids, new_kp2d, old_kp2d = list(map(np.array, zip(*tmp) if len(tmp) > 0 else ([], [], [])))

        R = None
        mask = 0
        if len(old_kp2d) >= self.min_2d2d_inliers / self.inlier_ratio_2d2d:
            # solve pose using ransac & 5-point algo
            E, mask2 = cv2.findEssentialMat(old_kp2d, new_kp2d, self.cam_mx, method=self.est_2d2d_method,
                                            prob=self.est_2d2d_prob, threshold=self.repr_err(nf))
            logging.info('E-mat: %d/%d' % (np.sum(mask2), len(old_kp2d)))

            if np.sum(mask2) >= self.min_2d2d_inliers:
                _, R, ur, mask = cv2.recoverPose(E, old_kp2d, new_kp2d, self.cam_mx, mask=mask2.copy())
                logging.info('E=>R: %d/%d' % (np.sum(mask), np.sum(mask2)))
                inliers = np.where(mask)[0]
                e_mx_qlt = self.pose_result_quality(rf, nf, dq=quaternion.from_rotation_matrix(R),
                                                    inlier_ids=ids[inliers], plot=0)
                if e_mx_qlt < self.pose_2d2d_quality_lim:
                    logging.info('Pose result quality too low: %.3f' % e_mx_qlt)
                    R = None

        if R is not None and len(inliers) >= self.min_2d2d_inliers \
                and len(inliers) / np.sum(mask2) >= self.min_inlier_ratio:

            # record keypoint stats
            with self._3d_map_lock:
                for id in ids:
                    self.state.map2d[id].total_count += 1
                for i in inliers.flatten():
                    self.state.map2d[ids[i]].inlier_count += 1
                    self.state.map2d[ids[i]].inlier_time = nf.time

            dq = quaternion.from_rotation_matrix(R)
            dr = ur.flatten()
            return dr, dq
        return None, None

    def estimate_pose(self, new_frame):
        rf, lf, nf = self.state.keyframes[-1], self.state.last_frame, new_frame
        dr, dq = None, None
        inliers = None
        method = None

        # solve pose using ransac & ap3p based on 3d-2d matches

        tmp = [(id, pt2d, self.state.map3d[id].pt3d)
               for id, pt2d in nf.kps_uv_norm.items()
               if id in self.state.map3d and self.state.map3d[id].active]
        ids, pts2d, pts3d = list(map(np.array, zip(*tmp) if len(tmp) else ([], [], [])))
        logging.info('Tracked 3D-points: %d/%d' % (len(pts3d), len(nf.kps_uv)))

        ok = False
        if len(pts3d) >= self.min_inliers:
            if self.use_3d2d_ransac:
                ok, rv, r, inliers = cv2.solvePnPRansac(pts3d, pts2d, self.cam_mx, None,
                                                        iterationsCount=self.est_3d2d_iter_count,
                                                        reprojectionError=self.repr_err(nf),
                                                        flags=self.est_3d2d_method)

                logging.info('PnP: %d/%d' % (0 if inliers is None else len(inliers), len(pts2d)))
            else:
                self._call_ba()

                logging.info('BA-1: %d' % (len(pts2d),))

        if ok and len(inliers) >= self.min_inliers and len(inliers)/len(pts3d) > self.min_inlier_ratio:
            q = tools.angleaxis_to_q(rv)

            if 0:
                import matplotlib.pyplot as plt
                a = np.array([nf.kps_uv[id] for id in ids]).squeeze()
                b = self.cam.calc_img_R(nf.to_rel(pts3d, post=False), distort=True, legacy=True)*nf.img_sc
                nf.pose.post = Pose(r, q)
                c = self.cam.calc_img_R(nf.to_rel(pts3d, post=True), distort=True, legacy=True)*nf.img_sc
                plt.figure(1, figsize=(8, 6))
                plt.imshow(nf.image)
                plt.plot(a[:, 0], a[:, 1], 'bx')
                plt.plot(b[:, 0], b[:, 1], 'rx')
                plt.plot(c[:, 0], c[:, 1], 'gx')
                plt.plot(c[inliers, 0], c[inliers, 1], 'go', mfc='none')
                plt.tight_layout()
                plt.xlim(-50, nf.image.shape[1] + 50)
                plt.ylim(nf.image.shape[0] + 50, -50)
                plt.show()

            # record keypoint stats
            for id in ids:
                self.state.map3d[id].total_count += 1
            for i in inliers.flatten():
                self.state.map3d[ids[i]].inlier_count += 1
                self.state.map3d[ids[i]].inlier_time = nf.time

            # calculate delta-q and delta-r
            dq = q * rf.pose.post.quat.conj()

            # solvePnPRansac apparently randomly gives 180deg wrong answer,
            #  - too high translation in correct direction, why? related to delayed application of ba result?
            if abs(tools.q_to_ypr(dq)[0]) > math.pi * 0.9:
                logging.warning('rotated pnp-ransac solution by 180deg around z-axis')
                q_fix = tools.ypr_to_q(math.pi, 0, 0)
                dq = dq * q_fix
                q = dq * rf.pose.post.quat
                r = tools.q_times_v(q_fix, r)

            dr = r.flatten() - tools.q_times_v(dq, rf.pose.post.loc)
            method = VisualOdometry.POSE_2D3D
        else:
            if inliers is None:
                logging.info(' => Too few 3D points matched for reliable pose estimation')
            elif len(inliers) < self.min_inliers:
                logging.info(' => PnP was left with too few inliers')
            elif len(inliers)/len(pts3d) < self.min_inlier_ratio:
                logging.info(' => PnP too few inliers compared to total matches')
            else:
                logging.info(' => PnP Failed')

            dr, dq = None, None

            if self.state.first_result_given and self.state.tracking_failed:  # give one chance with tracking_failed
                logging.info(' => clearing all 3d points')
                self.state.initialized = False

        if not self.state.first_result_given and dr is None:
            dr, dq = self.initialization(rf, nf)
            method = VisualOdometry.POSE_2D2D

        if dr is None or dq is None or np.isclose(dq.norm(), 0):
            nf.pose.post = None
        else:
            # TODO: (2) calculate uncertainty / error var
            # d_r_s2 = np.ones(3) * 0.1
            # d_so3_s2 = np.ones(3) * 0.01

            # update pose and uncertainty
            nf.pose.post = Pose(
                tools.q_times_v(dq, rf.pose.post.loc) + dr,
                (dq * rf.pose.post.quat).normalized(),
                self.loc_sds,
                self.rot_sds,
                #rf.pose.post.loc_s2 + tools.q_times_v(rf.pose.post.quat, d_r_s2),
                #rf.pose.post.so3_s2 + tools.q_times_v(rf.pose.post.quat, d_so3_s2),
            )
            nf.pose.method = method
            self.state.last_success_time = nf.time

        if dr is not None and dq is not None:
            self._log_pose_diff('prior->post', nf.pose.prior.loc, nf.pose.prior.quat, nf.pose.post.loc, nf.pose.post.quat)

        if self.verbose:
            self._draw_tracks(nf, pause=self.pause)
            if self.verbose > 1:
                self._draw_pts3d(nf, pause=self.pause, plt3d=True)

    def _log_pose_diff(self, title, r0, q0, r1, q1):
        dq = q1 * q0.conj()
        dr = r1 - tools.q_times_v(dq, r0)
        logging.info(title + ' dq: ' + ' '.join(['%.3fdeg' % math.degrees(a) for a in tools.q_to_ypr(dq)])
              + '   dv: ' + ' '.join(['%.3fm' % a for a in dr]))

    def pose_result_quality(self, rf, nf, dq=None, inlier_ids=None, plot=False):
        if dq is None:
            dq = nf.pose.post.quat * rf.pose.post.quat.conj()
        if inlier_ids is None:
            inlier_ids = [id for id, kp in self.state.map2d.items()
                            if id in nf.kps_uv and id in rf.kps_uv and kp.inlier_count > 0]
        if len(inlier_ids) == 0:
            return 0.0

        pt2_o = np.array([p for id, p in rf.kps_uv_norm.items() if id in inlier_ids])[:, 0, :]
        pt2_n = np.array([p for id, p in nf.kps_uv_norm.items() if id in inlier_ids])[:, 0, :]
        pt2_o2 = tools.q_times_img_coords(dq, pt2_o, self.cam, distort=False, opengl=False)

        # calculate angles and norms of (pt2_o2 - pt2_n), calculate a stat to describe diverseness/uniformity
        x = (pt2_o2 - pt2_n).dot(np.array([1, 1j]))
        d = np.abs(x)
        a = np.angle(x)
        qlt = 0.5 * np.std(d) / np.mean(d) + 0.5 * np.std(a) / np.pi

        if plot:  # or qlt < 0.1:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.imshow(nf.image)
            ax.plot(pt2_n[:, 0] * nf.img_sc, pt2_n[:, 1] * nf.img_sc, 'o')
            ax.plot(pt2_o2[:, 0] * nf.img_sc, pt2_o2[:, 1] * nf.img_sc, 'x')
            plt.title('essential mx qlt [0-1]: %.3f' % qlt)
            plt.show()
            print('')

        return qlt

    def is_new_keyframe(self, new_frame):
        # check if
        #   a) should detect new feats as old ones don't work,
        #   b) can triangulate many 2d points,
        #   c) viewpoint changed for many 3d points, or
        #   d) orientation changed significantly
        #   e) prior orientation (=> phase angle => appearance) changed significantly

        rf, nf = self.state.keyframes[-1], new_frame

        # pose solution available
        if nf.pose.post is None:
            return False

        # if no kp triangulated yet
        if not self.state.first_result_given:
            return len(self.state.map2d) > 0 \
               and len(self.triangulation_kps(nf)) > self.ini_kf_triangulation_trigger

        #   a) should detect new feats as old ones don't work
        if len(nf.kps_uv)/rf.ini_kp_count < self.new_kf_min_inlier_ratio:
            logging.debug('new kf: too few feats')
            return True

        #   b) orientation changed significantly
        if tools.angle_between_q(nf.pose.post.quat, rf.pose.post.quat) > self.new_kf_rot_angle:
            logging.debug('new kf: orientation change')
            return True

        # #   c) sc orientation changed significantly
        # if tools.angle_between_q(nf.sc_q, rf.sc_q) > self.new_sc_rot_angle:
        #     return True

        #   d) viewpoint changed for many 3d points
        if self.use_ba \
                and len(self.state.map3d) > 0 \
                and len(self.viewpoint_changed_kps(nf))/len(self.state.map3d) > self.new_kf_trans_kp3d_ratio:
            logging.debug('new kf: viewpoint change')
            return True

        #   e) can triangulate many 2d points
        if len(self.state.map2d) > 0:
            k, n = len(self.triangulation_kps(nf)), len(self.state.map2d)
            actives = len([pt for pt in self.state.map3d.values() if pt.inlier_count > 0 and pt.active])
            if k > 0 and (k / n > self.new_kf_triangulation_trigger_ratio or actives < self.min_inliers * 2):
                logging.debug('new kf: triangulation %.0f%% (%d/%d)' % (100 * k/n, k, n))
                return True

        logging.debug('no new kf')
        return False

    def add_new_keyframe(self, new_frame):
        new_frame.set_id()
        self.state.keyframes.append(new_frame)
        if self.new_keyframe_ba and len(self.state.map3d) > 0:
            # poses = np.zeros((2, 7))
            # poses[0, :3] = tools.q_times_v(new_frame.pose.post.quat.conj(), -new_frame.pose.post.loc)
            # poses[0, 3:] = quaternion.as_float_array(new_frame.pose.post.quat.conj())
            self._bundle_adjustment(current_only=True)
            # poses[1, :3] = tools.q_times_v(new_frame.pose.post.quat.conj(), -new_frame.pose.post.loc)
            # poses[1, 3:] = quaternion.as_float_array(new_frame.pose.post.quat.conj())
            # tools.plot_poses(poses, axis=(0, 0, 1), up=(0, -1, 0))
        self.detect_features(new_frame)
        self.triangulate(new_frame)

    def triangulation_kps(self, new_frame):
        """
        return 2d keypoints that can be triangulated as they have more displacement than new_kf_min_displ_fov_ratio
        together with the corresponding reference frame
        """
        kpset = self.cache.get('triangulation_kps', None)
        if kpset is not None:
            return kpset

        n = len(self.state.keyframes)
        ids = [kp_id for kp_id in new_frame.kps_uv_norm.keys() if kp_id in self.state.map2d]
        k = len(ids)
        n_uv_arr = np.zeros((k, 2))
        r_uv_arr = np.ones((k, n, 2)) * np.nan
        for i, kp_id in enumerate(ids):
            n_uv_arr[i, :] = new_frame.kps_uv_norm[kp_id]
            for j, f in enumerate(self.state.keyframes):
                if kp_id in f.kps_uv_norm:
                    r_uv_arr[i, j, :] = f.kps_uv_norm[kp_id]

        D = np.ones((k, n)) * np.nan
        for j, f in enumerate(self.state.keyframes):
            idxs = np.where(np.logical_not(np.isnan(r_uv_arr[:, j, 0])))[0]
            if len(idxs) > 0:
                p_uvs = tools.q_times_img_coords(new_frame.pose.post.quat * f.pose.post.quat.conj(),
                                                 r_uv_arr[idxs, j, :], self.cam, opengl=False, distort=False)
                D[idxs, j] = np.linalg.norm(p_uvs - n_uv_arr[idxs, :], axis=1)

        # return dict(kp_id => ref_frame)
        kpset = {}
        fov_d = np.linalg.norm(np.array(new_frame.image.shape) / new_frame.img_sc)
        ok = np.where(np.any(np.logical_not(np.isnan(D)), axis=1))[0]
        f_idxs = np.nanargmax(D[ok, :], axis=1)
        for d, kp_id, fr_idx in zip(D[ok], np.array(ids)[ok], f_idxs):
            if d[fr_idx] / fov_d > self.new_kf_min_displ_fov_ratio:
                kpset[kp_id] = self.state.keyframes[fr_idx]

        self.cache['triangulation_kps'] = kpset
        return kpset

    def viewpoint_changed_kps(self, new_frame):
        """
        return 3d keypoints that are viewed from a significantly different angle compared to previous keyframe,
        return values is a set of keypoint ids
        """
        kpset = self.cache.get('viewpoint_changed_kps', None)
        if kpset is not None:
            return kpset

        ref_frame = self.state.keyframes[-1]

        tmp = [(id, self.state.map3d[id].pt3d)
               for id, uv in new_frame.kps_uv.items()
               if id in self.state.map3d]
        ids, pts3d = list(map(np.array, zip(*tmp) if len(tmp) > 0 else ([], [])))
        #pts3d = np.array(pts3d).reshape(-1, 3)
        dloc = (new_frame.pose.post.loc - ref_frame.pose.post.loc).reshape(1, 3)
        da = tools.angle_between_rows(pts3d, pts3d + dloc)
        kpset = set(ids[da > self.new_kf_trans_kp3d_angle])

        self.cache['viewpoint_changed_kps'] = kpset
        return kpset

    def triangulate(self, new_frame):
        # triangulate matched 2d keypoints to get 3d keypoints
        tr_kps = self.triangulation_kps(new_frame)

        # transformation from camera to world coordinate origin
        T1 = new_frame.to_mx()

        added_count = 0
        f_pts3d, frames = {}, {}
        for kp_id, ref_frame in tr_kps.items():
            # TODO:
            #  - check how is done in vins, need kalman filter?
            #  - do 1-frame ba after triangulation
            #  - use multipoint triangulation instead
            #  - triangulation with optimization over a distance prior

            T0 = ref_frame.to_mx()
            uv0 = ref_frame.kps_uv_norm[kp_id]
            uv1 = new_frame.kps_uv_norm[kp_id]
            kp4d = cv2.triangulatePoints(self.cam_mx.dot(T0), self.cam_mx.dot(T1),
                                         uv0.reshape((-1, 1, 2)), uv1.reshape((-1, 1, 2)))
            pt3d = (kp4d.T[:, :3] / kp4d.T[:, 3:])[0]

            if ref_frame.id not in f_pts3d:
                f_pts3d[ref_frame.id] = []
            f_pts3d[ref_frame.id].append((kp_id, pt3d))
            frames[ref_frame.id] = ref_frame

        inl_ids, inl_pts3d, removals = [], [], []
        for rf_id, ref_frame in frames.items():
            if not f_pts3d[rf_id]:
                continue
            ids, pts3d = map(np.array, zip(*f_pts3d[rf_id]))
            inl, _, rem = self.sanity_check_pt3d(ids, pts3d, [ref_frame, new_frame])
            inl_ids.extend(ids[inl])
            inl_pts3d.extend(pts3d[inl])
            removals.extend(ids[rem])

        with self._3d_map_lock:
            for kp_id in removals:
                self.del_keypoint(kp_id)

            for kp_id, pt3d in zip(inl_ids, inl_pts3d):
                kp = self.state.map2d.pop(kp_id)
                kp.pt3d = pt3d
                kp.pt3d_added_frame_id = new_frame.id
                self.state.map3d[kp_id] = kp
                added_count += 1

        logging.info('%d/%d keypoints successfully triangulated' % (added_count, len(tr_kps)))

    def sanity_check_pt3d(self, ids, pts3d, frames, max_dist_coef=2):
        if len(ids) == 0:
            return [], [], []

        ids = np.array(ids)
        mask = np.zeros((len(ids),), dtype=np.uint8)

        #  check that at least two frames has point in front, if not, discard
        pts3d_fs = np.array([f.to_rel(pts3d) for f in frames])
        mask[np.sum(pts3d_fs[:, :, 2] > 0, axis=0) < 2] = 1

        #  check that distance to the point is not too much, parallax should be high enough
        fov_d_rad = math.radians(np.linalg.norm(np.array([self.cam.x_fov, self.cam.y_fov])))
        idx = np.argmax(tools.distance_mx(np.array([f.pose.post.loc for f in frames])))
        i, j = np.unravel_index(idx, (len(frames),)*2)
        sep = tools.parallax(frames[i].pose.post.loc, frames[j].pose.post.loc, pts3d[mask == 0])
        idxs = np.where(mask == 0)[0]
        mask[idxs[sep / fov_d_rad * max_dist_coef < self.max_repr_err_fov_ratio]] = 1

        # check that reprojection error is not too large
        idxs = np.where(mask == 0)[0]
        kps4d = np.hstack((pts3d[idxs], np.ones((len(idxs), 1))))

        for f in frames[-3:]:   # TODO: for now only 3 latest frames used, somehow got too many rejections after ba
            norm_err, max_err = self.repr_err(f), self.max_repr_err(f)
            tmp = np.array([[i, f.kps_uv_norm[id]] for i, id in enumerate(ids[idxs]) if id in f.kps_uv_norm]).T
            if len(tmp) > 0:
                kp_idxs, uvs = tmp[0].astype(int), np.concatenate(tmp[1, :])
                proj_pts2d = self.cam_mx.dot(f.to_mx()).dot(kps4d[kp_idxs, :].T).T
                err = np.linalg.norm(uvs - proj_pts2d[:, :2]/proj_pts2d[:, 2:], axis=1)
                mask[idxs[kp_idxs]] = np.bitwise_or.reduce((mask[idxs[kp_idxs]],
                                                            1*(err > norm_err).astype(np.int),
                                                            2*(err > max_err).astype(np.int)))

        good = np.where(mask == 0)[0]
        wait = np.where(mask == 1)[0]
        remove = np.where(mask > 1)[0]
        return good, wait, remove

    def bundle_adjustment(self, max_keyframes=None, sync=False):
        self._ba_max_keyframes = max_keyframes
        if len(self._ba_started) <= 1:
            self._ba_started.append((self.state.keyframes[-1].id, time.time()))
        else:
            logging.warning('bundle adjustment in queue dropped')
            self._ba_started[-1] = (self.state.keyframes[-1].id, time.time())

        if self.threaded_ba:
            logging.info('giving go-ahead for bundle adjustment')
            self._ba_stop_lock.acquire(blocking=False)
            try:
                self._ba_start_lock.release()   # start ba in it's thread
            except RuntimeError:
                pass  # if already unlocked, no problem

            if sync:
                # lock released at the end of ba
                self._ba_stop_lock.acquire()
        else:
            self._bundle_adjustment()

    def _get_visual_ba_args(self, current_only=False):
        with self._3d_map_lock:
            if current_only:
                max_keyframes = 1
            else:
                max_keyframes = len(self.state.keyframes)
                max_keyframes = max_keyframes if self._ba_max_keyframes is None else (self._ba_max_keyframes + 1)

            keyframes = self.state.keyframes[-max_keyframes:]
            tmp = [
                (pt.id, pt.pt3d)
                for pt in self.state.map3d.values()
                if pt.inlier_count > 0  # only include if been an inlier
            ]

        if len(tmp) == 0:
            return
        ids, pts3d = map(np.array, zip(*tmp))
        idmap = dict(zip(ids, np.arange(len(ids))))

        # poses_mx = np.array([
        #     np.hstack((
        #         tools.q_to_angleaxis(f.pose.post.quat, compact=True),            # already in cam -> world
        #         f.pose.post.loc.flatten()
        #     ))
        #     for f in keyframes
        # ])

        # flip to world -> cam
        poses_mx = np.array([
            np.hstack((
                tools.q_to_angleaxis(f.pose.post.quat.conj(), compact=True),
                tools.q_times_v(f.pose.post.quat.conj(), -f.pose.post.loc).flatten()
            ))
            for f in keyframes
        ])

        cam_idxs, pt3d_idxs, pts2d, v_pts2d = list(map(np.array, zip(*[
            (i, idmap[id], uv.flatten(), f.kps_uv_vel.get(id, np.array([0, 0])).flatten())
            for i, f in enumerate(keyframes)
                for id, uv in f.kps_uv_norm.items() if id in idmap
        ])))

        return keyframes, ids, poses_mx, pts3d, pts2d, v_pts2d, cam_idxs, pt3d_idxs

    def _call_ba(self, ba_fun, args, kwargs):
        if not self.threaded_ba or self._ba_arg_queue is None:
            res = ba_fun(*args, **kwargs)
        else:
            self._ba_arg_queue.put(('ba', ba_fun, args, kwargs))
            res = self._ba_res_queue.get()
            try:
                while True:
                    level, msg = self._ba_log_queue.get(block=False)
                    getattr(logging, level)(msg)
            except:
                pass
        return res

    def _bundle_adjustment(self, current_only=False):
        logging.info('starting bundle adjustment')

        keyframes, ids, poses_mx, pts3d, pts2d, v_pts2d, cam_idxs, pt3d_idxs = self._get_visual_ba_args(current_only)
        skip_pose_n = 1 if not current_only else 0

        args = (poses_mx, pts3d, pts2d, cam_idxs, pt3d_idxs, self.cam_mx)
        kwargs = dict(max_nfev=self.max_ba_fun_eval, skip_pose_n=skip_pose_n, poses_only=current_only,
                      huber_coef=self.repr_err(keyframes[-1]))

        poses_ba, pts3d_ba = self._call_ba(vis_bundle_adj, args, kwargs)

        self._update_poses(keyframes, ids, poses_ba, pts3d_ba, skip_pose_n=skip_pose_n, pop_ba_queue=not current_only)

    def _update_poses(self, keyframes, ids, poses_ba, pts3d_ba, skip_pose_n=1, pop_ba_queue=True):
        if np.any(np.isnan(poses_ba)) or np.any(np.isnan(pts3d_ba)):
            logging.warning('bundle adjustment results in invalid 3d points')
            return

        with self._new_frame_lock:
            for i, p in enumerate(poses_ba):
                f = keyframes[i + skip_pose_n]
                rn, qn = p[3:], tools.angleaxis_to_q(p[:3])

                # flip from world -> cam into cam -> world
                rn = tools.q_times_v(qn.conj(), -rn)
                qn = qn.conj()

                if i == len(poses_ba) - 1:
                    # last frame adjusted by this rotation and translation
                    ro, qo = f.pose.post.loc, f.pose.post.quat
                    dr, dq = rn - ro, qn * qo.conj()        # TODO: or is it qo.conj() * qn
                f.pose.post.loc, f.pose.post.quat = rn, qn

            # sanity check for pts3d_ba:
            #  - active 3d keypoints are in front of at least two keyframes
            #  - reprojection error is acceptable
            #  - parallax is enough
            if len(pts3d_ba) > 0:
                good, borderline, rem_idxs = self.sanity_check_pt3d(ids, pts3d_ba, keyframes, max_dist_coef=3)
                inliers = np.concatenate((good, borderline))
                rem_ids, good_ids, good_pts3d = ids[rem_idxs], ids[inliers], pts3d_ba[inliers]
            else:
                rem_ids, good_ids, good_pts3d = [[]] * 3

            for kp_id in rem_ids:
                if kp_id in self.state.map3d:
                    self.del_keypoint(kp_id)

            # transform 3d map so that consistent with first frame being unchanged
            for kp_id, pt3d in zip(good_ids, good_pts3d):
                if kp_id in self.state.map3d:
                    self.state.map3d[kp_id].pt3d = pt3d

            # update keyframes that are newer than those included in the bundle adjustment run
            #   - also update recently triangulated new 3d points
            new_keyframes = [kf for kf in self.state.keyframes if kf.id > keyframes[-1].id]
            new_3d_kps = [kp for kp in self.state.map3d.values() if kp.pt3d_added_frame_id > keyframes[-1].id]

            if len(new_keyframes) > 0:
                for f in new_keyframes:
                    f.pose.post.quat = dq * f.pose.post.quat       # TODO: or is it f.pose.post.quat * dq
                    cnr = f.pose.post.loc + dr
                    f.pose.post.loc = cnr
                for kp in new_3d_kps:
                    kp.pt3d = tools.q_times_v(dq, kp.pt3d + dr)    # TODO: or is it q_times_v(dq, kp.pt3d) + dr

            if pop_ba_queue:
                st_kf_id, st_t = self._ba_started.pop(0)
                c = self.state.keyframes[-1].id - st_kf_id
                t = (time.time() - st_t)
                self._ba_started_kf_id = None
                logging.info('bundle adjustment complete after %d keyframes and %.2fs, queue len: %d'
                             % (c, t, len(self._ba_started)))

            if self.threaded_ba:
                try:
                    self._ba_stop_lock.release()   # only used when running ba in synchronized mode
                except RuntimeError:
                    pass  # can't happen if in synchronized mode

    def is_maintenance_time(self):
        return (self.state.keyframes[-1].id - self.state.keyframes[0].id + 1) % self.ba_interval == 0

    def maintain_map(self):
        if len(self.state.keyframes) > self.max_keyframes:
            self.prune_keyframes()

        # Remove 3d keypoints from map.
        # No need to remove 2d keypoints here as they are removed
        # elsewhere if 1) tracking fails, or 2) triangulated into 3d points.
        self.prune_map3d(inactive=True)

        if self.use_ba and len(self.state.map3d) >= self.min_inliers:
            self.bundle_adjustment(max_keyframes=self.max_ba_keyframes)

    def prune_keyframes(self):
        with self._3d_map_lock:
            rem_kfs = self.state.keyframes[:-self.max_keyframes]
            self.state.keyframes = self.state.keyframes[-self.max_keyframes:]
            self.removed_keyframes.extend(rem_kfs)
            # no need to remove 3d points as currently only those are retained that can be tracked
            # no need to transform poses or 3d points as basis is the absolute pose
            logging.info('%d keyframes dropped' % len(rem_kfs))

    def prune_map3d(self, inactive=False):
        with self._3d_map_lock:
            rem = []
            removed = 0
            lim, kfs = min(self.max_keyframes, self.max_ba_keyframes), self.state.keyframes
            for id, kp in self.state.map3d.items():
                if not inactive and not kp.active:
                    continue
                if (kp.total_count >= self.removal_usage_limit
                        and kp.inlier_count / kp.total_count <= self.removal_ratio) \
                        or (len(kfs) >= lim and kp.pt3d_added_frame_id < kfs[-lim].id
                            and (kp.inlier_time is None or kp.inlier_time <= kfs[-lim].time)):
                    rem.append(id)
            for id in rem:
                removed += self.del_keypoint(id, kf_lim=2)
            logging.info('%d 3d keypoints removed or marked as inactive' % removed)

    def del_keypoint(self, id, kf_lim=None):
        if kf_lim is not None and id in self.state.map3d:
            lim = min(self.max_keyframes, self.max_ba_keyframes)
            if len([1 for f in self.state.keyframes[-lim:] if id in f.kps_uv]) >= kf_lim:
                ret = int(self.state.map3d[id].active)
                self.state.map3d[id].active = False
                return ret

        ret = int(id in self.state.map3d)
        if ret:
            self.removed_keypoints.append(self.state.map3d[id])
            # TODO: (*) save f.kps_uv, f.kps_uv_norm

        self.state.map2d.pop(id, False)
        self.state.map3d.pop(id, False)
        for f in self.state.keyframes:
            f.kps_uv.pop(id, False)
            f.kps_uv_norm.pop(id, False)
            f.kps_uv_vel.pop(id, False)
        self.state.last_frame.kps_uv.pop(id, False)
        self.state.last_frame.kps_uv_norm.pop(id, False)
        self.state.last_frame.kps_uv_vel.pop(id, False)
        return ret

    def arr2kp(self, arr, size=7):
        return [cv2.KeyPoint(p[0, 0], p[0, 1], size) for p in arr]

    def kp2arr(self, kp):
        return np.array([k.pt for k in kp], dtype='f4').reshape((-1, 1, 2))

    def get_3d_map_pts(self):
        return np.array([pt.pt3d for pt in self.state.map3d.values()]).reshape((-1, 3))

    @staticmethod
    def get_2d_pts(frame):
        return np.array([uv.flatten() for uv in frame.kps_uv.values()]).reshape((-1, 2))

    def _col(self, id, fl=False, bgr=True):
        n = 1000
        if self._track_colors is None:
            self._track_colors = np.hstack((np.random.randint(0, 179, (n, 1)), np.random.randint(0, 255, (n, 1))))
        h, s = self._track_colors[id % n]
        v = min(255, 255 * ((self.state.map3d[id].inlier_count if id in self.state.map3d else 0) + 1) / 4)
        col = cv2.cvtColor(np.array([h, s, v], dtype=np.uint8).reshape((1, 1, 3)),
                           cv2.COLOR_HSV2BGR if bgr else cv2.COLOR_HSV2RGB).flatten()
        return (col/255 if fl else col).tolist()

    def distort_pts(self, uv):
        return uv

    def _draw_tracks(self, new_frame, pause=True, label='tracks'):
        f0, f1 = self.state.last_frame, new_frame
        ids, uv1 = zip(*[(id, uv.flatten()) for id, uv in f1.kps_uv.items()]) if len(f1.kps_uv) > 0 else ([], [])
        uv0 = [f0.kps_uv[id].flatten() for id in ids]

        id2proj = {}
        pts3d = np.array([self.state.map3d[id].pt3d for id in ids if id in self.state.map3d]).reshape((-1, 3))
        if new_frame.pose.post and len(pts3d) > 0:
            q_cf = new_frame.pose.post.quat.conj()
            v_cf = tools.q_times_v(q_cf, -new_frame.pose.post.loc)

            if 0:
                pts3d_cf = tools.q_times_mx(q_cf.conj(), pts3d - v_cf)
                uvph = self.cam_mx.dot(pts3d_cf.T).T
                uvp = new_frame.img_sc * uvph[:, :2] / uvph[:, 2:]
            elif 0:
                from .vis_gps_bundleadj import project
                poses = np.array([[*tools.q_to_angleaxis(q_cf, compact=True), *v_cf]])
                uvp = project(pts3d.astype(np.float32), poses, self.cam_mx)
                uvp *= new_frame.img_sc
            else:
                pts3d_cf = tools.q_times_mx(q_cf.conj(), pts3d - v_cf)
                uvp = new_frame.img_sc * self.cam.project(pts3d_cf.astype(np.float32))

            id2proj = [id for id in ids if id in self.state.map3d]
            id2proj = dict(zip(id2proj, range(len(id2proj))))

        if self._track_image is None:
            self._track_image = np.zeros((*f1.image.shape, 3), dtype=np.uint8)
        else:
            self._track_image = (self._track_image * 0.8).astype(np.uint8)
        img = cv2.cvtColor(f1.image, cv2.COLOR_GRAY2RGB)

        for id, (x0, y0), (x1, y1) in zip(ids, uv0, uv1):
            self._track_image = cv2.line(self._track_image, (x1, y1), (x0, y0), self._col(id), 1)
            img = cv2.circle(img, (x1, y1), 5, self._col(id), 1)   # negative thickness => filled circle
            if id in id2proj:
                xp, yp = np.array(uvp[id2proj[id]]).astype(int)
                # img = cv2.circle(img, (xp, yp), 5, self._col(id), 1)
                img = cv2.rectangle(img, (xp-2, yp-2), (xp+2, yp+2), self._col(id), 1)
                img = cv2.line(img, (xp, yp), (x1, y1), self._col(id), 1)
        img = cv2.add(img, self._track_image)

        if self._track_save_path:
            cv2.imwrite(os.path.join(self._track_save_path, 'frame_%d.png' % self._frame_count), img)

        img_sc = 768/img.shape[0]
        cv2.imshow(label, cv2.resize(img, None, fx=img_sc, fy=img_sc, interpolation=cv2.INTER_CUBIC))
        cv2.waitKey(0 if pause else 25)

    def _draw_pts3d(self, new_frame, pause=True, label='3d points', shape=None, m_per_px=None, plt3d=True):
        if len(self.state.map3d) == 0:
            return

        m0, m1 = self._prev_map3d, self.state.map3d
        ids, pts1 = zip(*[(id, pt.pt3d.copy().flatten()) for id, pt in m1.items() if pt.inlier_count > 0]) if len(m1) > 0 else ([], [])
        pts0 = [(m0[id] if id in m0 else None) for id in ids]
        self._prev_map3d = dict(zip(ids, pts1))

        import matplotlib.pyplot as plt
        if self._map_fig is None or not plt.fignum_exists(self._map_fig[0].number):
            if plt3d:
                from mpl_toolkits.mplot3d import Axes3D
                self._map_fig = [None, None]
                self._map_fig[0] = plt.figure(1)
                self._map_fig[1] = self._map_fig[0].add_subplot(111, projection='3d')
            else:
                self._map_fig = plt.subplots()

        self._map_fig[1].clear()
        if plt3d:
            self._map_fig[1].set_xlabel('x')
            self._map_fig[1].set_ylabel('y')
            self._map_fig[1].set_zlabel('z')
        else:
            self._map_fig[1].set_aspect('equal')

        for id, pt0, pt1 in zip(ids, pts0, pts1):
            x1, y1, z1 = pt1
            if pt0 is not None:
                x0, y0, z0 = pt0
                if plt3d:
                    self._map_fig[1].plot((x0, x1), (y0, y1), (z0, z1), color=self._col(id, fl=True, bgr=False))
                else:
                    self._map_fig[1].plot((x0, x1), (y0, y1), color=self._col(id, fl=True, bgr=False))
            if plt3d:
                self._map_fig[1].plot((x1,), (y1,), (z1,), 'o', color=self._col(id, fl=True, bgr=False), mfc='none')
            else:
                self._map_fig[1].plot((x1,), (y1,), 'o', color=self._col(id, fl=True, bgr=False), mfc='none')

        # draw s/c position,
        lfp, nfp = self.state.last_frame.pose.post, new_frame.pose.post
        if lfp is not None and nfp is not None:
            x0, y0, z0 = tools.q_times_v(lfp.quat.conj(), -lfp.loc)
            x1, y1, z1 = tools.q_times_v(nfp.quat.conj(), -nfp.loc)
            if plt3d:
                self._map_fig[1].plot((x0, x1), (y0, y1), (z0, z1), color='r')
                self._map_fig[1].plot([x1], [y1], [z1], 's', color='r', mfc='none')
            else:
                self._map_fig[1].plot((x0, x1), (y0, y1), color='r')
                self._map_fig[1].plot(x1, y1, 'x', color='r')

        # add origin
        if plt3d:
            self._map_fig[1].plot([0], [0], [0], 'x', color='b')
        else:
            self._map_fig[1].plot(0, 0, 'x', color='b')
        if not pause:
            plt.pause(0.05)
        else:
            plt.show()

    def _cv_draw_pts3d(self, new_frame, pause=True, label='3d points', shape=(768, 768), m_per_px=1.3):
        if len(self.state.map3d) == 0:
            return

        m0, m1 = self._prev_map3d, self.state.map3d
        ids, pts1 = zip(*[(id, pt.pt3d.copy().flatten()) for id, pt in m1.items()]) if len(m1) > 0 else ([], [])
        pts0 = [(m0[id] if id in m0 else None) for id in ids]
        self._prev_map3d = dict(zip(ids, pts1))

        if self._map_image is None:
            self._map_image = np.zeros((*shape, 3), dtype=np.uint8)
        else:
            self._map_image = (self._map_image * 0.9).astype(np.uint8)
        img = np.zeros((*shape, 3), dtype=np.uint8)

        if self._init_map_center is None:
            self._init_map_center = 0
        map_center = self._init_map_center
        cx, cz = shape[1] // 2, shape[0] // 2

        def massage(pt):
            x, y, z = pt - map_center
            x = cx + x/m_per_px
            z = cz + z/m_per_px
            lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
            return np.clip(int(round(0 if np.isnan(x) else x)), lo, hi), np.clip(int(round(0 if np.isnan(z) else z)), lo, hi)

        for id, pt0, pt1 in zip(ids, pts0, pts1):
            try:
                x1, y1 = massage(pt1)
                if pt0 is not None:
                    x0, y0 = massage(pt0)
                    self._map_image = cv2.line(self._map_image, (x1, y1), (x0, y0), self._col(id), 1)
                img = cv2.circle(img, (x1, y1), 5, self._col(id), -1)
            except:
                return

        # draw s/c position,
        lfp, nfp = self.state.last_frame.pose.post, new_frame.pose.post
        if lfp is not None and nfp is not None:
            x0, y0 = massage(tools.q_times_v(lfp.quat.conj(), -lfp.loc))
            x1, y1 = massage(tools.q_times_v(nfp.quat.conj(), -nfp.loc))
            self._map_image = cv2.line(self._map_image, (x1, y1), (x0, y0), (0, 0, 255), 1)
            img = cv2.circle(img, (x1, y1), 8, (0, 0, 255), -1)

        # add origin as a blue cross
        img = cv2.line(img, (cx-3, cz-3), (cx+3, cz+3), (255, 0, 0), 1)
        img = cv2.line(img, (cx-3, cz+3), (cx+3, cz-3), (255, 0, 0), 1)

        img = cv2.add(img, self._map_image)
        cv2.imshow(label, img)
        cv2.waitKey(0 if pause else 25)

    def _draw_matches(self, img1, img2, kp1, kp2, mask, pause=True, label='matches'):
        idxs = np.array(list(range(len(kp1))))[mask]
        matches = [[cv2.DMatch(i, i, 0)] for i in idxs]
        draw_params = {
            #            matchColor: (88, 88, 88),
            'singlePointColor': (0, 0, 255),
            #            'flags': cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        }
        # scale images, show
        img_sc = 768/img1.shape[0]
        sc_img1 = cv2.cvtColor(cv2.resize(img1, None, fx=img_sc, fy=img_sc, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        sc_img2 = cv2.cvtColor(cv2.resize(img2, None, fx=img_sc, fy=img_sc, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
        img3 = cv2.drawMatchesKnn(sc_img1, self.arr2kp(kp1*img_sc), sc_img2, self.arr2kp(kp2*img_sc), matches, None, **draw_params)
        cv2.imshow(label, img3)
        cv2.waitKey(0 if pause else 25)


class LogWriter:
    def __init__(self, log_queue=None):
        self.log_queue = log_queue

    def write(self, msg):
        if msg.strip() != '':
            if self.log_queue is None:
                logging.info(msg)
            else:
                self.log_queue.put(('info', msg))

    def flush(self):
        if hasattr(self.log_queue, 'flush'):
            self.log_queue.flush()


def mp_bundle_adj(arg_queue, log_queue, res_queue):
    while True:
        cmd, ba_fun, args, kwargs = arg_queue.get()
        if cmd == 'ba':
            res_queue.put(ba_fun(*args, log_writer=LogWriter(log_queue), **kwargs))
        else:
            break
