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
import logging
import math
import threading
import multiprocessing as mp
from functools import lru_cache

import numpy as np
import quaternion   # adds to numpy  # noqa # pylint: disable=unused-import
import cv2

from visnav.algo import tools
from visnav.algo.tools import Pose, DeltaPose
from visnav.algo.bundleadj import vis_bundle_adj
from visnav.algo.featdet import detect_gridded

logger = tools.get_logger("odo")


class PoseEstimate:
    def __init__(self, prior: Pose, post: Pose, method):
        self.prior = prior
        self.post = post
        self.method = method

    @property
    def any(self):
        return self.post or self.prior


class Measure:
    def __init__(self, data, time_off, time_adj=0):
        self.data = data
        self.time_off = time_off
        self.time_adj = time_adj


class Frame:
    _NEXT_ID = 1

    def __init__(self, time, image, img_sc, pose: PoseEstimate, measure: Measure=None, frame_num=None, orig_image=None,
                 kps_uv: dict=None, kps_uv_norm: dict=None, kps_uv_vel: dict=None, repr_err: dict=None, id=None):
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
        self.repr_err = repr_err or {}
        self.ini_kp_count = len(self.kps_uv)
        self.frame_num = frame_num
        self.orig_image = orig_image

    @property
    def repr_err_sd(self):
        return self._calc_repr_err_sd(tuple(tuple(err) for err in self.repr_err.values()))

    @staticmethod
    @lru_cache(maxsize=1)
    def _calc_repr_err_sd(errs):
        errs = np.array(errs)
        return None if len(errs) <= 1 else np.median(np.linalg.norm(errs, axis=1))

    def set_id(self):
        assert self._id is None, 'id already given, cant set it twice'
        self._id = Frame._NEXT_ID
        Frame._NEXT_ID += 1

    def to_rel(self, pt3d, post=True):
        if len(pt3d) == 0:
            return pt3d
        pose = getattr(self.pose, 'post' if post else 'prior')
        fun = tools.q_times_v if len(pt3d.shape) == 1 else tools.q_times_mx
        return fun(pose.quat, pt3d) + pose.loc.reshape((1, 3))

    def to_mx(self, post=True):
        pose = getattr(self.pose, 'post' if post else 'prior')
        return self._to_mx(tuple(pose.quat.components), tuple(pose.loc.flatten()))

    @staticmethod
    @lru_cache(maxsize=4)
    def _to_mx(q, v):
        return np.hstack((quaternion.as_rotation_matrix(np.quaternion(*q)), np.array(v).reshape((-1, 1)))).astype(np.float32)

    @property
    def id(self):
        return self._id

    def __hash__(self):
        return self.id


class Keypoint:
    _NEXT_ID = 1

    def __init__(self, id=None, pt3d=None):
        self._id = Keypoint._NEXT_ID if id is None else id
        Keypoint._NEXT_ID = max(self._id + 1, Keypoint._NEXT_ID)
        self.pt3d = pt3d
        self.pt3d_added_frame_id = None
        self.total_count = 0
        self.inlier_count = 0
        self.inlier_time = None
        self.active = True
        self.bad_qlt = False

    @property
    def id(self):
        return self._id

    def __hash__(self):
        return self._id


class State:
    def __init__(self):
        self.initialized = False
        self.keyframes = []
        self.map2d = {}         # id => Keypoint, all keypoints with only uv coordinates (still no 3d coords)
        self.feat_dscr = {}     # id => descriptor, populated if using ORB features for tracking
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
    POSE_2D2D, POSE_2D3D, POSE_2D3D_OPT = range(3)

    # keypoint detection and tracking
    DEF_ORB_FEATURE_TRACKING = False        # use orb features instead of optical flow
    DEF_VERIFY_FEATURE_TRACKS = False       # used regardless if orb features used
    DEF_MIN_KEYPOINT_DIST = 15
    DEF_NO_KP_BORDER = 10
    DEF_MAX_KEYPOINTS = 315
    DEF_DETECTION_GRID = (3, 3)
    DEF_MIN_TRACKING_QUALITY = 0.0001       # optical flow only
    DEF_REPR_REFINE_KP_UV = False           #
    DEF_REPR_REFINE_COEF = 0.2              # move tracked uv point by this ratio towards reprojected point
    DEF_REFINE_KP_UV = False
    DEF_MAX_KP_REFINE_DIST = 7              # max distance in px that will still refine kp (as opposed to discarding)
    DEF_ORB_SEARCH_DIST = 0.03              # orb feats only, in fov ratio

    DEF_INIT_BIAS_SDS = np.ones(6) * 5e-3   # bias drift sds, x, y, z, then so3
    DEF_INIT_LOC_SDS = np.ones(3) * 3e-2
    DEF_INIT_ROT_SDS = np.ones(3) * 3e-2
    DEF_LOC_SDS = np.ones(3) * 5e-3
    DEF_ROT_SDS = np.ones(3) * 1e-2

    # keyframe addition
    DEF_NEW_KF_MIN_KP_RATIO = 0.70                 # remaining inliers from previous keyframe features
    DEF_NEW_KF_MIN_DISPL_FOV_RATIO = 0.008         # displacement relative to the fov for triangulation
    DEF_NEW_KF_TRIANGULATION_TRIGGER_RATIO = 0.2   # ratio of 2d points tracked that can be triangulated
    DEF_INI_KF_TRIANGULATION_TRIGGER = 20          # need at least this qty of 2d points that can be tri. for first kf
    DEF_NEW_KF_MIN_KP_DISPL = 0.04                 # change in viewpoint relative to a 3d point
    DEF_NEW_KF_KP_DISPL_RATIO = 0.15               # ratio of 3d points with significant viewpoint change
    DEF_NEW_KF_ROT_ANGLE = math.radians(5)         # new keyframe if orientation changed by this much
    DEF_REPR_ERR_FOV_RATIO = 0.0005                # expected reprojection error (related to DEF_NEW_KF_MIN_DISPL_FOV_RATIO)
    DEF_MAX_REPR_ERR_FOV_RATIO = 0.003             # max tolerable reprojection error (related to DEF_NEW_KF_MIN_DISPL_FOV_RATIO)
    DEF_KF_BIAS_SDS = np.ones(6) * 5e-4            # bias drift sds, x, y, z, then so3

    # map maintenance
    DEF_MAX_KEYFRAMES = 8
    DEF_WINDOW_FIFO_LEN = 3
    DEF_MAX_MARG_RATIO = 0.90
    DEF_REMOVAL_USAGE_LIMIT = 2        # 3d keypoint valid for removal if it was available for use this many times
    DEF_REMOVAL_RATIO = 0.15           # 3d keypoint inlier participation ratio below which the keypoint is discarded
    DEF_MIN_RETAIN_OBS = 3             # keep inactive keypoints if still this many observations in keyframe window
    DEF_MM_BIAS_SDS = np.ones(6) * 2e-3  # bias drift sds, x, y, z, then so3

    DEF_INLIER_RATIO_2D2D = 0.50        # expected ratio of inliers when using the 5-point algo for pose
    DEF_MIN_2D2D_INLIERS = 20           # discard pose estimate if less inliers than this   (was 60, with 7px min dist feats)
    DEF_MIN_INLIERS = 15                # discard pose estimate if less inliers than this   (was 35, with 7px min dist feats)
    DEF_MIN_INLIER_RATIO = 0.01         # discard pose estimate if less inliers than this   (was 0.08 for asteroid)
    DEF_RESET_TIMEOUT = 6               # reinitialize if this many seconds without successful pose estimate
    DEF_MIN_FEATURE_INTENSITY = 10      # min level of intensity required near a keypoint
    DEF_MAX_FEATURE_INTENSITY = 250     # need lower than this intensity near a keypoint
    DEF_POSE_2D2D_QUALITY_LIM = 0.1     # minimum pose result quality
    DEF_CHECK_2D2D_RESULT = True        # fail 2d2d initialization if possibly only rotation between frames

    DEF_EST_2D2D_PROB = 0.99           # relates to max RANSAC iterations
    DEF_EST_2D2D_METHOD = cv2.RANSAC    # cv2.LMEDS is fast but inaccurate, cv2.RANSAC is the other choice

    DEF_OPT_INIT_RANSAC = False         # run pnp ransac before motion-only-ba to get better initial pose
    DEF_USE_3D2D_RANSAC = True
    DEF_EST_3D2D_ITER_COUNT = 1000      # max RANSAC iterations
    DEF_EST_3D2D_METHOD = cv2.SOLVEPNP_AP3P  # RANSAC kernel function  # SOLVEPNP_AP3P

    DEF_ONLINE_CAM_CALIB = False
    DEF_ROLLING_SHUTTER = False
    DEF_ROLLING_SHUTTER_AXIS = '+y'     # image axis affected by rolling shutter, +: older values along axis, -: against
    DEF_ROLLING_SHUTTER_DELAY = 0

    DEF_NEW_KEYFRAME_BA = False
    DEF_USE_BA = True
    DEF_THREADED_BA = False             # run ba in an own thread
    DEF_BA_INTERVAL = 4                 # run ba every this many keyframes
    DEF_MAX_BA_FUN_EVAL = 30            # max cost function evaluations during ba

    def __init__(self, cam, img_width=None, wf2bf: Pose = None, bf2cf: Pose = None,
                 verbose=0, pause=0, **kwargs):
        self.cam = cam
        self.img_width = img_width
        self.verbose = verbose
        if verbose > 0:
            logger.setLevel(logging.DEBUG)
        self.pause = pause
        self.wf2bf = wf2bf or Pose.identity
        self.bf2cf = bf2cf or Pose.identity

        # set params
        for attr in dir(self.__class__):
            if attr[:4] == 'DEF_':
                key = attr[4:].lower()
                setattr(self, key, kwargs.pop(key, getattr(self.__class__, attr)))
        assert len(kwargs) == 0, 'extra keyword arguments given: %s' % (kwargs,)

        self.lk_params = {
            'winSize': (16, 16),    # was (32, 32)
            'maxLevel': 4,          # was 4
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.2),  # was: ..., 10, 0.05)
            'minEigThreshold': self.min_tracking_quality,
        }
        self.kp_params = {
            'maxCorners': self.max_keypoints,
            'qualityLevel': 0.05,  # default around 0.05?
            'minDistance': self.min_keypoint_dist,
            'blockSize': 4,
        }
        self.orb_params = {
            'edgeThreshold': 11,         # default: 31
            'fastThreshold': 7,          # default: 20
            'firstLevel': 0,             # always 0
            'nlevels': 8,                # default: 8
            'patchSize': 31,             # default: 31
            'scaleFactor': 1.2,          # default: 1.2
            'scoreType': cv2.ORB_FAST_SCORE,  # default ORB_HARRIS_SCORE, other: ORB_FAST_SCORE
            'WTA_K': 2,                  # default: 2
        }
        self.akaze_params = {
            'descriptor_type': cv2.AKAZE_DESCRIPTOR_MLDB,  # default: cv2.AKAZE_DESCRIPTOR_MLDB
            'descriptor_channels': 3,  # default: 3
            'descriptor_size': 0,  # default: 0
            'diffusivity': cv2.KAZE_DIFF_CHARBONNIER,  # default: cv2.KAZE_DIFF_PM_G2
            'threshold': 0.00005,  # default: 0.001
            'nOctaves': 4,  # default: 4
            'nOctaveLayers': 4,  # default: 4
        }

        # state
        self.state = self.get_new_state()

        # removed frames and keypoints
        self.removed_keyframes = []
        self.removed_keypoints = []

        # has camera been calibrated
        self.cam_calibrated = not self.online_cam_calib     # experimental, doesnt work

        # current frame specific temp value cache
        self.cache = {}
        self._map_fig = None
        self._frame_count = 0
        self._track_save_path = None     # for debug purposes
        self._track_image_height = 1200 if verbose <= 2 else 400  # for debug purposes
        self._track_image = None    # for debug purposes
        self._track_colors = None   # for debug purposes
        self._map_image = None    # for debug purposes
        self._map_colors = None   # for debug purposes
        self._nadir_looking = False  # for debug purposes
        self._bottom_bar_img = None  # for debug purposes
        self._z_range = (-25, 300)  # for debug purposes, in meters
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

    @staticmethod
    def get_new_state():
        return State()

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

    def repr_err(self, frame, adaptive=True):
        norm_err = self.repr_err_fov_ratio * np.linalg.norm(np.array(frame.image.shape) / frame.img_sc)
        if adaptive and frame.repr_err_sd:
            norm_err = max(frame.repr_err_sd * 2, norm_err)
        return norm_err

    def max_repr_err(self, frame, adaptive=True):
        max_err = self.max_repr_err_fov_ratio * np.linalg.norm(np.array(frame.image.shape) / frame.img_sc)
        if adaptive and frame.repr_err_sd:
            max_err = max(frame.repr_err_sd * 3, max_err)
        return max_err

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

        # # remove 3d points that haven't contributed to poses lately
        # self.prune_map3d()

        # maybe do failure recovery
        if new_frame.pose.post is None:
            dt = (new_frame.time - self.state.last_success_time).total_seconds()
            if dt > self.reset_timeout:
                # if fail for too long, reinitialize (excl if only one keyframe)
                self.state.initialized = False
            elif self.state.first_result_given and not self.state.tracking_failed:
                # frame maybe corrupted, fail once and try again next time
                logger.info('Image maybe corrupted, failing one frame')
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
            logger.info('new keyframe (ID=%s) added' % new_frame.id)

            if not self.state.first_result_given and new_frame.pose.post and len(self.state.map2d) > 0 \
                    and len(self.triangulation_kps(new_frame)) > self.ini_kf_triangulation_trigger:
                # triangulate, solve pnp for all keyframes, ba
                self.initialize_first_keyframes()

            elif self.is_maintenance_time():
                # maybe do map maintenance    # TODO: (3) run in yet another thread
                self.maintain_map()
                bias_sds = self.mm_bias_sds
                self.state.first_mm_done = self.state.first_mm_done or self.state.keyframes[-1].id

            if self.verbose > 2:
                self._draw_pts3d(new_frame, pause=self.pause, plt3d=True)

            if self.verbose > 1:
                self._cv_draw_pts3d(new_frame, label=None, shape=(self._track_image_height,) * 2)
                if 1:
                    self._draw_bottom_bar()

        if new_frame.pose.method == VisualOdometry.POSE_2D2D:
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
            # new_frame.pose.post.loc_s2 = self.init_loc_sds
            # new_frame.pose.post.so3_s2 = self.init_rot_sds

        self.state.last_frame = new_frame
        return copy.deepcopy(new_frame), bias_sds

    def initialize_frame(self, time, image, measure):
        logger.info('new frame')
        orig_image = image

        # maybe reduce color depth
        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # maybe scale image
        img_sc = 1
        if self.img_width is not None:
            img_sc = self.img_width / image.shape[1]
            image = cv2.resize(image, None, fx=img_sc, fy=img_sc)

        lp = copy.deepcopy(self.state.last_frame.pose.any) if self.state.last_frame else Pose.identity
        nf = Frame(time, image, img_sc, PoseEstimate(prior=lp, post=None, method=None), measure=measure,
                   frame_num=self._frame_count, orig_image=orig_image)

        self._frame_count += 1
        return nf

    def flush_state(self):
        self.removed_keyframes.extend(self.state.keyframes)
        self.removed_keyframes.sort(key=lambda x: x.id)
        self.removed_keypoints.extend(self.state.map3d.values())

    def all_keyframes(self):
        return sorted(self.removed_keyframes + self.state.keyframes, key=lambda x: x.id)

    def all_pts3d(self):
        return self.removed_keypoints + list(self.state.map3d.values())

    def initialize_track(self, new_frame):
        if self.state is not None:
            self.flush_state()

        logger.info('initializing tracking')
        self.state = self.get_new_state()
        new_frame.pose.post = copy.deepcopy(new_frame.pose.prior)
        self.add_new_keyframe(new_frame)

        # check that init ok and enough features found
        if len(self.state.map2d) > self.min_inliers * 2:
            self.state.last_frame = new_frame
            self.state.last_success_time = new_frame.time
            self.state.initialized = True

    def initialize_first_keyframes(self, debug=False):
        self.triangulate(self.state.keyframes[-1])
        if len(self.state.map3d) < self.min_inliers * 2:
            return

        if debug:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, len(self.state.keyframes), sharex=True, sharey=True, figsize=(9, 5))
            for i in range(len(self.state.keyframes)):
                axs[0][i].imshow(self._overlay_pts(self.state.keyframes[i], size=7))

        keyframes = []
        for kf in self.state.keyframes:
            ok = True
            if not kf.pose.post or kf.pose.method == VisualOdometry.POSE_2D2D:
                ok = self.solve_pnp(kf, use_3d2d_ransac=True, min_inliers=12)
            # elif kf.pose.method == VisualOdometry.POSE_2D2D:
            #     kf.pose.method = VisualOdometry.POSE_2D3D
            if ok:
                keyframes.append(kf)

        self.state.keyframes = keyframes
        self._bundle_adjustment(same_thread=True)

        if debug:
            for i in range(len(self.state.keyframes)):
                axs[1][i].imshow(self._overlay_pts(self.state.keyframes[i], size=7))
            plt.tight_layout()
            plt.show()

    def check_features(self, new_frame, old_kp2d, old_kp2d_norm, new_kp2d, new_kp2d_norm, mask):
        mask = mask.flatten()
        if self.verify_feature_tracks or self.orb_feature_tracking:
            if 1:
                E, mask2 = cv2.findEssentialMat(old_kp2d_norm, new_kp2d_norm, self.cam.cam_mx,
                                                threshold=2 if self.refine_kp_uv or self.orb_feature_tracking else 1,
                                                method=self.est_2d2d_method, prob=self.est_2d2d_prob)
            else:
                F, mask2 = cv2.findFundamentalMat(old_kp2d_norm, new_kp2d_norm, cv2.FM_RANSAC, 1, #self.repr_err(new_frame),
                                                  self.est_2d2d_prob)
            if mask2 is None:
                mask[:] = False
            else:
                mask = np.logical_and(mask, mask2.flatten())

        # check that not too close to image border
        w, h = self.img_width, round(new_frame.image.shape[0] * self.img_width / new_frame.image.shape[1])
        mask = np.logical_and.reduce((mask,
                                      new_kp2d[:, 0, 0] >= self.no_kp_border,
                                      new_kp2d[:, 0, 1] >= self.no_kp_border,
                                      new_kp2d[:, 0, 0] <= w - self.no_kp_border,
                                      new_kp2d[:, 0, 1] <= h - self.no_kp_border))

        if not self.orb_feature_tracking and np.sum(mask) > 1:
            # check that not too close to each other
            idxs = np.where(mask)[0]
            D = tools.distance_mx(new_kp2d[idxs].squeeze())
            for i, k in enumerate(idxs):
                mask[k] = np.all(D[i, :i] > self.min_keypoint_dist * 0.5)

        return mask.astype(bool)

    def detect_features(self, new_frame):
        kp2d, dscr = self._detect_features(new_frame.image, new_frame.kps_uv.values())
        kp2d = np.array(kp2d)
        kp2d_norm = self.cam.undistort(kp2d / new_frame.img_sc)

        for i in range(len(kp2d)):
            kp = Keypoint()  # Keypoint({new_frame.id: pt})
            new_frame.kps_uv[kp.id] = kp2d[i, :]
            new_frame.kps_uv_norm[kp.id] = kp2d_norm[i]
            self.state.map2d[kp.id] = kp
            if dscr is not None:
                self.state.feat_dscr[kp.id] = dscr[i]
        new_frame.ini_kp_count = len(new_frame.kps_uv)

        logger.info('%d new keypoints detected' % len(kp2d))

    def _feature_detection_mask(self, image):
        mask = 255 * np.ones(image.shape, dtype=np.uint8)
        mask[:self.no_kp_border, :] = 0
        mask[-self.no_kp_border:, :] = 0
        mask[:, :self.no_kp_border] = 0
        mask[:, -self.no_kp_border:] = 0
        return mask

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
            logger.warning('no features detectable because of masking')
            return []

        # detect Shi-Tomasi keypoints
        if self.orb_feature_tracking:
            if 1:
                det = cv2.ORB_create(**self.orb_params)
            else:
                det = cv2.AKAZE_create(**self.akaze_params)
        else:
            det = cv2.GFTTDetector_create(**self.kp_params)

        # detect features in a grid so that features are spread out
        kp2d = detect_gridded(det, image, mask, *self.detection_grid, self.max_keypoints)
        kp2d, dscr = det.compute(image, kp2d) if self.orb_feature_tracking else (kp2d, None)
        return ([], None) if kp2d is None else (self.kp2arr(kp2d), dscr)

    def is_active_kp(self, id):
        return id in self.state.map2d or id in self.state.map3d and self.state.map3d[id].active

    def track_keypoints(self, new_frame):
        lf, nf = self.state.last_frame, new_frame

        if len(lf.kps_uv) == 0:
            return

        if self.orb_feature_tracking:
            ids, dt_arr, old_kp2d, old_kp2d_norm, new_kp2d, mask = self.track_orb_features(nf)
        else:
            # track keypoints using Lukas-Kanade method
            tmp = [(id, uv, lf.kps_uv_norm[id]) for id, uv in lf.kps_uv.items() if self.is_active_kp(id)]
            ids, old_kp2d, old_kp2d_norm = list(map(np.array, zip(*tmp) if len(tmp) > 0 else ([], [], [])))
            dt_arr = np.ones((len(ids),)) * (nf.time - lf.time).total_seconds()

            new_kp2d, mask, err = cv2.calcOpticalFlowPyrLK(lf.image, nf.image, old_kp2d, None, **self.lk_params)

            if self.refine_kp_uv and np.sum(mask) > 0:
                new_kp2d, mask = self.refine_gftt(new_frame, new_kp2d, mask)

        new_kp2d_norm = self.cam.undistort(np.array(new_kp2d) / nf.img_sc)
        kps_uv_vel = np.array([(uv1 - uv0) / dt for dt, uv0, uv1 in zip(dt_arr, old_kp2d_norm, new_kp2d_norm)])

        if self.rolling_shutter:
            new_kp2d_norm_hat = new_kp2d_norm.copy()

            def calc_delay(uv_norm):
                a = uv_norm[:, 0:1, 1:2] if self.rolling_shutter_axis[1] == 'y' else uv_norm[:, 0:1, 0:1]
                b = self.cam.height if self.rolling_shutter_axis[1] == 'y' else self.cam.width
                if self.rolling_shutter_axis[0] == '-':
                    a = (b - a)
                return a / b * self.rolling_shutter_delay

            if len(lf.kps_uv_vel) == 0:
                # old_kp2d_norm is uncompensated px coords, calc velocity based on uncompensated values,
                # then compensate px coords of old and new based on velocity
                new_kp2d_norm = new_kp2d_norm_hat + calc_delay(new_kp2d_norm_hat) * kps_uv_vel
                old_kp2d_norm = old_kp2d_norm + calc_delay(old_kp2d_norm) * kps_uv_vel
                lf.kps_uv_norm = dict(zip(ids[mask], old_kp2d_norm[mask]))
                lf.kps_uv_vel = dict(zip(ids[mask], kps_uv_vel[mask]))
            else:
                # old_kp2d_norm is in compensated px coords, solve for compensated new px coords:
                # x1 = xhat1 + alpha * v1, where v1 = (x1 - x0)/frame_dt, and alpha = rs_dt * row_i/row_max, then:
                # (1 - alpha/frame_dt) * x1 == xhat1 - alpha/frame_dt * x0 ==>
                # x1 = (xhat1 - alpha/frame_dt * x0) / (1 - alpha/frame_dt)
                alpha_dt = calc_delay(new_kp2d_norm_hat) / (nf.time - lf.time).total_seconds()
                new_kp2d_norm = (new_kp2d_norm_hat - alpha_dt * old_kp2d_norm) / (1 - alpha_dt)
                kps_uv_vel = np.array([(uv1 - uv0) / dt for dt, uv0, uv1 in zip(dt_arr, old_kp2d_norm, new_kp2d_norm)])

        # extra sanity check on tracked points, set mask to false if keypoint quality too poor
        mask = self.check_features(nf, old_kp2d, old_kp2d_norm, new_kp2d, new_kp2d_norm, mask)
        if 0:
            self._plot_tracks(nf.image, old_kp2d, new_kp2d, mask)

        if not self.orb_feature_tracking:
            # mark non-tracked 3d-keypoints belonging to at least two keyframes as non-active, otherwise delete
            with self._3d_map_lock:
                for id in ids[np.logical_not(mask)]:
                    if id in self.state.map3d:
                        # marked non-active, removed during next map maintenance
                        self.state.map3d[id].active = False
                    elif id in self.state.map2d:
                        self.state.feat_dscr.pop(id, False)
                        self.state.map2d.pop(id)

        ids = ids[mask]
        nf.kps_uv = dict(zip(ids, new_kp2d[mask]))
        nf.kps_uv_norm = dict(zip(ids, new_kp2d_norm[mask]))
        nf.kps_uv_vel = dict(zip(ids, kps_uv_vel[mask]))

        logger.info('Tracking: %d/%d' % (len(new_kp2d), len(old_kp2d)))

    def refine_gftt(self, new_frame, new_kp2d, mask):
        # detect all corners that pass the criteria
        kp_params = self.kp_params.copy()
        kp_params['maxCorners'] = self.max_keypoints * 10
        kp_params['minDistance'] = 1
        kp_params['qualityLevel'] *= 1.0
        det = cv2.GFTTDetector_create(**kp_params)
        if 0:
            refined_kp2d = det.detect(new_frame.image)
        else:
            refined_kp2d = detect_gridded(det, new_frame.image, None, *self.detection_grid, kp_params['maxCorners'])
        refined_kp2d = self.kp2arr(refined_kp2d)

        # refine previously tracked keypoint uv coords
        D = tools.distance_mx(new_kp2d.squeeze(), refined_kp2d.squeeze())
        idxs = np.argmin(D, axis=1)
        cross_idxs = np.argmin(D, axis=0)
        mask = np.logical_and(mask.squeeze(), D[np.arange(len(idxs)), idxs] < self.max_kp_refine_dist)

        # set new uv-coords
        old, new = [], []
        for i, idx in zip(np.where(mask)[0], idxs[mask]):
            if i == cross_idxs[idx]:
                # if two mapped to same, pick the closest one and discard the other(s)
                old.append(new_kp2d[i].flatten())
                new.append(refined_kp2d[idx].flatten())
                new_kp2d[i] = refined_kp2d[idx]
            else:
                mask[i] = 0

        if 0:
            import matplotlib.pyplot as plt
            plt.figure(10, figsize=(14, 10))
            plt.gcf().clear()
            plt.imshow(new_frame.image)
            # old, new = np.array(old), np.array(new)
            for o, n in zip(old, new):
                plt.plot([o[0], n[0]], [o[1], n[1]], 'b')
                plt.plot([n[0]], [n[1]], 'bo', mfc='none')
            plt.pause(0.05)

        # # add as new keypoints those that are farther than min_keypoint_dist from existing ones
        # add_I = D[cross_idxs, np.arange(len(cross_idxs))] >= self.min_keypoint_dist
        # kp2d, kp2d_norm = kp2d[add_I], kp2d_norm[add_I]
        return new_kp2d, mask.astype(np.uint8)

    def track_orb_features(self, new_frame, propagate=False):
        nf, lf = new_frame, self.state.last_frame

        # take all successfully tracked features from last frame
        if propagate:
            # propagate by estimated keypoint velocity
            tmp = [(id, (nf.time - lf.time).total_seconds(), uv, lf.kps_uv_vel.get(id, [[0, 0]]),
                    self.state.feat_dscr.get(id, None)) for id, uv in lf.kps_uv_norm.items()]
            ids, dt_arr, old_kp2d_norm, old_kp2d_vel, old_dscr = list(
                map(np.array, zip(*tmp) if len(tmp) > 0 else ([],) * 5))
            old_kp2d_norm = old_kp2d_norm + old_kp2d_vel * dt_arr[:, None, None]
            old_kp2d = (self.cam.distort(old_kp2d_norm.squeeze())[:, None, :] * lf.img_sc + 0.5).astype(int)
        else:
            # use previous keypoint location as the best guess
            tmp = [(id, (nf.time - lf.time).total_seconds(), uv, lf.kps_uv_norm[id], self.state.feat_dscr.get(id, None))
                   for id, uv in lf.kps_uv.items()]
            ids, dt_arr, old_kp2d, old_kp2d_norm, old_dscr = list(
                map(np.array, zip(*tmp) if len(tmp) > 0 else ([],) * 5))

        # # add triangulated features from last np keyframes
        # tmp = [(id, (nf.time - rf.time).total_seconds(), uv, uv_vel, self.state.feat_dscr.get(id, None))
        #        for id, uv in lf.kps_uv.items()]

        # ids, dt_arr, old_kp2d, old_dscr = list(map(np.array, zip(*tmp) if len(tmp) > 0 else ([], [])))

        if 1:
            det = cv2.ORB_create(**self.orb_params)
        else:
            det = cv2.AKAZE_create(**self.akaze_params)
        kp2d = detect_gridded(det, nf.image, None, *self.detection_grid, self.max_keypoints * 10)
        kp2d, dscr = det.compute(nf.image, kp2d)

        kp2d = self.kp2arr(kp2d)
        D = tools.distance_mx(old_kp2d.reshape((-1, 2)), kp2d.reshape((-1, 2)))
        search_mask = (D < self.orb_search_dist * np.linalg.norm(np.array(nf.image.shape))).astype(np.uint8)
        matcher = cv2.BFMatcher(det.defaultNorm(), False)  # doesn't support crosscheck if mask given

        if 0:
            # ratio test as per "Lowe's paper"
            matches = matcher.knnMatch(old_dscr, dscr, k=2, mask=search_mask)
            coef = 0.85
            matches = list(
                m[0]
                for m in matches
                if len(m) > 1 and m[0].distance < coef * m[1].distance
            )
        else:
            # own cross checking
            matches = matcher.match(old_dscr, dscr, mask=search_mask)
            q2t = {m.queryIdx: m.trainIdx for m in matches}
            matches = [m for m in matches if q2t[m.queryIdx] == m.trainIdx]

        new_kp2d = np.ones_like(old_kp2d) * -1
        new_kp2d[[m.queryIdx for m in matches], :, :] = kp2d[[m.trainIdx for m in matches], :, :]
        mask = (new_kp2d[:, 0, 0] >= 0).astype(np.uint8)

        if 0:
            self._plot_tracks(nf.image, old_kp2d, new_kp2d, mask)

        return ids, dt_arr, old_kp2d, old_kp2d_norm, new_kp2d, mask

    def _plot_tracks(self, image, old_kp2d, new_kp2d, mask, pause=True):
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for (x0, y0), (x1, y1) in zip(old_kp2d[mask.astype(bool), 0, :],
                                      new_kp2d[mask.astype(bool), 0, :]):
            img = cv2.line(img, (x1, y1), (x0, y0), (0, 255, 0), 1)
            img = cv2.circle(img, (x1, y1), 5, (0, 255, 0), 1)
        img_sc = 768 / img.shape[0]
        cv2.imshow('track debug', cv2.resize(img, None, fx=img_sc, fy=img_sc, interpolation=cv2.INTER_AREA))
        cv2.waitKey(0 if pause else 25)

    def solve_2d2d(self, ref_frame, new_frame):
        rf, nf = ref_frame, new_frame

        # include all tracked keypoints, i.e. also 3d points
        # TODO: (3) better to compare rf post to nf prior?
        tmp = [(id, pt2d, rf.kps_uv_norm[id])
               for id, pt2d in nf.kps_uv_norm.items()
               if id in rf.kps_uv_norm]
        ids, new_kp2d, old_kp2d = list(map(np.array, zip(*tmp) if len(tmp) > 0 else ([], [], [])))

        R, p_delta, qlt_check_failed = None, None, False
        if len(old_kp2d) >= self.min_2d2d_inliers / self.inlier_ratio_2d2d:
            # solve pose using ransac & 5-point algo
            E, mask2 = cv2.findEssentialMat(old_kp2d, new_kp2d, self.cam.cam_mx, method=self.est_2d2d_method,
                                            prob=self.est_2d2d_prob, threshold=self.repr_err(nf))
            logger.info('E-mat: %d/%d' % (np.sum(mask2), len(old_kp2d)))

            if np.sum(mask2) >= self.min_2d2d_inliers:
                _, R, ur, mask = cv2.recoverPose(E, old_kp2d, new_kp2d, self.cam.cam_mx, mask=mask2.copy())
                logger.info('E=>R: %d/%d' % (np.sum(mask), np.sum(mask2)))
                inliers = np.where(mask)[0]

                if len(inliers) > self.min_2d2d_inliers and len(inliers) / np.sum(mask2) >= self.min_inlier_ratio:
                    if self.check_2d2d_result:
                        e_mx_qlt = self.pose_result_quality(rf, nf, dq=quaternion.from_rotation_matrix(R),
                                                            inlier_ids=ids[inliers], plot=0)
                        if e_mx_qlt < self.pose_2d2d_quality_lim:
                            logger.info('Pose result quality too low: %.3f' % e_mx_qlt)
                            R = None
                else:
                    R = None

        if R is not None:

            # record keypoint stats
            with self._3d_map_lock:
                for id in ids:
                    self.get_kp(id).total_count += 1
                for i in inliers.flatten():
                    self.get_kp(ids[i]).inlier_count += 1
                    self.get_kp(ids[i]).inlier_time = nf.time

            dq = quaternion.from_rotation_matrix(R)
            p_delta = DeltaPose(ur.flatten(), dq)

        ok = p_delta is not None
        if ok:
            nf.pose.post = rf.pose.post + p_delta
            nf.pose.method = VisualOdometry.POSE_2D2D

        return ok

    def get_kp(self, id):
        return self.state.map2d.get(id, self.state.map3d.get(id, None))

    def solve_pnp(self, kf, use_3d2d_ransac, min_inliers=None, lf=None):
        min_inliers = min_inliers or self.min_inliers
        inliers = None

        tmp = [(id, pt2d, self.state.map3d[id].pt3d)
               for id, pt2d in kf.kps_uv_norm.items()
               if id in self.state.map3d and self.state.map3d[id].active]
        ids, pts2d, pts3d = list(map(np.array, zip(*tmp) if len(tmp) else ([], [], [])))
        logger.info('Tracked 3D-points: %d/%d' % (len(pts3d), len(kf.kps_uv)))

        ok, p_new = False, None
        if len(pts3d) >= min_inliers:
            old_err_sd = (lf and lf.repr_err_sd or 0)

            if self.opt_init_ransac or use_3d2d_ransac:
                ok, rv, r, inliers = cv2.solvePnPRansac(pts3d, pts2d, self.cam.cam_mx, None,
                                                        iterationsCount=self.est_3d2d_iter_count,
                                                        reprojectionError=self.repr_err(kf),
                                                        flags=self.est_3d2d_method)

                # calculate delta-q and delta-r
                # dq = tools.angleaxis_to_q(rv) * kf.pose.prior.quat.conj()
                p_new = Pose(r, tools.angleaxis_to_q(rv))
                p_delta = p_new - kf.pose.prior

                # solvePnPRansac apparently randomly gives 180deg wrong answer,
                #  - too high translation in correct direction, why? related to delayed application of ba result?
                if abs(tools.q_to_ypr(p_delta.quat)[0]) > math.pi * 0.9:
                    logger.warning('rotated pnp-ransac solution by 180deg around z-axis')
                    q_fix = tools.ypr_to_q(math.pi, 0, 0)
                    p_delta.quat = p_delta.quat * q_fix
                    r = tools.q_times_v(q_fix, r)
                    p_delta.loc = r.flatten() - tools.q_times_v(p_delta.quat, kf.pose.prior.loc)
                    p_new = kf.pose.prior + p_delta

                kf_repr_err = None

            if not use_3d2d_ransac:
                if self.opt_init_ransac and ok and len(inliers) > min_inliers and len(inliers)/len(pts3d) > self.min_inlier_ratio:
                    kf.pose.post = p_new
                else:
                    kf.pose.post = copy.deepcopy(lf and lf.pose.post or kf.pose.prior)

                if lf:
                    kf.repr_err = lf.repr_err.copy()

                self._bundle_adjustment([kf], current_only=True, same_thread=True)

                inliers, _, _, kf_repr_err = self.sanity_check_pt3d(ids, pts3d, [kf], repr_err_only=True)
                ok = True

        if ok and len(inliers) >= min_inliers and len(inliers)/len(pts3d) > self.min_inlier_ratio:
            if use_3d2d_ransac:
                kf.pose.post = p_new

            # if 0:
            #     import matplotlib.pyplot as plt
            #     a = np.array([nf.kps_uv[id] for id in ids]).squeeze()
            #     b = self.cam.calc_img_R(nf.to_rel(pts3d, post=False), distort=True, legacy=True)*nf.img_sc
            #     nf.pose.post = Pose(r, q)
            #     c = self.cam.calc_img_R(nf.to_rel(pts3d, post=True), distort=True, legacy=True)*nf.img_sc
            #     plt.figure(1, figsize=(8, 6))
            #     plt.imshow(nf.image)
            #     plt.plot(a[:, 0], a[:, 1], 'bx')
            #     plt.plot(b[:, 0], b[:, 1], 'rx')
            #     plt.plot(c[:, 0], c[:, 1], 'gx')
            #     plt.plot(c[inliers, 0], c[inliers, 1], 'go', mfc='none')
            #     plt.tight_layout()
            #     plt.xlim(-50, nf.image.shape[1] + 50)
            #     plt.ylim(nf.image.shape[0] + 50, -50)
            #     plt.show()

            # record keypoint stats
            for id in ids:
                self.state.map3d[id].total_count += 1
            for i in inliers.flatten():
                self.state.map3d[ids[i]].inlier_count += 1
                self.state.map3d[ids[i]].inlier_time = kf.time

            if kf_repr_err is None:
                _, _, _, kf_repr_err = self.sanity_check_pt3d(ids, pts3d, [kf], repr_err_only=True)

            for id, err in kf_repr_err[0].items():
                kf.repr_err[id] = err

            kf.pose.method = VisualOdometry.POSE_2D3D
            if self.repr_refine_kp_uv and kf.repr_err_sd > self.repr_err(kf, adaptive=False):
                tmp = [(id, kf.kps_uv_norm[id], err) for id, err in kf_repr_err[0].items()]
                ids, kps_uv_norm_, err = map(np.array, zip(*tmp))
                kps_uv_norm = kps_uv_norm_ - self.repr_refine_coef * err[:, None, :]
                kps_uv = self.cam.distort(kps_uv_norm.squeeze())[:, None, :] * kf.img_sc
                for i, id in enumerate(ids):
                    kf.kps_uv_norm[id] = kps_uv_norm[i]
                    kf.kps_uv[id] = kps_uv[i]
                    kf.repr_err[id] = (1 - self.repr_refine_coef) * err[i, :]

            logger.info('%s: %d/%d, px err sd: %.1f => %.1f' % (
                'PnP' if use_3d2d_ransac else 'OPT',
                0 if inliers is None else len(inliers),
                len(pts2d), old_err_sd, kf.repr_err_sd))

            return True

        else:
            if inliers is None:
                logger.info(' => Too few 3D points matched for reliable pose estimation')
            elif len(inliers) < min_inliers:
                logger.info(' => PnP was left with too few inliers')
            elif len(inliers)/len(pts3d) < self.min_inlier_ratio:
                logger.info(' => PnP too few inliers compared to total matches')
            else:
                logger.info(' => PnP Failed')

            return False

    def estimate_pose(self, new_frame):
        nf = new_frame

        ok = self.solve_pnp(nf, self.use_3d2d_ransac, lf=self.state.last_frame)

        if not ok and self.state.first_result_given and self.state.tracking_failed:  # give one chance with tracking_failed
            logger.info(' => clearing all 3d points')
            self.state.initialized = False

        if not ok and not self.state.first_result_given:
            rf = [kf for kf in self.state.keyframes if kf.pose.post][-1]
            ok = self.solve_2d2d(rf, nf)

        if ok:
            self.state.last_success_time = nf.time
            self._log_pose_diff('prior->post', nf.pose.prior.loc, nf.pose.prior.quat, nf.pose.post.loc, nf.pose.post.quat)
        else:
            nf.pose.post = None

        if self.verbose > 1:
            self._draw_tracks(nf, pause=self.pause)

    def _log_pose_diff(self, title, r0, q0, r1, q1):
        dq = q1 * q0.conj()
        dr = r1 - tools.q_times_v(dq, r0)
        logger.info(title + ' dq: ' + ' '.join(['%.3fdeg' % math.degrees(a) for a in tools.q_to_ypr(dq)])
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
        if not self.state.initialized:
            return False

        # check if
        #   a) should detect new feats as old ones don't work,
        #   b) can triangulate many 2d points,
        #   c) viewpoint changed for many 3d points, or
        #   d) orientation changed significantly
        #   e) prior orientation (=> phase angle => appearance) changed significantly

        rf, nf = self.state.keyframes[-1], new_frame

        # # if no kp triangulated yet
        # if not self.state.first_result_given:
        #     return len(self.state.map2d) > 0 \
        #        and len(self.triangulation_kps(nf)) > self.ini_kf_triangulation_trigger

        #   a) should detect new feats as old ones don't work
        if len(nf.kps_uv)/self.max_keypoints < self.new_kf_min_kp_ratio:
            logger.debug('new kf: too few keypoints')
            return True

        #   b) orientation changed significantly
        if tools.angle_between_q(nf.pose.any.quat, rf.pose.any.quat) > self.new_kf_rot_angle:
            logger.debug('new kf: orientation change')
            return True

        #   d) viewpoint changed for many 3d points
        if self.use_ba \
                and len(self.state.map3d) > 0 \
                and len(self.viewpoint_changed_kps(nf))/len(nf.kps_uv) > self.new_kf_kp_displ_ratio:
            logger.debug('new kf: viewpoint change')
            return True

        #   e) can triangulate many 2d points
        if len(self.state.map2d) > 0 and self.state.first_result_given:
            actives = len([pt for pt in self.state.map3d.values() if pt.inlier_count > 0 and pt.active])
            if self.new_kf_triangulation_trigger_ratio < 1.0 or actives < self.min_inliers * 2:
                k, n = len(self.triangulation_kps(nf)), len(self.state.map2d)
                if k > 0 and (k / n > self.new_kf_triangulation_trigger_ratio or actives < self.min_inliers * 2):
                    logger.debug('new kf: triangulation %.0f%% (%d/%d)' % (100 * k/n, k, n))
                    return True
                elif actives < self.min_inliers * 2:
                    logger.debug('loosing track soon: only %d active 3d points, cannot triangulate more' % (actives,))
        elif len(self.state.map2d) >= self.ini_kf_triangulation_trigger and not self.state.first_result_given:
            if len(self.triangulation_kps(nf)) >= self.ini_kf_triangulation_trigger:
                return True

        logger.debug('no new kf')
        return False

    def add_new_keyframe(self, new_frame):
        new_frame.set_id()
        self.state.keyframes.append(new_frame)
        if self.new_keyframe_ba and len(self.state.map3d) > 0:
            # poses = np.zeros((2, 7))
            # poses[0, :3] = tools.q_times_v(new_frame.pose.post.quat.conj(), -new_frame.pose.post.loc)
            # poses[0, 3:] = quaternion.as_float_array(new_frame.pose.post.quat.conj())
            self._bundle_adjustment(current_only=True, same_thread=True)
            # poses[1, :3] = tools.q_times_v(new_frame.pose.post.quat.conj(), -new_frame.pose.post.loc)
            # poses[1, 3:] = quaternion.as_float_array(new_frame.pose.post.quat.conj())
            # tools.plot_poses(poses, axis=(0, 0, 1), up=(0, -1, 0))
        self.detect_features(new_frame)
        if self.state.first_result_given:
            self.triangulate(new_frame)

    def triangulation_kps(self, new_frame):
        """
        return 2d keypoints that can be triangulated as they have more displacement than new_kf_min_displ_fov_ratio
        together with the corresponding reference frame
        """
        kpset = self.cache.get('triangulation_kps', None)
        if kpset is not None:
            return kpset

        # TODO: do as a configuration param, change also at sanity_check_pt3d/
        max_repr_err_mult = 4.0
        max_repr_err = self.repr_err(new_frame, adaptive=False) * max_repr_err_mult
        if (new_frame.repr_err_sd or 0) > max_repr_err:
            logger.debug('frame repr err too high for triangulation (%.2f > %.2f)' % (new_frame.repr_err_sd, max_repr_err))
            return {}

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
            if len(idxs) > 0 and f.pose.post and (f.repr_err_sd or 0) < self.repr_err(f, adaptive=False) * max_repr_err_mult:
                p_uvs = tools.q_times_img_coords(new_frame.pose.any.quat * f.pose.post.quat.conj(), r_uv_arr[idxs, j, :],
                                                 self.cam, opengl=False, distort=False)
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

        tmp = [(id, uv, ref_frame.kps_uv_norm[id])
               for id, uv in new_frame.kps_uv_norm.items()
               if id in ref_frame.kps_uv_norm]
        ids, n_uv_arr, r_uv_arr = list(map(np.array, zip(*tmp) if len(tmp) > 0 else ([], [], [])))
        p_uvs = tools.q_times_img_coords(new_frame.pose.any.quat * ref_frame.pose.post.quat.conj(),
                                         r_uv_arr.squeeze(), self.cam, opengl=False, distort=False)
        displ = np.linalg.norm(p_uvs - n_uv_arr.squeeze(), axis=1)
        fov_d = np.linalg.norm(np.array(new_frame.image.shape) / new_frame.img_sc)
        rel_displ = displ / fov_d
        kpset = set(ids[rel_displ > self.new_kf_min_kp_displ])
        logger.debug('avg kp displ: %.3f, over lim: %.1f%%' % (np.mean(rel_displ),
                                                               100*np.mean(rel_displ > self.new_kf_min_kp_displ)))

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
            kp4d = cv2.triangulatePoints(self.cam.cam_mx.dot(T0), self.cam.cam_mx.dot(T1),
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
            inl, _, rem, kf_repr_err = self.sanity_check_pt3d(ids, pts3d, [ref_frame, new_frame])
            inl_ids.extend(ids[inl])
            inl_pts3d.extend(pts3d[inl])
            removals.extend(ids[rem])

        with self._3d_map_lock:
            for kp_id in removals:
                self.del_keypoint(kp_id, bad_qlt=True)

            for kp_id, pt3d in zip(inl_ids, inl_pts3d):
                kp = self.state.map2d.pop(kp_id)
                kp.pt3d = pt3d
                kp.pt3d_added_frame_id = new_frame.id
                self.state.map3d[kp_id] = kp
                added_count += 1

        logger.info('%d/%d keypoints successfully triangulated' % (added_count, len(tr_kps)))

    def sanity_check_pt3d(self, ids, pts3d, frames, max_dist_coef=2, repr_err_only=False):
        if len(ids) == 0:
            return [], [], []

        ids = np.array(ids)
        mask = np.zeros((len(ids),), dtype=np.uint8)

        if not repr_err_only:
            #  check that point is in front of each frame its observed in
            for f in frames:
                in_frame = [id in f.kps_uv for id in ids]
                mask[in_frame] = (f.to_rel(pts3d[in_frame])[:, 2] < 0).astype(np.uint8)

        # check that reprojection error is not too large
        idxs = np.where(mask == 0)[0]
        kps4d = np.hstack((pts3d[idxs], np.ones((len(idxs), 1), dtype=np.float32)))

        kf_repr_err = []
        for j, f in enumerate(frames):
            norm_err, max_err = self.repr_err(f), self.max_repr_err(f)
            tmp = np.array([[i, f.kps_uv_norm[id]] for i, id in enumerate(ids[idxs]) if id in f.kps_uv_norm], dtype=object).T
            err = []
            if len(tmp) > 0:
                kp_idxs, uvs = tmp[0].astype(int), np.concatenate(tmp[1, :])
                proj_pts2d = self.cam.cam_mx.dot(f.to_mx()).dot(kps4d[kp_idxs, :].T).T
                uvp = proj_pts2d[:, :2]/proj_pts2d[:, 2:]
                err = uvs - uvp
                errn = np.linalg.norm(err, axis=1)
                err_sd = np.mean(np.std(err, axis=0))
                if 1:
                    norm_err = max(norm_err, err_sd * 2)
                    max_err = max(max_err, err_sd * 3)

                if len(frames) - j < 3:
                    # TODO: check if necessary to limit to three last frames, earlier got too many rejections after BA
                    mask[idxs[kp_idxs]] = np.bitwise_or.reduce((mask[idxs[kp_idxs]],
                                                                1*(errn > norm_err).astype(np.int),
                                                                2*(errn > max_err).astype(np.int)))
                    if 0 and (np.sum(errn > max_err) > 50 or len(frames) == 1 and np.sum(errn < norm_err) < 20):
                        import matplotlib.pyplot as plt
                        plt.figure(4)
                        plt.scatter(uvs[:, 0], uvs[:, 1])
                        plt.scatter(uvp[:, 0], uvp[:, 1])
                        plt.gca().invert_yaxis()
                        plt.show()
            kf_repr_err.append(dict(zip(ids[idxs], err)))

        if not repr_err_only:
            #  check that distance to the point is not too much, parallax should be high enough
            fov_d_rad = math.radians(np.linalg.norm(np.array([self.cam.x_fov, self.cam.y_fov])))
            idx = np.argmax(tools.distance_mx(np.array([f.pose.post.loc for f in frames])))
            i, j = np.unravel_index(idx, (len(frames),)*2)
            sep = tools.parallax(frames[i].pose.post.loc, frames[j].pose.post.loc, pts3d[mask == 0])
            idxs = np.where(mask == 0)[0]
            mask[idxs[sep / fov_d_rad * max_dist_coef < self.max_repr_err_fov_ratio]] = 1

        good = np.where(mask == 0)[0]
        wait = np.where(mask == 1)[0]
        remove = np.where(mask > 1)[0]
        return good, wait, remove, kf_repr_err

    def bundle_adjustment(self, max_keyframes=None, sync=False):
        self._ba_max_keyframes = max_keyframes
        if len(self._ba_started) <= 1:
            self._ba_started.append((self.state.keyframes[-1].id, time.time()))
        else:
            logger.warning('bundle adjustment in queue dropped')
            self._ba_started[-1] = (self.state.keyframes[-1].id, time.time())

        if self.threaded_ba:
            logger.info('giving go-ahead for bundle adjustment')
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

    def _get_visual_ba_args(self, keyframes=None, active_kp_only=False, distorted=False):
        if keyframes is None:
            keyframes = self.state.keyframes

        ids = set(pt_id for kf in keyframes for pt_id in kf.kps_uv.keys()
                  if pt_id in self.state.map3d and (not active_kp_only or self.state.map3d[pt_id].active))

        if len(ids) == 0:
            return

        tmp = [(id, self.state.map3d[id].pt3d) for id in ids]
        ids, pts3d = map(np.array, zip(*tmp))
        idmap = dict(zip(ids, np.arange(len(ids))))

        if 0:
            # flip to world -> cam
            poses_mx = np.array([
                np.hstack((
                    tools.q_to_angleaxis(f.pose.post.quat.conj(), compact=True),
                    tools.q_times_v(f.pose.post.quat.conj(), -f.pose.post.loc).flatten()
                ))
                for f in keyframes
            ])
        else:
            # already in cam -> world
            poses_mx = np.array([
                np.hstack((
                    tools.q_to_angleaxis(f.pose.post.quat, compact=True),
                    f.pose.post.loc.flatten()
                ))
                for f in keyframes
            ], dtype=np.float32)

        cam_idxs, pt3d_idxs, pts2d, v_pts2d, err_sd = list(map(np.array, zip(*[
            (i, idmap[id], uv.flatten()/(f.img_sc if distorted else 1),
                f.kps_uv_vel.get(id, np.array([0, 0])).flatten(),
                # np.linalg.norm(f.repr_err.get(id, 0)) or self.repr_err(f)
                # np.sqrt(np.linalg.norm(f.repr_err.get(id, 0))**2 + self.repr_err(f)**2)
                # self.repr_err(f)
                # np.sqrt(self.repr_err(f)**2 + self.repr_err(f, adaptive=False)**2)
                self.max_repr_err(f, adaptive=False)
            )
            for i, f in enumerate(keyframes)
                for id, uv in (f.kps_uv if distorted else f.kps_uv_norm).items() if id in idmap
        ])))

        return keyframes, ids, poses_mx, pts3d, pts2d, v_pts2d.astype(np.float32), err_sd.astype(np.float32), cam_idxs, pt3d_idxs

    def _call_ba(self, ba_fun, args, kwargs, parallize=True):
        if not self.threaded_ba or self._ba_arg_queue is None or not parallize:
            res = ba_fun(*args, **kwargs)
        else:
            self._ba_arg_queue.put(('ba', ba_fun, args, kwargs))
            res = self._ba_res_queue.get()
            try:
                while True:
                    level, msg = self._ba_log_queue.get(block=False)
                    getattr(logger, level)(msg)
            except:
                pass
        return res

    def _bundle_adjustment(self, keyframes=None, current_only=False, same_thread=False):
        logger.info('starting bundle adjustment')
        with self._3d_map_lock:
            if keyframes is None and current_only:
                keyframes = self.state.keyframes[-1:]

            keyframes, ids, poses_mx, pts3d, pts2d, v_pts2d, px_err_sd, cam_idxs, pt3d_idxs = \
                self._get_visual_ba_args(keyframes, current_only)

        skip_pose_n = 0 if current_only else 1

        args = (poses_mx, pts3d, pts2d, cam_idxs, pt3d_idxs, self.cam.cam_mx)
        kwargs = dict(max_nfev=self.max_ba_fun_eval, skip_pose_n=skip_pose_n, poses_only=current_only,
                      huber_coef=self.repr_err(keyframes[-1]))

        poses_ba, pts3d_ba = self._call_ba(vis_bundle_adj, args, kwargs, parallize=not same_thread)

        with self._new_frame_lock:
            self._update_poses(keyframes, ids, poses_ba, pts3d_ba, skip_pose_n=skip_pose_n,
                               pop_ba_queue=not same_thread)

        if not self.cam_calibrated and len(keyframes) >= self.max_keyframes:  #  self.ba_interval:
            # experimental, doesnt work
            self.calibrate_cam(keyframes)
            if 0:
                # disable for continuous calibration
                self.cam_calibrated = True

    def _update_poses(self, keyframes, ids, poses_ba, pts3d_ba, dist_ba=None, cam_intr=None, skip_pose_n=1, pop_ba_queue=True):
        if np.any(np.isnan(poses_ba)) or np.any(np.isnan(pts3d_ba)):
            logger.warning('bundle adjustment results in invalid 3d points')
            return

        for i, p in enumerate(poses_ba):
            f = keyframes[i + skip_pose_n]
            p_new = Pose(p[3:], tools.angleaxis_to_q(p[:3]))

            # flip from world -> cam into cam -> world
            if 0:
                p_new = -p_new

            if i == len(poses_ba) - 1:
                # last frame adjusted by this rotation and translation
                p_old = f.pose.post
                p_delta = p_new - p_old
            f.pose.post = p_new

        # sanity check for pts3d_ba:
        #  - active 3d keypoints are in front of at least two keyframes
        #  - reprojection error is acceptable
        #  - parallax is enough
        if len(pts3d_ba) > 0:
            good, borderline, rem_idxs, kf_repr_err = self.sanity_check_pt3d(ids, pts3d_ba, keyframes, max_dist_coef=3)
            inliers = np.concatenate((good, borderline))
            rem_ids, good_ids, good_pts3d = ids[rem_idxs], ids[inliers], pts3d_ba[inliers]
        else:
            rem_ids, good_ids, good_pts3d, kf_repr_err = [[]] * 4

        if len(kf_repr_err) > 0:
            for i, kf in enumerate(keyframes):
                for id, err in kf_repr_err[i].items():
                    kf.repr_err[id] = err

        logger.info('removing %d keypoints based on sanity check after ba' % len(rem_ids))
        for kp_id in rem_ids:
            if kp_id in self.state.map3d:
                self.del_keypoint(kp_id, bad_qlt=True)

        # transform 3d map so that consistent with first frame being unchanged
        for kp_id, pt3d in zip(good_ids, good_pts3d):
            if kp_id in self.state.map3d:
                self.state.map3d[kp_id].pt3d = pt3d

        # update camera model, uv_norms
        if dist_ba is not None:
            assert len(dist_ba) == 2, 'invalid length of distortion coefficient array'
            self.cam.dist_coefs[:2] = dist_ba

        if cam_intr is not None:
            if len(cam_intr) != 2:
                self.cam.cam_mx[0, 0] = self.cam.cam_mx[1, 1] = cam_intr[0]
            if len(cam_intr) > 1:
                self.cam.cam_mx[0, 2], self.cam.cam_mx[1, 2] = cam_intr[-2:]

        if dist_ba is not None or cam_intr is not None:
            # TODO: debug dist_ba and cam_intr assignment, currently tracking fails after
            #       assigning new params
            self.update_uv_norms(keyframes)

        # update keyframes that are newer than those included in the bundle adjustment run
        #   - also update recently triangulated new 3d points
        if keyframes[-1].id is None:
            new_keyframes, new_3d_kps = [], []
        else:
            new_keyframes = [kf for kf in self.state.keyframes if kf.id > keyframes[-1].id]
            new_3d_kps = [kp for kp in self.state.map3d.values() if kp.pt3d_added_frame_id > keyframes[-1].id]

        if len(new_keyframes) > 0:
            for f in new_keyframes:
                f.pose.post = f.pose.post + p_delta
            for kp in new_3d_kps:
                assert False, 'doesnt work!!'
                kp.pt3d = tools.q_times_v(dq, kp.pt3d) + tools.q_times_v(qn.conj(), dr)   # TODO: debug this!!

        if pop_ba_queue:
            st_kf_id, st_t = self._ba_started.pop(0)
            c = self.state.keyframes[-1].id - st_kf_id
            t = (time.time() - st_t)
            self._ba_started_kf_id = None
            logger.info('bundle adjustment complete after %d keyframes and %.2fs, queue len: %d'
                         % (c, t, len(self._ba_started)))

        if self.threaded_ba:
            try:
                self._ba_stop_lock.release()   # only used when running ba in synchronized mode
            except RuntimeError:
                pass  # can't happen if in synchronized mode

    def is_maintenance_time(self):
        return self.state.keyframes[-1].id % self.ba_interval == 0

    def maintain_map(self):
        if not self.use_ba:
            rem_kf_ids = []
            if len(self.state.keyframes) > self.max_keyframes:
                rem_kf_ids = self.prune_keyframes()

            # Remove 3d keypoints from map.
            # No need to remove 2d keypoints here as they are removed
            # elsewhere if 1) tracking fails, or 2) triangulated into 3d points.
            rem_kp_ids = self.prune_map3d(rem_kf_ids)

            self.del_keyframes(rem_kf_ids)
            self.del_keypoints(rem_kp_ids, kf_lim=self.min_retain_obs)
            logger.info('pruned %d keyframes and %d keypoints' % (len(rem_kf_ids), len(rem_kp_ids)))

        elif len(self.state.map3d) >= self.min_inliers:
            # pruning done before and after ba
            self.bundle_adjustment(max_keyframes=self.max_keyframes)

    def calibrate_cam(self, keyframes):
        # experimental, doesnt work
        dist = np.array(self.cam.dist_coefs) if self.cam.dist_coefs is not None else np.zeros((5, 1), dtype=float)
        _, _, _, pts3d, pts2d, _, _, cam_idxs, pt3d_idxs = self._get_visual_ba_args(keyframes, distorted=True)
        c_pts2d, c_pts3d = [], []
        for i in range(len(keyframes)):
            I = np.where(cam_idxs == i)[0]
            c_pts2d.append(pts2d[I])
            c_pts3d.append(pts3d[pt3d_idxs[I]])

        # fix principal point as otherwise leads to unstability
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
        if self.online_cam_calib > 5:
            flags |= cv2.CALIB_RATIONAL_MODEL
        if self.online_cam_calib < 5 and self.online_cam_calib != 3:
            flags |= cv2.CALIB_FIX_K3
        if self.online_cam_calib < 4:
            flags |= cv2.CALIB_FIX_TANGENT_DIST
        if self.online_cam_calib < 2:
            flags |= cv2.CALIB_FIX_K2

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(c_pts3d, c_pts2d, (self.cam.width, self.cam.height),
                                                           self.cam.cam_mx.copy(), dist, flags=flags)
        self.cam.cam_mx = mtx
        self.cam.dist_coefs = list(dist)
        print('\n\n\t\tdist: %s, mx: %s\n\n' % (dist, mtx[:2, :3]))

        # re-undistort keypoints
        with self._new_frame_lock:
            frames = {f.id: f for f in (keyframes + self.state.keyframes + [self.state.last_frame])}
            self.update_uv_norms(frames.values())

    def update_uv_norms(self, frames):
        for f in frames:
            tmp = [(id, uv) for id, uv in f.kps_uv.items()]
            ids, kp2d = list(map(np.array, zip(*tmp) if len(tmp) > 0 else ([], [])))
            kp2d_norm = self.cam.undistort(np.array(kp2d) / f.img_sc)
            f.kps_uv_norm = dict(zip(ids, kp2d_norm))

    def prune_keyframes(self):
        with self._3d_map_lock:
            if self.window_fifo_len >= self.max_keyframes:
                rem_kf_ids = [kf.id for kf in self.state.keyframes[:-self.max_keyframes]]
            else:
                # remove all keyframes with zero active keypoints (respect protected fifo part)
                active_kps = {id for id, kp in self.state.map3d.items() if kp.active}
                rem_kf_ids = [kf.id for kf in self.state.keyframes[:-self.window_fifo_len]
                              if len(active_kps.intersection(set(kf.kps_uv.keys()))) == 0]
                kf_ids = [kf.id for kf in self.state.keyframes if kf.id not in rem_kf_ids]

                # if still needed, select keyframes to remove based on a scheme where we try to follow exponential
                # keyframe intervals as measured by their ids
                for i in range(len(kf_ids) - self.max_keyframes):
                    id_diff = np.diff(kf_ids[: -self.window_fifo_len + 1])
                    range_n = len(kf_ids) - self.window_fifo_len
                    trg_diff = 10 ** (1 - np.arange(range_n) / range_n)
                    dev = trg_diff - id_diff
                    rem_idx = np.argmax(dev) + 1
                    rem_kf_ids.append(kf_ids[rem_idx])
                    kf_ids.pop(rem_idx)
        return rem_kf_ids

    def del_keyframes(self, ids):
        with self._3d_map_lock:
            rem_kfs = [kf for kf in self.state.keyframes if kf.id in ids]
            self.state.keyframes = [kf for kf in self.state.keyframes if kf.id not in ids]
            self.removed_keyframes.extend(rem_kfs)
            self.removed_keyframes.sort(key=lambda x: x.id)
            logger.info('%d keyframes dropped' % len(rem_kfs))

    def prune_map3d(self, rem_kf_ids=None):
        rem_kf_ids = rem_kf_ids or []
        with self._3d_map_lock:
            rem_ids = []
            for k_id, kp in self.state.map3d.items():
                if kp.total_count >= self.removal_usage_limit and kp.inlier_count / kp.total_count < self.removal_ratio:
                    # not a very good feature
                    rem_ids.append(k_id)
                else:
                    n_obs = len([1 for f in self.state.keyframes if f.id not in rem_kf_ids and k_id in f.kps_uv])
                    if kp.active and n_obs < 2 or not kp.active and n_obs < self.min_retain_obs:
                        # not enough observations
                        rem_ids.append(k_id)
        return rem_ids

    def del_keypoints(self, ids, kf_lim=None, bad_qlt=False):
        with self._3d_map_lock:
            removed = 0
            for id in ids:
                removed += self.del_keypoint(id, kf_lim=kf_lim, bad_qlt=bad_qlt)
            logger.info('%d 3d keypoints removed or marked as inactive' % removed)

    def del_keypoint(self, id, kf_lim=None, bad_qlt=False):
        if id in self.state.map3d and bad_qlt:
            self.state.map3d[id].bad_qlt = True

        if kf_lim is not None and id in self.state.map3d:
            if len([1 for f in self.state.keyframes if id in f.kps_uv]) >= kf_lim:
                ret = int(self.state.map3d[id].active)
                self.state.map3d[id].active = False
                return ret

        ret = int(id in self.state.map3d)
        if ret and not bad_qlt:
            self.removed_keypoints.append(self.state.map3d[id])
            # TODO: (*) save f.kps_uv, f.kps_uv_norm  (and uncomment below part where removal happens)

        self.state.feat_dscr.pop(id, False)
        self.state.map2d.pop(id, False)
        self.state.map3d.pop(id, False)
        if 0:
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

    def get_3d_map_pts(self, all=False):
        return np.array([pt.pt3d for pt in (list(self.state.map3d.values()) + (self.removed_keypoints if all else []))]).reshape((-1, 3))

    @staticmethod
    def get_2d_pts(frame):
        return np.array([uv.flatten() for uv in frame.kps_uv.values()]).reshape((-1, 2))

    def _col(self, id, fl=False, bgr=True):
        n = 1000
        if self._track_colors is None:
            self._track_colors = np.hstack((np.random.randint(0, 179, (n, 1)), np.random.randint(0, 255, (n, 1))))
        (h, s), v = self._track_colors[id % n], 255

        if id in self.state.map3d:
            wf2cf_q = self.wf2bf.quat * self.bf2cf.quat
            x, y, z = tools.q_times_v(wf2cf_q, self.state.map3d[id].pt3d)
            z_sc = self._z_range[1] - self._z_range[0]
            h = 179 - int(min(179, max(0, 179 * (z + self._z_range[0]) / z_sc + 179 // 2)))
            s = 255
        else:
            s = int(s * 0.5)

        v = min(255, 255 * ((self.state.map3d[id].inlier_count if id in self.state.map3d else 0) + 1) / 4)
        col = cv2.cvtColor(np.array([h, s, v], dtype=np.uint8).reshape((1, 1, 3)),
                           cv2.COLOR_HSV2BGR if bgr else cv2.COLOR_HSV2RGB).flatten()
        return (col/255 if fl else col).tolist()

    def _overlay_pts(self, kf, size='repr-err', return_pts=False, show=False, method='dist-pts'):
        if method in ('undist-img', 'ba'):
            image = cv2.resize(kf.image, None, fx=1/kf.img_sc, fy=1/kf.img_sc)
            image = cv2.undistort(image, self.cam.cam_mx, np.array(self.cam.dist_coefs))
            image = cv2.resize(image, None, fx=kf.img_sc, fy=kf.img_sc)
            tmp = [(id, ((uv * kf.img_sc).flatten() + 0.5).astype(int)) for id, uv in kf.kps_uv_norm.items()]
        else:
            image = kf.image
            tmp = [(id, (uv.flatten() + 0.5).astype(int)) for id, uv in kf.kps_uv.items()]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        ids, uvs = zip(*tmp) if len(tmp) > 0 else ([], [])

        id2proj, uvp = {}, None
        pts3d = np.array([self.state.map3d[id].pt3d for id in ids if id in self.state.map3d]).reshape((-1, 3))
        if kf.pose.post and len(pts3d) > 0:
            q_cf = kf.pose.post.quat.conj()
            v_cf = tools.q_times_v(q_cf, -kf.pose.post.loc)

            if method == 'undist-img':
                pts3d_cf = tools.q_times_mx(q_cf.conj(), pts3d - v_cf)
                uvph = self.cam.cam_mx.dot(pts3d_cf.T).T
                uvp = kf.img_sc * uvph[:, :2] / uvph[:, 2:]
            elif method == 'ba':
                from .vis_gps_bundleadj import project
                poses = np.array([[*tools.q_to_angleaxis(kf.pose.post.quat, compact=True), *kf.pose.post.loc]])
                uvp = project(pts3d.astype(np.float32), poses, self.cam.cam_mx)
                uvp *= kf.img_sc
            elif method == 'dist-pts':
                pts3d_cf = tools.q_times_mx(q_cf.conj(), pts3d - v_cf)
                uvp = kf.img_sc * self.cam.project(pts3d_cf.astype(np.float32))
            else:
                assert False, 'invalid method %s' % method

            id2proj = [id for id in ids if id in self.state.map3d]
            id2proj = dict(zip(id2proj, range(len(id2proj))))

        for id, (x, y) in zip(ids, uvs):
            if size == 'repr-err':
                c_diam = (math.ceil(np.linalg.norm(kf.repr_err.get(id, [0.1, 0.1]))*0.5) + 1) * 2 + 1
            else:
                c_diam = size
            image = cv2.circle(image, (x, y), c_diam, self._col(id), 1)   # negative thickness => filled circle
            if id in id2proj:
                xp, yp = (np.array(uvp[id2proj[id]]) + 0.5).astype(int)
                image = cv2.rectangle(image, (xp-2, yp-2), (xp+2, yp+2), self._col(id), 1)
                image = cv2.line(image, (xp, yp), (x, y), self._col(id), 1)

        if show:
            cv2.imshow('keypoint reprojection', image)
            cv2.waitKey()

        if return_pts:
            return image, ids, uvs, uvp
        return image

    def _draw_tracks(self, new_frame, pause=True, label='tracks'):
        f0, f1 = self.state.last_frame, new_frame
        track_img, ids, uv1, uvp = self._overlay_pts(f1, return_pts=True)

        tmp = [(id, uv) for id, uv in zip(ids, uv1) if id in f0.kps_uv]
        ids, uv1 = zip(*tmp) if len(tmp) > 0 else ([], [])
        uv0 = [(f0.kps_uv[id].flatten() + 0.5).astype(int) for id in ids]

        if self._track_image is None:
            self._track_image = np.zeros((*f1.image.shape, 3), dtype=np.uint8)
        else:
            self._track_image = (self._track_image * 0.8).astype(np.uint8)

        for id, (x0, y0), (x1, y1) in zip(ids, uv0, uv1):
            self._track_image = cv2.line(self._track_image, (x1, y1), (x0, y0), self._col(id), 1)
        track_img = cv2.add(track_img, self._track_image)

        if self._map_image is not None:
            s = self._track_image_height
            sc = s / track_img.shape[0]
            track_img = cv2.resize(track_img, None, fx=sc, fy=sc)
            img = np.zeros((s, track_img.shape[1] + s, 3), dtype=np.uint8)
            img[:s, :track_img.shape[1]] = track_img
            img[:s, track_img.shape[1]:track_img.shape[1] + s] = self._map_image
        else:
            img = track_img

        if self._bottom_bar_img is not None:
            sc = img.shape[1] / self._bottom_bar_img.shape[1]
            img = np.concatenate((img, cv2.resize(self._bottom_bar_img, None, fx=sc, fy=sc)), axis=0)

        if self._track_save_path:
            cv2.imwrite(os.path.join(self._track_save_path, 'frame_%d.png' % self._frame_count), img)

        img_sc = self._track_image_height/img.shape[0]
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
                self._map_fig[1].view_init(elev=90, azim=-90)
            else:
                self._map_fig = plt.subplots()

        self._map_fig[1].clear()
        if plt3d:
            self._map_fig[1].set_xlabel('x')
            self._map_fig[1].set_ylabel('y')
            self._map_fig[1].set_zlabel('z')
        else:
            self._map_fig[1].set_aspect('equal')

        wf2cf_q = self.wf2bf.quat * self.bf2cf.quat
        for id, pt0, pt1 in zip(ids, pts0, pts1):
            x1, y1, z1 = tools.q_times_v(wf2cf_q, pt1)
            if pt0 is not None:
                x0, y0, z0 = tools.q_times_v(wf2cf_q, pt0)
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
            x0, y0, z0 = (-lfp).to_global(self.bf2cf).to_global(self.wf2bf).loc
            x1, y1, z1 = (-nfp).to_global(self.bf2cf).to_global(self.wf2bf).loc
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

    def _cv_draw_pts3d(self, new_frame, pause=True, label='3d points', shape=(768, 768), min_side_m=50):
        self._map_image = np.zeros((*shape, 3), dtype=np.uint8)
        if len(self.state.map3d) == 0:
            return self._map_image

        # wf: +z out from image, +y is up (north), +x is right (east)
        wf2cf_q = self.wf2bf.quat * self.bf2cf.quat

        # map scaling
        pts3d = tools.q_times_mx(wf2cf_q, self.get_3d_map_pts(all=True))
        min_x, min_y = np.min(pts3d[:, :2], axis=0)
        max_x, max_y = np.max(pts3d[:, :2], axis=0)
        m_per_px = max(min_side_m, max_x - min_x, max_y - min_y) / shape[0]
        m_per_px = np.exp(np.ceil(np.log10(m_per_px)*20)/20)
        off_x = np.round((min_x + (max_x - min_x) / 2) / m_per_px / 30) * m_per_px * 30
        off_y = np.round((min_y + (max_y - min_y) / 2) / m_per_px / 30) * m_per_px * 30

        def colormap(z, base, vary='h'):
            hsv = cv2.cvtColor(np.array(base, dtype=np.uint8).reshape((1, 1, 3)), cv2.COLOR_BGR2HSV).flatten()
            maxv = 179 if vary == 'h' else 255
            z_sc = self._z_range[1] - self._z_range[0]
            hsv[{'h': 0, 's': 1, 'v': 2}[vary]] = maxv - int(min(maxv, max(0, maxv * (z + self._z_range[0]) / z_sc + maxv//2)))
            return tuple(map(int, cv2.cvtColor(np.array(hsv, dtype=np.uint8).reshape((1, 1, 3)), cv2.COLOR_HSV2BGR).flatten()))

        def loc2uvc(cf_loc, base=(0, 255, 0)):
            wf_loc = tools.q_times_v(wf2cf_q, cf_loc) - np.array([off_x, off_y, 0])
            x, y, z = np.array([1/m_per_px, -1/m_per_px, 1]) * wf_loc + np.array([shape[1] // 2, shape[0] // 2, 0])
            lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
            return np.clip(int(round(0 if np.isnan(x) else x)), lo, hi), \
                   np.clip(int(round(0 if np.isnan(y) else y)), lo, hi), \
                   colormap(z, base)

        def pose2uvc(pose, base):
            return loc2uvc((-pose).loc, base)

        for pt in list(self.state.map3d.values()) + self.removed_keypoints:
            u, v, c = loc2uvc(pt.pt3d, (0, 255 if pt.active else 128, 0))
            self._map_image[v:v+2, u:u+2, :] = c

        # draw trajectories
        measures = []
        traj_b = []
        for kf in self.all_keyframes() + ([] if new_frame is not self.state.keyframes[-1] else [new_frame]):
            if kf.pose.post is not None:
                ub1, vb1, _ = pose2uvc(kf.pose.post, (0, 0, 255))
                traj_b.append([ub1, vb1])
            if kf.measure is not None:
                measures.append(kf.pose.prior)

        traj_a = []
        for pose in measures:
            ua1, va1, _ = pose2uvc(pose, (255, 0, 0))
            traj_a.append([ua1, va1])

        self._map_image = cv2.polylines(self._map_image, [np.array(traj_a, np.int32).reshape((-1, 1, 2))], 0, (0, 0, 255), 1)
        self._map_image = cv2.polylines(self._map_image, [np.array(traj_b, np.int32).reshape((-1, 1, 2))], 0, (255, 128, 0), 1)

        # draw s/c position
        sc = np.array([[0, -1, 0], [-1, 1, 0], [0, 0, 0], [1, 1, 0], [0, -1, 0]]) * 10 * m_per_px

        wf_a = -measures[-1]
        sc_a = tools.q_times_mx(wf2cf_q, tools.q_times_mx(wf_a.quat, sc*2/3) + wf_a.loc) - np.array([off_x, off_y, 0])
        sc_a = np.array([1, -1]) * sc_a[:, :2] / m_per_px + np.array([shape[1] // 2, shape[0] // 2])

        wf_b = -new_frame.pose.post
        sc_b = tools.q_times_mx(wf2cf_q, tools.q_times_mx(wf_b.quat, sc) + wf_b.loc) - np.array([off_x, off_y, 0])
        sc_b = np.array([1, -1]) * sc_b[:, :2] / m_per_px + np.array([shape[1] // 2, shape[0] // 2])

        self._map_image = cv2.fillPoly(self._map_image, [np.array(sc_a, np.int32).reshape((-1, 1, 2))], (0, 0, 255))
        self._map_image = cv2.polylines(self._map_image, [np.array(sc_b, np.int32).reshape((-1, 1, 2))], 0, (255, 128, 0), 1)

        # add origin as a blue cross
        # self._map_image = cv2.line(self._map_image, (cx-3, cz-3), (cx+3, cz+3), (255, 0, 0), 1)
        # self._map_image = cv2.line(self._map_image, (cx-3, cz+3), (cx+3, cz-3), (255, 0, 0), 1)

        if label is not None:
            cv2.imshow(label, self._map_image)
            cv2.waitKey(0 if pause else 25)
        else:
            return self._map_image

    def _draw_bottom_bar(self, interactive=False):
        keyframes = self.all_keyframes()

        e_idxs = [i for i, kf in enumerate(keyframes) if kf.pose.post is not None]
        m_idxs = [i for i, kf in enumerate(keyframes) if kf.measure is not None]

        e_b = [(-keyframes[i].pose.post).to_global(self.bf2cf) for i in e_idxs]
        m_b = [(-keyframes[i].pose.prior).to_global(self.bf2cf) for i in m_idxs]

        e_w = [p.to_global(self.wf2bf) for p in e_b]
        m_w = [p.to_global(self.wf2bf) for p in m_b]

        e_w_loc = np.array([p.loc for p in e_w])
        m_w_loc = np.array([p.loc for p in m_w])

        if self._nadir_looking and False:
            # TODO: better way, now somehow works heuristically
            dq = tools.eul_to_q((-np.pi / 2,), 'y')
            e_b_ypr = np.array([tools.q_to_ypr(dq.conj() * p.quat) for p in e_b])[:, (2, 0, 1)] / np.pi * 180
            m_b_ypr = np.array([tools.q_to_ypr(dq.conj() * p.quat) for p in m_b])[:, (2, 0, 1)] / np.pi * 180
        else:
            e_b_ypr = np.array([tools.q_to_ypr(p.quat) for p in e_b]) / np.pi * 180
            m_b_ypr = np.array([tools.q_to_ypr(p.quat) for p in m_b]) / np.pi * 180

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(11, 2))
        axs = fig.subplots(1, 4, sharex=True)
        i = 0
        if 1:
            axs[i].set_ylabel('alt [m]')
            axs[i].plot(e_idxs, e_w_loc[:, 2])
            axs[i].plot(m_idxs, m_w_loc[:, 2])
            i += 1

        if 0:
            vel = np.zeros((len(keyframes),))
            for j, kf in enumerate(keyframes):
                vel[j] = np.median([v.flatten()[1] for v in kf.kps_uv_vel.values()]) if kf.kps_uv_vel else np.nan
            axs[i].set_ylabel('median y vel [px/s]')
            axs[i].plot(vel)
            i += 1

        if 1:
            axs[i].set_ylabel('yaw [deg]')
            axs[i].plot(e_idxs, e_b_ypr[:, 0])
            axs[i].plot(m_idxs, m_b_ypr[:, 0])
            i += 1

        if 1:
            axs[i].set_ylabel('pitch [deg]')
            axs[i].plot(e_idxs, e_b_ypr[:, 1])
            axs[i].plot(m_idxs, m_b_ypr[:, 1])
            i += 1

        if 1:
            axs[i].set_ylabel('roll [deg]')
            axs[i].plot(e_idxs, e_b_ypr[:, 2])
            axs[i].plot(m_idxs, m_b_ypr[:, 2])
            i += 1

        plt.tight_layout()

        if not interactive:
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            w, h = np.array(fig.canvas.devicePixelRatio()) * fig.canvas.get_width_height()
            img = np.flip(img.reshape((h, w, 3)), axis=2)
            self._bottom_bar_img = img
            plt.close(fig)
        else:
            plt.show()

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
                logger.info(msg)
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
