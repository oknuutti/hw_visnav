import logging
import math
import copy
from collections import Counter

import numpy as np
import quaternion

from visnav.algo import tools
from visnav.algo.tools import Pose
from visnav.algo.model import Camera
from visnav.algo.odo.base import Frame, PoseEstimate, Keypoint
from visnav.algo.odo.visgps_odo import VisualGPSNav


class SimpleOdometry(VisualGPSNav):
    def __init__(self, cam: Camera, max_repr_err: float, repr_err: float = 1.0, verbose: int = 1,
                 logger: logging.Logger = None):
        logger = logger or logging.getLogger("odo").getChild("simple")
        super(VisualGPSNav, self).__init__(cam, verbose=verbose, logger=logger,
                                           window_fifo_len=1000000, max_keyframes=1000000)
        self._max_repr_err = max_repr_err
        self._repr_err = repr_err
        self._rem_kp_ids = []
        self.check_2d2d_result = False

    def first_frame(self, kp_ids, kp_obs):
        assert len(kp_ids) == len(kp_obs), 'kp_ids and kp_obs need to have same length'
        nf = self._add_frame(True)
        self._add_kp_obs(nf, kp_ids, kp_obs, True)
        self.state.last_frame = nf

    def next_frame(self, kp_ids, kp_obs, is_keyframe):
        assert len(kp_ids) == len(kp_obs), 'kp_ids and kp_obs need to have same length'

        nf = self._add_frame(is_keyframe)
        self._add_kp_obs(nf, kp_ids, kp_obs, is_keyframe)

        rem_kp_ids = []

        # initial pose
        if is_keyframe and not self.state.initialized:
            rf = self.state.keyframes[0]
            ok = super(VisualGPSNav, self).solve_2d2d(rf, nf)
            if ok:
                self.initialize_first_keyframes()   # also optimizes all poses and 3d coords
                self.state.initialized = True
            else:
                self.state.last_frame = nf
                return [None] * 5
        elif not is_keyframe and not self.state.initialized:
            return [None] * 5
        elif self.state.initialized:
            # estimate current pose based on keypoints with 3d coords, don't optimize 3d coords
            ok = self.solve_pnp(nf, use_3d2d_ransac=True, min_inliers=12)
            if not ok:
                nf.pose.post = nf.pose.prior

            frames = self.state.keyframes + ([] if is_keyframe else [nf])
            self._bundle_adjustment(frames, current_only=True, same_thread=True)

            if is_keyframe:
                # triangulate new keypoints that have no initial 3d coords
                self.triangulate(nf)

                # optimize all poses and keypoint 3d coords
                self._bundle_adjustment(same_thread=True)

                # suggest to stop tracking the following features as reprojection error too large
                rem_kp_ids = np.unique([id for id, err in nf.repr_err.items() if np.linalg.norm(err) > self.max_repr_err()]
                                       + self._rem_kp_ids)
                self._rem_kp_ids = []
                self._rem_kp_obs(nf, rem_kp_ids)

        self.state.last_frame = nf

        # return poses, 3d points, discard kp ids
        if is_keyframe:
            pos, ori = zip(*[(kf.pose.post.loc, kf.pose.post.quat) for kf in self.state.keyframes])
            kp_ids, kp_3d = zip(*[(kp.id, kp.pt3d) for kp in self.state.map3d.values()]) \
                if len(self.state.map3d) else ([], [])
        else:
            pos, ori, kp_3d, rem_kp_ids = nf.pose.post.loc, nf.pose.post.quat, None, None

        return list(map(np.array, (pos, ori, kp_ids, kp_3d, rem_kp_ids)))

    def new_kps(self, kp_ids, kp_obs):
        assert len(kp_ids) == len(kp_obs), 'kp_ids and kp_obs need to have same length'
        cf = self.state.keyframes[-1]
        self._add_kp_obs(cf, kp_ids, kp_obs, is_keyframe=True)
        self.logger.info("New keypoints detected: %d" % (len(kp_ids),))

    def tracking_stats(self):
        # optimize all poses and keypoint 3d coords
        self._bundle_adjustment(same_thread=True)

        # remove keypoints with too large reprojection error from the last keyframe
        nf = self.state.keyframes[-1]
        rem_kp_ids = np.unique([id for id, err in nf.repr_err.items() if np.linalg.norm(err) > self.max_repr_err()]
                               + self._rem_kp_ids)
        self._rem_kp_obs(nf, rem_kp_ids)

        # reoptimize all poses and keypoint 3d coords if necessary
        if len(rem_kp_ids) > 0:
            self._bundle_adjustment(same_thread=True)

        # to reduce bias in error stats, we will exclude all features that are present in the last keyframe
        #  - Note that long track length features might end up under-represented,
        #    however, if we don't exclude, short track length features might end up over-represented
        excl_ids = set(nf.kps_uv.keys())

        # extract poses and structure
        pos, ori = zip(*[(kf.pose.post.loc, kf.pose.post.quat) for kf in self.state.keyframes])
        kps = [kp for kp in self.state.map3d.values() if kp.id not in excl_ids]
        kp_ids, kp_3d = zip(*[(kp.id, kp.pt3d) for kp in kps]) if len(kps) > 0 else ([], [])

        # for how many features it was possible to estimate 3d coordinates
        n2d = len([id for id in self.state.map2d.keys() if id not in excl_ids])
        succ_rate = len(kp_ids) / (n2d + len(kp_ids))

        # average tracking length in keyframes by counting all observations
        track_len = np.array(list(Counter([id for kf in self.state.keyframes
                                                  for id in kf.kps_uv.keys()
                                                      if id not in excl_ids]).values()))

        # extract reprojection errors
        repr_err = np.array([err for kf in self.state.keyframes
                                    for id, err in kf.repr_err.items()
                                        if id not in excl_ids]).reshape((-1, 2))
        repr_err = np.linalg.norm(repr_err, axis=1)

        return pos, ori, kp_ids, kp_3d, succ_rate, track_len, repr_err

    def repr_err(self, frame=None, adaptive=None):
        return self._repr_err

    def max_repr_err(self, frame=None, adaptive=None):
        return self._max_repr_err

    def del_keypoint(self, id, kf_lim=None, bad_qlt=False):
        self._rem_kp_ids.append(id)
        return 1

    def prune_map3d(self, rem_kf_ids=None):
        return []

    def _add_frame(self, is_keyframe):
        self.cache.clear()
        lp = copy.deepcopy(self.state.last_frame.pose.any) if self.state.last_frame else Pose.identity
        nf = Frame(None, None, None, PoseEstimate(prior=lp, post=None if self._frame_count > 0 else lp, method=None),
                   frame_num=self._frame_count)
        if is_keyframe:
            nf.set_id()
            self._frame_count += 1
            self.state.keyframes.append(nf)
        return nf

    def _add_kp_obs(self, nf, kp_ids, kp_obs, is_keyframe):
        kp2d_norm = self.cam.undistort(kp_obs)
        for i in range(len(kp_obs)):
            id = kp_ids[i]
            nf.kps_uv[id] = kp_obs[i, :]
            nf.kps_uv_norm[id] = kp2d_norm[i]
            if is_keyframe and id not in self.state.map2d and id not in self.state.map3d:
                self.state.map2d[id] = Keypoint(id=id)
        nf.ini_kp_count = len(nf.kps_uv)

    @staticmethod
    def _rem_kp_obs(cf, kp_ids):
        if len(kp_ids) > 0:
            cf.kps_uv = {id: kp for id, kp in cf.kps_uv.items() if id not in kp_ids}
            cf.kps_uv_norm = {id: kp for id, kp in cf.kps_uv_norm.items() if id not in kp_ids}
