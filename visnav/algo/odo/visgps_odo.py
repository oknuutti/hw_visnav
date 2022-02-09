import logging
import math

import numpy as np
import quaternion

from visnav.algo import tools
from visnav.algo.tools import Pose
from visnav.algo.odo.base import VisualOdometry, State
from visnav.algo.odo.vis_gps_bundleadj import vis_gps_bundle_adj

logger = logging.getLogger("odo").getChild("visgps")


class VisualGPSState(State):
    def __init__(self):
        super(VisualGPSState, self).__init__()
        self.ba_prior = (None,) * 4
        self.ba_prior_fids = np.array([])


class VisualGPSNav(VisualOdometry):
    DEF_LOC_ERR_SD = 10     # in meters
    DEF_ORI_ERR_SD = math.radians(10)
    DEF_BA_DIST_COEF = False
    DEF_BA_N_CAM_INTR = 0
    DEF_ENABLE_MARGINALIZATION = False

    def __init__(self, *args, geodetic_origin=None, ori_off_q=None, ba_err_logger=None, **kwargs):
        super(VisualGPSNav, self).__init__(*args, **kwargs)
        self.geodetic_origin = geodetic_origin    # (lat, lon, height)
        self.ori_off_q = ori_off_q
        self.ba_err_logger = ba_err_logger

    @staticmethod
    def get_new_state():
        return VisualGPSState()

    def initialize_frame(self, time, image, measure):
        nf = super(VisualGPSNav, self).initialize_frame(time, image, measure)
        if nf.measure:
            old_kf = [kf for kf in self.state.keyframes if kf.measure]
            nf.measure.time_adj = old_kf[-1].measure.time_adj if len(old_kf) > 0 else 0

            geodetic_origin = self.geodetic_origin
            lat, lon, alt = nf.measure.data[0:3]
            roll, pitch, yaw = nf.measure.data[3:6]
            b2c_roll, b2c_pitch, b2c_yaw = nf.measure.data[6:9]

            # world frame: +z up, +x is east, +y is north
            # body frame: +z down, +x is fw towards north, +y is right wing (east)
            # camera frame: +z into the image plane, -y is up, +x is right

            # wf2bf_q = tools.eul_to_q((np.pi, -np.pi / 2), 'xz')
            # bf2cf = Pose(None, tools.eul_to_q((np.pi / 2, np.pi / 2), 'yz'))

            if 1:
                w2b_bf_q = tools.ypr_to_q(yaw, pitch, roll)
                b2c_bf_q = tools.ypr_to_q(b2c_yaw, b2c_pitch, b2c_roll)
                yaw, pitch, roll = tools.q_to_ypr(w2b_bf_q * b2c_bf_q * (self.ori_off_q or quaternion.one))
                cf_w2c = tools.lla_ypr_to_loc_quat(geodetic_origin, [lat, lon, alt], [yaw, pitch, roll], b2c=self.bf2cf)
                if len(old_kf) > 0:
                    cf_w2c.vel = (cf_w2c.loc - (-old_kf[-1].pose.prior).loc) / (nf.time - old_kf[-1].time).total_seconds()
                else:
                    cf_w2c.vel = np.array([0, 0, 0])
                nf.pose.prior = -cf_w2c
            else:
                wf_body_r = tools.to_cartesian(lat, lon, alt, *geodetic_origin)
                bf_world_body_q = tools.ypr_to_q(yaw, pitch, roll)
                if 0:
                    cf_world_q = bf_cam_q.conj() * bf_world_body_q.conj()

                    cf_body_r = tools.q_times_v(bf_cam_q.conj(), -bf_cam_r)
                    cf_body_world_r = tools.q_times_v(cf_world_q * self.wf2bf_q.conj(), -wf_body_r)

                    nf.pose.prior = Pose(cf_body_r + cf_body_world_r, cf_world_q * bf_cam_q)
                else:
                    bf_world_q = bf_world_body_q.conj() * self.wf2bf_q.conj()
                    cf_world_q = bf_cam_q.conj() * bf_world_q

                    cf_body_r = tools.q_times_v(bf_cam_q.conj(), -bf_cam_r)
                    cf_body_world_r = tools.q_times_v(cf_world_q, -wf_body_r)
                    # bf_cam_q.conj() * bf_world_body_q.conj() * self.wf2bf_q.conj()

                    nf.pose.prior = Pose(cf_body_r + cf_body_world_r, bf_cam_q.conj() * bf_world_body_q.conj() * bf_cam_q)
                    # bf_cam_q.conj() * bf_world_body_q.conj() * bf_cam_q

                    # (NokiaSensor.w2b_q * NokiaSensor.b2c_q) * prior.quat.conj()
                    # self.wf2bf_q * bf_cam_q * (bf_cam_q.conj() * bf_world_body_q.conj() * bf_cam_q).conj()
                    # => self.wf2bf_q * bf_cam_q //*// bf_cam_q.conj() * bf_world_body_q * bf_cam_q //*//
                    # self.wf2bf_q * bf_cam_q //*// bf_cam_q.conj() * bf_world_body_q * bf_cam_q //*// bf_cam_q.conj() * bf_world_body_q.conj() * self.wf2bf_q.conj()

        return nf

    def initialize_track(self, new_frame):
        # require first frame to have a measure
        if new_frame.measure:
            super(VisualGPSNav, self).initialize_track(new_frame)

    def is_new_keyframe(self, new_frame):
        if not new_frame.pose.post:
            return False

        # Favour keyframes with gps measurements.
        # However, if really needed, allow a new keyframe even without a gps measurement.
        # If all_meas is true, include all gps measurements in keyframes
        all_meas = False
        require_meas = True     # to disable, would need some constraints on pose change between frames (e.g. imu)
                                #  - without such constraints, poses jump between gps and vo results

        if new_frame.measure:
            if all_meas:
                is_new_kf = True
                logger.debug('new kf: new gps measure')
            else:
                is_new_kf = None
        elif require_meas:
            is_new_kf = False
        elif not self.state.first_result_given:
            is_new_kf = False
        elif len(new_frame.kps_uv) / self.max_keypoints < self.new_kf_min_kp_ratio * 0.8:
            is_new_kf = True
            logger.debug('new kf: too few keypoints, no gps measure')
        elif len([pt for pt in self.state.map3d.values() if pt.inlier_count > 0 and pt.active]) < self.min_inliers * 2:
            is_new_kf = len(self.triangulation_kps(new_frame)) > 0
            if is_new_kf:
                logger.debug('new kf: active 3d points < 2 x min_inliers, no gps measure')
        else:
            is_new_kf = False

        return super(VisualGPSNav, self).is_new_keyframe(new_frame) if is_new_kf is None else is_new_kf

    def solve_2d2d(self, ref_frame, new_frame, use_prior=False):
        if 1 and (ref_frame.measure is None or new_frame.measure is None):
            return False
        elif use_prior:  # TODO: useless?
            assert False, 'not in use'
            p_delta = new_frame.pose.prior - ref_frame.pose.prior
            new_frame.pose.post = ref_frame.pose + p_delta
            return True

        ok = super(VisualGPSNav, self).solve_2d2d(ref_frame, new_frame)
        if not ok:
            return False

        p_delta = new_frame.pose.post - ref_frame.pose.post
        dist = np.linalg.norm(new_frame.pose.prior.loc - ref_frame.pose.prior.loc)
        p_delta.loc = dist * p_delta.loc / np.linalg.norm(p_delta.loc)
        new_frame.pose.post = ref_frame.pose.post + p_delta
        return True

    def _bundle_adjustment(self, keyframes=None, current_only=False, same_thread=False):
        logger.info('starting bundle adjustment')
        skip_meas = False

        with self._new_frame_lock:
            if keyframes is None:
                if current_only:
                    keyframes = self.state.keyframes[-1:]
                else:
                    keyframes = self.state.keyframes

        # possibly do camera calibration before bundle adjustment
        if not self.cam_calibrated and len(keyframes) >= self.ba_interval:  #  self.ba_interval, self.max_keyframes
            # experimental, doesnt work0
            self.calibrate_cam(keyframes)
            if 0:
                # disable for continuous calibration
                self.cam_calibrated = True

        # TODO:
        #     - keyframe pruning from the middle like in orb-slam
        #     - using a changing subset of keypoints like in HybVIO ()
        #     - supply initial points to LK-feature tracker based on predicted state and existing 3d points
        #     - inverse distance formulation, project covariance also (see vins)
        #     ?- speed and angular speed priors that limit them to reasonably low values
        #     - when get basic stuff working:
        #           - try ransac again with loose repr err param values
        #           - try time-diff optimization (feedback for frame-skipping at nokia.py)

        with self._new_frame_lock:
            dist_coefs = None
            if self.ba_dist_coef and not current_only and len(keyframes) >= self.max_keyframes:
                assert np.where(np.array(self.cam.dist_coefs) != 0)[0][-1] + 1 <= 2, 'too many distortion coefficients'
                dist_coefs = self.cam.dist_coefs[:2]

            keyframes, ids, poses_mx, pts3d, pts2d, v_pts2d, px_err_sd, cam_idxs, pt3d_idxs = \
                    self._get_visual_ba_args(keyframes, current_only, distorted=dist_coefs is not None)
            skip_pose_n = (1 if skip_meas else 0) if not current_only else 0

            if not skip_meas:
                meas_idxs = np.array([i for i, kf in enumerate(keyframes) if kf.measure is not None], dtype=int)
                meas_q = {i: keyframes[i].pose.prior.quat.conj() for i in meas_idxs}
                meas_r = np.array([tools.q_times_v(meas_q[i], -keyframes[i].pose.prior.loc) for i in meas_idxs])
                meas_aa = np.array([tools.q_to_angleaxis(meas_q[i], compact=True) for i in meas_idxs])
                t_off = np.array([keyframes[i].measure.time_off + keyframes[i].measure.time_adj for i in meas_idxs]).reshape((-1, 1))
                if 1:  # use velocity measure instead of pixel vel
                    v_pts2d = np.array([tools.q_times_v(meas_q[i], -keyframes[i].pose.prior.vel) for i in meas_idxs])
            else:
                meas_idxs, meas_r, meas_aa, t_off = [np.empty(0)] * 4

            if not current_only:
                rem_kf_ids = self.prune_keyframes()
                rem_kp_ids = self.prune_map3d(rem_kf_ids)
                logger.info('pruning %d keyframes and %d keypoints' % (len(rem_kf_ids), len(rem_kp_ids)))

        meas_idxs = meas_idxs.astype(int)
        meas_r = meas_r.reshape((-1, 3))
        meas_aa = meas_aa.reshape((-1, 3))

        args = (poses_mx, pts3d, pts2d, v_pts2d, cam_idxs, pt3d_idxs, self.cam.cam_mx, dist_coefs, px_err_sd, meas_r,
                meas_aa, t_off, meas_idxs, self.loc_err_sd, self.ori_err_sd)
        kwargs = dict(max_nfev=self.max_ba_fun_eval, skip_pose_n=skip_pose_n, huber_coef=(1, 5, 0.5),
                      poses_only=current_only, weighted_residuals=True)

        if self.ba_n_cam_intr and not current_only and len(keyframes) >= self.max_keyframes:
            kwargs['n_cam_intr'] = self.ba_n_cam_intr

        if self.enable_marginalization and not current_only:
            kf_ids = [kf.id for kf in keyframes]
            kwargs.update(dict(
                prior_k=self.state.ba_prior[0],
                prior_x=self.state.ba_prior[1],
                prior_r=self.state.ba_prior[2],
                prior_J=self.state.ba_prior[3],
                prior_cam_idxs=np.where(np.in1d(self.state.ba_prior_fids, kf_ids))[0],
                marginalize_pose_idxs=np.where(np.in1d(kf_ids, rem_kf_ids))[0],
                marginalize_pt3d_idxs=np.where(np.in1d(ids, rem_kp_ids))[0],
            ))

        poses_ba, pts3d_ba, dist_ba, cam_intr, t_off, new_prior, errs = \
            self._call_ba(vis_gps_bundle_adj, args, kwargs, parallize=not same_thread)

        if self.enable_marginalization and not current_only:
            self.state.ba_prior = new_prior
            self.state.ba_prior_fids = kf_ids

        if dist_ba is not None:
            print('\nDIST: %s\n' % dist_ba)
        if cam_intr is not None:
            print('\nCAM INTR: %s\n' % cam_intr)

        if 0 and not current_only:
            # show result
            from visnav.postproc import replay
            replay([kf.image for kf in keyframes], pts2d, self.cam, poses_ba, cam_idxs, pts3d_ba, pt3d_idxs,
                   frame_ids=[kf.id for kf in keyframes])

        with self._new_frame_lock:
            for i, dt in zip(meas_idxs, t_off):
                keyframes[i].measure.time_adj = dt - keyframes[i].measure.time_off

            self._update_poses(keyframes, ids, poses_ba, pts3d_ba, dist_ba, cam_intr, skip_pose_n=skip_pose_n, pop_ba_queue=not same_thread)

            if not current_only:
                self.del_keyframes(rem_kf_ids)
                self.del_keypoints(rem_kp_ids, kf_lim=None if self.enable_marginalization else self.min_retain_obs)

        if 0 and not current_only:
            # show result
            from visnav.run import replay_keyframes
            replay_keyframes(self.cam, self.all_keyframes(), self.all_pts3d())

        if self.ba_err_logger is not None and not current_only:
            self.ba_err_logger(keyframes[-1].id, errs)
