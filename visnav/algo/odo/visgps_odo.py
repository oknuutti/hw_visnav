import logging
import math

import numpy as np
import quaternion

from visnav.algo.odo.base import VisualOdometry, Pose
from visnav.algo.odo.vis_gps_bundleadj import vis_gps_bundle_adj
from visnav.algo import tools


class VisualGPSNav(VisualOdometry):
    DEF_LOC_ERR_SD = 10     # in meters
    DEF_ORI_ERR_SD = math.radians(10)

    def __init__(self, *args, geodetic_origin=None, **kwargs):
        super(VisualGPSNav, self).__init__(*args, **kwargs)
        self.geodetic_origin = geodetic_origin    # (lat, lon, height)

    def initialize_frame(self, time, image, measure):
        nf = super(VisualGPSNav, self).initialize_frame(time, image, measure)
        if nf.measure:
            ta = [kf.measure.time_adj for kf in self.state.keyframes if kf.measure]
            nf.measure.time_adj = ta[-1] if len(ta) > 0 else 0

            geodetic_origin = self.geodetic_origin
            lat, lon, alt = nf.measure.data[:3]
            roll, pitch, yaw = nf.measure.data[3:]

            # world frame: +z up, +x is east, +y is north
            # body frame: +z down, +x is fw towards north, +y is right wing (east)
            # camera frame: +z into the image plane, -y is up, +x is right

            if 1:
                # TODO: check function lla_ypr_to_loc_quat, there's something strange still there
                cf_w2c_r, cf_w2c_q = tools.lla_ypr_to_loc_quat(geodetic_origin, [lat, lon, alt], [yaw, pitch, roll],
                                                               b2c_q=self.bf_cam_pose.quat)
                cf_c2w_q = cf_w2c_q.conj()
                cf_c2w_r = tools.q_times_v(cf_c2w_q, -cf_w2c_r)
                nf.pose.prior = Pose(cf_c2w_r, cf_c2w_q)
            else:
                wf_body_r = tools.to_cartesian(lat, lon, alt, *geodetic_origin)
                bf_world_body_q = tools.ypr_to_q(yaw, pitch, roll)
                if 0:
                    cf_world_q = bf_cam_q.conj() * bf_world_body_q.conj()

                    cf_body_r = tools.q_times_v(bf_cam_q.conj(), -bf_cam_r)
                    cf_body_world_r = tools.q_times_v(cf_world_q * self.wf_body_q.conj(), -wf_body_r)

                    nf.pose.prior = Pose(cf_body_r + cf_body_world_r, cf_world_q * bf_cam_q)
                else:
                    bf_world_q = bf_world_body_q.conj() * self.wf_body_q.conj()
                    cf_world_q = bf_cam_q.conj() * bf_world_q

                    cf_body_r = tools.q_times_v(bf_cam_q.conj(), -bf_cam_r)
                    cf_body_world_r = tools.q_times_v(cf_world_q, -wf_body_r)
                    # bf_cam_q.conj() * bf_world_body_q.conj() * self.wf_body_q.conj()

                    nf.pose.prior = Pose(cf_body_r + cf_body_world_r, bf_cam_q.conj() * bf_world_body_q.conj() * bf_cam_q)
                    # bf_cam_q.conj() * bf_world_body_q.conj() * bf_cam_q

                    # (NokiaSensor.w2b_q * NokiaSensor.b2c_q) * prior.quat.conj()
                    # self.wf_body_q * bf_cam_q * (bf_cam_q.conj() * bf_world_body_q.conj() * bf_cam_q).conj()
                    # => self.wf_body_q * bf_cam_q //*// bf_cam_q.conj() * bf_world_body_q * bf_cam_q //*//
                    # self.wf_body_q * bf_cam_q //*// bf_cam_q.conj() * bf_world_body_q * bf_cam_q //*// bf_cam_q.conj() * bf_world_body_q.conj() * self.wf_body_q.conj()

        return nf

    def initialize_track(self, new_frame):
        # require first frame to have a measure
        if new_frame.measure:
            super(VisualGPSNav, self).initialize_track(new_frame)

    def is_new_keyframe(self, new_frame):
        if new_frame.measure and new_frame.pose.post:
            return True
        return super(VisualGPSNav, self).is_new_keyframe(new_frame)

    def solve_2d2d(self, ref_frame, new_frame, use_prior=False):
        if 1 and (ref_frame.measure is None or new_frame.measure is None):
            return False
        elif use_prior:  # TODO: useless?
            dq = new_frame.pose.prior.quat * ref_frame.pose.prior.quat.conj()
            dr = new_frame.pose.prior.loc - tools.q_times_v(dq, ref_frame.pose.post.loc)
            new_frame.pose.post = ref_frame.pose.post.new(dr, dq)
            return True

        ok = super(VisualGPSNav, self).solve_2d2d(ref_frame, new_frame)
        if not ok:
            return False

        dq = new_frame.pose.post.quat * ref_frame.pose.post.quat.conj()
        dr = new_frame.pose.post.loc.flatten() - tools.q_times_v(dq, ref_frame.pose.post.loc)
        dr = dr * np.linalg.norm(new_frame.pose.prior.loc - ref_frame.pose.prior.loc)/np.linalg.norm(dr)
        new_frame.pose.post = ref_frame.pose.post.new(dr, dq)
        return True

    def _bundle_adjustment(self, keyframes=None, current_only=False, same_thread=False):
        logging.info('starting bundle adjustment')
        skip_meas = False

        with self._new_frame_lock:
            if keyframes is None and current_only:
                keyframes = self.state.keyframes[-1:]

        # TODO: when drone rotates, estimate goes off the rails
        #       /=> maybe problem at BA as optimized 3d points move far away when rotating
        #          - also, points at the top of the image are estimated to be farther away than points at the bottom
        #          - jumpy ransac poses as points stupidly far
        #          - however, optimized poses barely move at all
        #       /=> why does optimization make a pose in the middle "lag", i.e. yaws: 146, 154, *152* (id=14), 163, 168 (id=16)
        #          => because corresponding yaw measure (145) drags it down
        #       /=> why points triangulated on a non-horizontal plane?
        #          => because movement of points gets explained by tilt rather than movement, why?
        #             => too slack reprojection error given for ransac
        #       /=> why reprojection error gets high during rotation, same as before with the "lag"?
        #           => disabling time-difference optimization helped
        #           => optical flow less accurate during rotation
        #       /=> don't converge always, why? => because starting pose from meas, which is too far from solution
        #       /=> meas and 3d-pt scales dont start to match (see trajectory plot), why?
        #           => doesnt even start from the same location
        #           => even global scale and location offset doesnt work, why?
        #       => Why drifts sideways when rotating along z-axis?
        #           => something wrong with rotation transformations ??? => all seem ok...
        #   - NEXT:
        #     - use gimbal orientation on new nokia datasets
        #     - camera pitch drift, does not happen on toy data
        #           /=> fixed with correct gimbal orientation? (no, would not result on increasing pitch error, only a bias)
        #           /=> or related to time sync? (probably not as then 3d points would get optimized to different alt.)
        #           => or camera calibration? Maybe, fixed by:
        #               => recalibration or online calibration based on low err 3d points)
        #           /=> or rolling shutter?      (probably not as happens with constant linear velocity)
        #     - rotation along cam axis still a problem, also happens with toy data
        #           => is because optical flow jitter during rotation
        #               - include 2d features in ba cost function?
        #               - use orb during rotations?
        #  - other ideas:
        #     - keyframe pruning from the middle like in orb-slam
        #     - inverse distance formulation, project covariance also (see vins)
        #     ?- speed and angular speed priors that limit them to reasonably low values
        #     ?- outlier rejection for features using ransac and F-matrix
        #     - when get basic stuff working:
        #           - try ransac again with loose repr err param values
        #           - try time-diff optimization (feedback for frame-skipping at nokia.py)

            keyframes, ids, poses_mx, pts3d, pts2d, v_pts2d, px_err_sd, cam_idxs, pt3d_idxs = \
                    self._get_visual_ba_args(keyframes, current_only)
            skip_pose_n = (1 if skip_meas else 0) if not current_only else 0

            if not skip_meas:
                meas_idxs = np.array([i for i, kf in enumerate(keyframes) if kf.measure is not None], dtype=int)
                meas_q = {i: keyframes[i].pose.prior.quat.conj() for i in meas_idxs}
                meas_r = np.array([tools.q_times_v(meas_q[i], -keyframes[i].pose.prior.loc) for i in meas_idxs], dtype=np.float32)
                meas_aa = np.array([tools.q_to_angleaxis(meas_q[i], compact=True) for i in meas_idxs], dtype=np.float32)
                t_off = np.array([keyframes[i].measure.time_off + keyframes[i].measure.time_adj for i in meas_idxs], dtype=np.float32)
            else:
                meas_idxs, meas_r, meas_aa, t_off = [np.empty(0)] * 4

        meas_idxs = meas_idxs.astype(int)
        meas_r = meas_r.reshape((-1, 3))
        meas_aa = meas_aa.reshape((-1, 3))

        args = (poses_mx, pts3d, pts2d, v_pts2d, cam_idxs, pt3d_idxs, self.cam_mx, px_err_sd, meas_r, meas_aa, t_off,
                meas_idxs, self.loc_err_sd, self.ori_err_sd)
        kwargs = dict(max_nfev=self.max_ba_fun_eval, skip_pose_n=skip_pose_n, huber_coef=(1, 5, 0.5), poses_only=current_only)

        poses_ba, pts3d_ba, t_off = self._call_ba(vis_gps_bundle_adj, args, kwargs, parallize=not same_thread)

        with self._new_frame_lock:
            for i, dt in zip(meas_idxs, t_off):
                keyframes[i].measure.time_adj = dt - keyframes[i].measure.time_off

            self._update_poses(keyframes, ids, poses_ba, pts3d_ba, skip_pose_n=skip_pose_n, pop_ba_queue=not same_thread)
