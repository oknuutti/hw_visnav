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

    def __init__(self, *args, geodetic_origin=None, ori_off_q=None, **kwargs):
        super(VisualGPSNav, self).__init__(*args, **kwargs)
        self.geodetic_origin = geodetic_origin    # (lat, lon, height)
        self.ori_off_q = ori_off_q

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

    def _bundle_adjustment(self, keyframes=None, current_only=False, same_thread=False, skip_meas=True):
        super(VisualGPSNav, self)._bundle_adjustment(keyframes=keyframes, current_only=current_only,
                                                     same_thread=same_thread, skip_meas=False)
