import copy
import logging
import math
from datetime import datetime

import numpy as np
import quaternion
import cv2

from TelemetryParsing import readTelemetryCsv

from visnav.algo import tools
from visnav.algo.tools import Pose
from visnav.algo.model import Camera
from visnav.algo.odo.base import Measure
from visnav.algo.odo.visgps_odo import VisualGPSNav
from visnav.algo.odometry import VisualOdometry

from visnav.missions.base import Mission


class ToyNokiaSensor(Mission):
    # world frame: +z up, +x is east, +y is north
    # body frame: +z down, +x is fw towards north, +y is right wing (east)
    # camera frame: +z into the image plane, -y is up (north), +x is right (east)
    w2b = Pose(None, tools.eul_to_q((np.pi, -np.pi / 2), 'xz'))
    b2c = Pose(None, tools.eul_to_q((np.pi / 2,), 'z'))

    def __init__(self, *args, **kwargs):
        super(ToyNokiaSensor, self).__init__(*args, **kwargs)
        self.real_cam = copy.deepcopy(self.init_cam())
        if 0:
            self.real_cam.dist_coefs = [-0.11250615, 0.14296794, 0, 0, -0.11678778]
        elif 1:
            self.real_cam.dist_coefs = [-0.11250615, 0.14296794, -0.00175085, 0.00057391, -0.11678778]
        else:
            self.real_cam.dist_coefs = [0, 0, 0, 0, 0]

    def init_data(self):
        t_time, *t_data = readTelemetryCsv(self.data_path, None, None)
        coord0 = t_data[0][0], t_data[1][0], t_data[2][0]
        time0 = t_time[0].astype(np.float64) / 1e6
        t_time = (t_time - t_time[0]).astype(np.float64) / 1e6
        t_data = np.array(list(zip(*t_data)), dtype=np.float32)

        if 1:
            # filter so that images would be less jumpy for better feature tracking (problem with yaw [-pi, pi] though)
            t_data[:, :3] = cv2.filter2D(t_data[:, :3], cv2.CV_32F, np.ones((3, 1))/3)

        meas_interval = 10
        frame_rate = 5

        def data_gen():
            first = True
            f_id = 0
            prev_t_t = None
            prev_meas = None
            poses = []

            for t_id, (t_t, meas_data) in enumerate(zip(t_time, t_data)):
                f_t = prev_t_t
                if t_id == 0 or self.first_frame is not None and t_t < self.first_frame/frame_rate + self.video_toff:
                    f_id += meas_interval
                    prev_t_t = t_t
                    prev_meas = meas_data
                    continue
                elif self.last_frame is not None and t_t > self.last_frame/frame_rate + self.video_toff:
                    break

                # read a measurement
                full_rot = False
                dt = t_t - prev_t_t
                lat, lon, alt, roll, pitch, yaw, gimbal_roll, gimbal_pitch, gimbal_yaw = prev_meas
                if not full_rot:
                    roll, pitch = 0, 0
                meas = Measure(data=np.array([lat, lon, alt, roll, pitch, yaw]), time_off=prev_t_t - f_t)

                for i in range(meas_interval):
                    pose = self.get_pose(coord0, prev_meas, meas_data, (f_t - prev_t_t) / dt, full_rot=full_rot)
                    poses.append(pose)
                    img = self.generate_image(pose)
                    f_ts = datetime.fromtimestamp(f_t + time0).strftime('%H:%M:%S.%f')
                    t_ts = datetime.fromtimestamp(prev_t_t + time0).strftime('%H:%M:%S.%f')
                    yield img, f_t, 'frame-%d (%s)' % (f_id, f_ts), meas, (
                                'meas-%d (%s)' % (t_id, t_ts)) if meas else None, None
                    meas = None
                    first = False
                    f_t += dt / meas_interval
                    f_id += 1
                prev_t_t = t_t
                prev_meas = meas_data
            # import matplotlib.pyplot as plt
            # plt.plot(np.array([p.loc for p in poses]))
            # plt.show()

        return data_gen(), time0, coord0

    def get_pose(self, coord0, mes0, mes1, weight, full_rot):
        ypr0 = np.flip(mes0[3:6]) if full_rot else [mes0[5], 0, 0]
        ypr1 = np.flip(mes1[3:6]) if full_rot else [mes1[5], 0, 0]

        cf_w2c0 = tools.lla_ypr_to_loc_quat(coord0, mes0[:3], ypr0, b2c=self.b2c)
        cf_w2c1 = tools.lla_ypr_to_loc_quat(coord0, mes1[:3], ypr1, b2c=self.b2c)
        delta = cf_w2c1 - cf_w2c0
        aa = tools.q_to_angleaxis(delta.quat)
        aa[0] = tools.wrap_rads(aa[0]) * weight

        cf_w2c = Pose(cf_w2c0.loc * (1 - weight) + cf_w2c1.loc * weight,
                      tools.angleaxis_to_q(aa) * cf_w2c0.quat)
        cf_c2w = -cf_w2c

        return cf_c2w

    _pts4d = None
    def generate_image(self, pose, grid=5):
        if self._pts4d is None:
            x, y, z = pose.loc
            if 0:
                xx, yy = np.meshgrid(np.linspace(x - z * 3, x + z * 3, math.ceil(z*3/grid)*2 + 1),
                                     np.linspace(y - z * 3, y + z * 3, math.ceil(z*3/grid)*2 + 1))
            else:
                xx = np.random.uniform(x - z * 3, x + z * 3, 2 * (math.ceil(z * 3 / grid) * 2 + 1,))
                yy = np.random.uniform(x - z * 3, x + z * 3, 2 * (math.ceil(z * 3 / grid) * 2 + 1,))

            if 0:
                zz = np.zeros_like(xx)
            else:
                zz = np.random.uniform(0, -40, xx.shape)

            pts3d = np.stack((xx.flatten(), yy.flatten(), zz.flatten()), axis=1).reshape((-1, 3))
            self._pts4d = np.hstack((pts3d, np.ones((len(pts3d), 1))))

        T = np.hstack((quaternion.as_rotation_matrix(pose.quat), pose.loc.reshape((-1, 1))))
        proj_pts2d = self.real_cam.cam_mx.dot(T).dot(self._pts4d.T).T
        uvp = proj_pts2d[:, :2] / proj_pts2d[:, 2:]
        I = np.logical_and.reduce((
            uvp[:, 0] >= 0,
            uvp[:, 1] >= 0,
            uvp[:, 0] < self.real_cam.width,
            uvp[:, 1] < self.real_cam.height,
        ))
        uvp = self.real_cam.distort(uvp[I, :])
        uvp = (uvp + 0.5).astype(int)

        img = np.ones((self.real_cam.height, self.real_cam.width), dtype=np.uint8)*64
        for i in range(5):
            for j in range(5):
                img[np.clip(uvp[:, 1]+i-2, 0, self.real_cam.height-1),
                    np.clip(uvp[:, 0]+j-2, 0, self.real_cam.width-1)] = 224
        return img

    def init_cam(self):
        w, h, p = 1920, 1080, 0.00375
        common_kwargs_worst = {
            'sensor_size': (w * p, h * p),
            'quantum_eff': 0.30,
            'px_saturation_e': 2200,  # snr_max = 20*log10(sqrt(sat_e)) dB
            'lambda_min': 350e-9, 'lambda_eff': 580e-9, 'lambda_max': 800e-9,
            'dark_noise_mu': 40, 'dark_noise_sd': 6.32, 'readout_noise_sd': 15,
            # dark_noise_sd should be sqrt(dark_noise_mu)
            'emp_coef': 1,  # dynamic range = 20*log10(sat_e/readout_noise))
            'exclusion_angle_x': 55,
            'exclusion_angle_y': 90,
        }
        common_kwargs_best = dict(common_kwargs_worst)
        common_kwargs_best.update({
            'quantum_eff': 0.4,
            'px_saturation_e': 3500,
            'dark_noise_mu': 25, 'dark_noise_sd': 5, 'readout_noise_sd': 5,
        })
        common_kwargs = common_kwargs_best

        cam = Camera(
            w,  # width in pixels
            h,  # height in pixels
            62.554,  # x fov in degrees  (could be 6 & 5.695, 5.15 & 4.89, 7.7 & 7.309)
            37.726,  # y fov in degrees
            f_stop=5,  # TODO: put better value here
            point_spread_fn=0.50,  # ratio of brightness in center pixel
            scattering_coef=2e-10,  # affects strength of haze/veil when sun shines on the lens
            dist_coefs=None if 0 else [-0.11250615, 0.14296794, -0.00175085, 0.00057391, -0.11678778],
            cam_mx=np.array([[1.58174667e+03, 0.00000000e+00, 9.97176182e+02],
                             [0.00000000e+00, 1.58154569e+03, 5.15553843e+02],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            **common_kwargs
        )

        return cam

    def init_odo(self):
        sc = 0.5

        params = {
            'use_ba': True,
            'new_keyframe_ba': False,
            'threaded_ba': False,    # TODO: debug adjustment to new keyframes & 3d points (!!) after ba completed

            'online_cam_calib': False,
            'verify_feature_tracks': True,
            'max_keypoints': 320,                # 315
            'min_keypoint_dist': round(50 * sc),
            'min_tracking_quality': 0.0005,      # def 0.0001, was 0.0003
            'repr_refine_kp_uv': False,
            'repr_refine_coef': 0.2,
            'refine_kp_uv': False,
            'max_kp_refine_dist': 5,

            'repr_err_fov_ratio': 0.0005,          # was 0.002
            'max_repr_err_fov_ratio': 0.003,      # was 0.003
            'est_2d2d_prob': 0.9999,
            'pose_2d2d_quality_lim': 0.04,

            'use_3d2d_ransac': False,
            'opt_init_ransac': False,
            # 'est_3d2d_iter_count': 10000,
            'new_kf_min_displ_fov_ratio': 0.016,
            'ini_kf_triangulation_trigger': 40,

            'max_keyframes': 50000,
            'max_ba_keyframes': 60,
            'ba_interval': 10,
            'max_ba_fun_eval': 25 * 2,
            'loc_err_sd': 2,
            'ori_err_sd': math.radians(5.0),
        }

        if 0:
            params.update({
                'orb_feature_tracking': True,
                'orb_search_dist': 0.03,            # in fov ratio
                'max_keypoints': 2000,
                'min_keypoint_dist': round(20 * sc),
                'repr_err_fov_ratio': 0.006,        # was 0.004
                'max_repr_err_fov_ratio': 0.008,    # was 0.006
            })

        logging.basicConfig(level=logging.INFO)
        odo = VisualGPSNav(self.cam, round(self.cam.width * sc),
                           geodetic_origin=self.coord0,
                           wf_body_q=self.w2b_q,
                           bf_cam_pose=Pose(np.array([0, 0, 0]), self.b2c_q),
                           verbose=1, pause=False, **params)
        return odo
