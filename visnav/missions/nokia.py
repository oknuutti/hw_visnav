import logging
import math
from datetime import datetime

import numpy as np
import quaternion
import cv2

from TelemetryParsing import readTelemetryCsv

from visnav.algo import tools
from visnav.algo.model import Camera
from visnav.algo.odo.base import Measure, Pose
from visnav.algo.odo.visgps_odo import VisualGPSNav
from visnav.algo.odometry import VisualOdometry

from visnav.missions.base import Mission


class NokiaSensor(Mission):
    # world frame: +z up, +x is east, +y is north
    # body frame: +z down, +x is fw towards north, +y is right wing (east)
    # camera frame: +z into the image plane, -y is up (north), +x is right (east)
    w2b_q = tools.eul_to_q((np.pi, -np.pi / 2), 'xz')
    b2c_q = tools.eul_to_q((np.pi / 2,), 'z')

    def init_data(self):
        t_time, *t_data = readTelemetryCsv(self.data_path, None, None)
        coord0 = t_data[0][0], t_data[1][0], t_data[2][0]
        time0 = t_time[0].astype(np.float64) / 1e6
        t_time = (t_time - t_time[0]).astype(np.float64) / 1e6
        t_data = list(zip(*t_data))

        def data_gen():
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            f_id, t_id, t_t = 0, 0, 0
            last_measure = False

            while cap.isOpened():
                if last_measure:
                    break

                f_id += 1
                ret, frame = cap.read()
                f_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 + self.video_toff
                if self.first_frame is not None and f_id < self.first_frame:
                    continue
                elif self.last_frame is not None and f_id > self.last_frame:
                    break

                # read a measurement
                meas = None
                if t_t <= f_t + 0.5/fps:
                    t_id = np.where(t_time > f_t + 0.5 / fps)[0][0] - 1
                    t_t = t_time[t_id]
                    lat, lon, alt, roll, pitch, yaw, gimbal_roll, gimbal_pitch, gimbal_yaw = t_data[t_id]
                    meas = Measure(data=np.array([lat, lon, alt, roll, pitch, yaw]), time_off=t_t - f_t)

                    if t_id >= len(t_time):
                        # last measure used
                        last_measure = True
                    else:
                        t_t = t_time[t_id + 1]

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                f_ts = datetime.fromtimestamp(f_t + time0 - self.video_toff).strftime('%H:%M:%S.%f')
                t_ts = datetime.fromtimestamp(t_t + time0).strftime('%H:%M:%S.%f')
                yield img, f_t, 'frame-%d (%s)' % (f_id, f_ts), meas, ('meas-%d (%s)' % (t_id, t_ts)) if meas else None, None

            cap.release()

        return data_gen(), time0, coord0

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
            dist_coefs=[-0.11250615, 0.14296794, -0.00175085, 0.00057391, -0.11678778],
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
            'threaded_ba': False,

            'max_keypoints': 315,
            'min_keypoint_dist': round(50 * sc),
            'min_tracking_quality': 0.0002,      # def 0.0001, was 0.0003

            'repr_err_fov_ratio': 0.0005,
            'max_repr_err_fov_ratio': 0.003,
            'est_2d2d_prob': 0.9999,
            'pose_2d2d_quality_lim': 0.04,

            'use_3d2d_ransac': True,
            'est_3d2d_iter_count': 10000,
            'new_kf_min_displ_fov_ratio': 0.016,
            'ini_kf_triangulation_trigger': 20,

            'max_keyframes': 64,
            'max_ba_keyframes': 64,
            'ba_interval': 8,
            'max_ba_fun_eval': 20,
            'loc_err_sd': 1.0,
            'ori_err_sd': math.radians(30.0),
        }

        logging.basicConfig(level=logging.INFO)
        if 1:
            odo = VisualGPSNav(self.cam, round(self.cam.width * sc),
                               geodetic_origin=self.coord0,
                               wf_body_q=self.w2b_q,
                               bf_cam_pose=Pose(np.array([0, 0, 0]), self.b2c_q),
                               verbose=2, pause=False, **params)
        else:
            params.pop('use_3d2d_ransac', False)
            params.pop('new_keyframe_ba', False)
            params.pop('loc_err_sd', False)
            params.pop('ori_err_sd', False)
            odo = VisualOdometry(self.cam, self.cam.width / 2, verbose=1, pause=False, **params)
        return odo
