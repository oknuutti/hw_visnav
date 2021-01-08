import logging
import os
import re
from datetime import datetime

import numpy as np
import quaternion
import cv2

from visnav.algo import tools
from visnav.algo.model import Camera
from visnav.algo.odometry import VisualOdometry, Pose
from visnav.missions.base import Mission


class NokiaSensor(Mission):
    def init_odo(self):
        params = {
            'use_ba': True,
            'threaded_ba': True,
            'max_keyframes': 8,
            'max_ba_keyframes': 8,
            'ba_interval': 4,
            'max_ba_fun_eval': 20,

            'min_keypoint_dist': 25,
            'pose_2d2d_quality_lim': 0.04,

            'asteroid': False,
        }
        logging.basicConfig(level=logging.INFO)
        odo = VisualOdometry(self.cam, self.cam.width / 3, verbose=1, pause=False,
                             use_scale_correction=False, est_cam_pose=False, **params)
        return odo

    def init_data(self):
        q = tools.eul_to_q((np.pi,), 'x')
        prior = Pose(np.array([0, 0, 0]), q, np.ones((3,)) * 0.1, np.ones((3,)) * 0.01)
        time0 = datetime.strptime('2020-12-17 15:42:00', '%Y-%m-%d %H:%M:%S').timestamp()

        def data_gen():
            cap = cv2.VideoCapture(self.datapath)
            i = 0
            while cap.isOpened():
                i += 1
                ret, frame = cap.read()
                if self.first_frame is not None and i < self.first_frame:
                    continue
                elif self.last_frame is not None and i > self.last_frame:
                    break
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield img, 'frame-%d' % i, None

            cap.release()

        return data_gen(), time0, prior

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
            dist_coefs=[-0.104090, 0.077530, -0.001243, -0.000088, 0.000000],
            cam_mx=np.array([[1580.356552, 0.000000, 994.026697],
                             [0.000000, 1580.553177, 518.938726],
                             [0.000000, 0.000000, 1.000000]]),
            **common_kwargs
        )

        return cam
