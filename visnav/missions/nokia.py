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

UNDISTORT_IMAGE = False

if 0:
    # nokia calibration
    DIST_COEFFS = [-0.10409018071479297, 0.07753040773494844, -0.0012432655612101095, -8.782982357309616e-05, 0.0]
    CALIB_K = [1580.356552415608, 0.0, 994.0266973706297, 0.0, 1580.5531767058067, 518.9387257616232, 0.0, 0.0, 1.0]
    CALIB_P = [1533.907470703125, 0.0, 994.5594689049904, 0.0, 1560.9056396484375, 517.2969790578827, 0.0, 0.0, 1.0]
else:
    # own calib
    DIST_COEFFS = [-0.11250615, 0.14296794, -0.00175085, 0.00057391, -0.11678778]
    CALIB_K = [1.58174667e+03, 0.00000000e+00, 9.97176182e+02, 0.00000000e+00, 1.58154569e+03, 5.15553843e+02,
               0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    CALIB_P = [1.51946704e+03, 0.00000000e+00, 9.99938541e+02, 0.00000000e+00, 1.51758582e+03, 5.12326558e+02,
               0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    CALIB_P, CALIB_K = CALIB_K, None


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
        if UNDISTORT_IMAGE:
            cam = self.init_cam()
            map_u, map_v = cv2.initUndistortRectifyMap(np.array(CALIB_K).reshape((3, 3)), np.array(DIST_COEFFS), None,
                                                       cam.cam_mx, (cam.width, cam.height),
                                                       cv2.CV_16SC2)

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

                    if 1:
                        roll, pitch = 0, 0
                    elif 0:
                        pitch = pitch - math.radians(5)

                    meas = Measure(data=np.array([lat, lon, alt, roll, pitch, yaw]), time_off=t_t - f_t)

                    if t_id >= len(t_time):
                        # last measure used
                        last_measure = True
                    else:
                        t_t = t_time[t_id + 1]

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if 0:
                    img, _ = self.preprocess_image(img, 2.0)

                if UNDISTORT_IMAGE:
                    img = cv2.remap(img, map_u, map_v, interpolation=cv2.INTER_LINEAR)

                f_ts = datetime.fromtimestamp(f_t + time0 - self.video_toff).strftime('%H:%M:%S.%f')
                t_ts = datetime.fromtimestamp(t_t + time0).strftime('%H:%M:%S.%f')
                yield img, f_t, 'frame-%d (%s)' % (f_id, f_ts), meas, ('meas-%d (%s)' % (t_id, t_ts)) if meas else None, None

            cap.release()

        return data_gen(), time0, coord0

    @staticmethod
    def preprocess_image(img, gamma):
        # data = np.atleast_3d(data)
        # bot_v, top_v = np.quantile(data[:, :, 0], (0.0005, 0.9999))
        # top_v = top_v * 1.2
        # img = (data[:, :, 0] - bot_v) / (top_v - bot_v)
        if gamma != 1:
            #img = np.clip(img.astype(float)/255, 0, 1) ** (1 / gamma)
            img = np.clip((img.astype(float)/255 - 0.2) * 2, 0, 255)
        img = np.clip(255 * img + 0.5, 0, 255).astype(np.uint8)
        return img, 1  # (bot_v, top_v)

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
            dist_coefs=None if UNDISTORT_IMAGE else DIST_COEFFS,
            cam_mx=np.array(CALIB_P).reshape((3, 3)),
            undist_proj_mx=None if UNDISTORT_IMAGE else np.array(CALIB_K).reshape((3, 3)),
            **common_kwargs
        )

        return cam

    def init_odo(self):
        sc = 0.5

        params = {
            'use_ba': True,
            'new_keyframe_ba': False,
            'threaded_ba': False,    # TODO: debug adjustment to new keyframes & 3d points (!!) after ba completed

            'verify_feature_tracks': True,
            'max_keypoints': 320,                # 315
            'min_keypoint_dist': round(50 * sc),
            'min_tracking_quality': 0.0005,      # def 0.0001, was 0.0003
            'repr_refine_kp_uv': False,
            'repr_refine_coef': 0.2,
            'refine_kp_uv': False,
            'max_kp_refine_dist': 5,

            'repr_err_fov_ratio': 0.0005,         # was 0.002
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
            'ba_interval': 6,
            'max_ba_fun_eval': 25 * 2,
            'loc_err_sd': 2,
            'ori_err_sd': math.radians(10.0),
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
                           verbose=2, pause=False, **params)
        return odo


def interp_loc(frames, time0):
    from scipy.interpolate import interp1d
    tmp = [(f.time.timestamp() - time0 + f.measure.time_off, (-f.pose.prior).loc)
           for f in frames if f.measure is not None]
    times, locs = map(np.array, zip(*tmp))

    # first filter
    locs = cv2.filter2D(locs.astype(np.float32), cv2.CV_32F, np.ones((3, 1)) / 3)
    funx = interp1d(times.astype(np.float32), locs[:, 0], kind='cubic', fill_value='extrapolate')
    funy = interp1d(times.astype(np.float32), locs[:, 1], kind='cubic', fill_value='extrapolate')
    funz = interp1d(times.astype(np.float32), locs[:, 2], kind='cubic', fill_value='extrapolate')
    return lambda t: np.array([[funx(t), funy(t), funz(t)]])
