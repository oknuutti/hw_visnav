import logging
import math
from datetime import datetime
import re

import numpy as np
import quaternion
import cv2
from tqdm import tqdm

from TelemetryParsing import readTelemetryCsv

from visnav.algo import tools
from visnav.algo.tools import Pose
from visnav.algo.model import Camera
from visnav.algo.odo.base import Measure
from visnav.algo.odo.visgps_odo import VisualGPSNav
from visnav.algo.odometry import VisualOdometry

from visnav.missions.base import Mission

UNDISTORT_IMAGE = False
DIST_COEF_N = 1


if 0:
    # nokia calibration
    DIST_COEFFS = [-0.10409018071479297, 0.07753040773494844, -0.0012432655612101095, -8.782982357309616e-05, 0.0]
    CALIB_K = [1580.356552415608, 0.0, 994.0266973706297,
               0.0, 1580.5531767058067, 518.9387257616232,
               0.0, 0.0, 1.0]
elif DIST_COEF_N == 5:
    # own calib, 5 dist coeffs
    # DIST_COEFFS = [-0.11250615, 0.14296794, -0.00175085, 0.00057391, -0.11678778]
    c, d = 1.0, 1.0     # 1.15, 5.31, 1.456, 1.12
    DIST_COEFFS = [-0.11250615, 0.14296794, -0.00175085, 0.00057391, -0.11678778]
    CALIB_K = [1.58174667e+03*c, 0.00000000e+00, 9.97176182e+02,
               0.00000000e+00, 1.58154569e+03*c, 5.15553843e+02*d,
               0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
elif DIST_COEF_N == 8:
    # own calib, 8 dist coeffs
    DIST_COEFFS = [-6.16252484e+00,  7.24901612e+00, -1.78015022e-03,  6.20574203e-04,
                    9.73407083e+00, -6.05836470e+00,  6.52823697e+00,  1.10393054e+01]
    CALIB_K = [[1.58091067e+03, 0.00000000e+00, 9.98071831e+02],
               [0.00000000e+00, 1.58073779e+03, 5.15360237e+02],
               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
elif DIST_COEF_N == 2:
    # own calib, 2 dist coeffs
    DIST_COEFFS = [-0.10364428,  0.07621745,  0.,          0.,          0.]
    CALIB_K = [[1.58089753e+03, 0.00000000e+00, 9.94527661e+02],    # 9.94527661e+02],
               [0.00000000e+00, 1.58053141e+03, 5.22590026e+02],    # 5.22590026e+02],
               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
elif DIST_COEF_N == 1:
    # own calib, 1 dist coeffs
    # DIST_COEFFS = [-0.07282351,  0.,  0.,          0.,          0.]     # -0.07282351
    # CALIB_K = [[1.57982944e+03, 0.00000000e+00, 1.00429069e+03],
    #            [0.00000000e+00, 1.57982258e+03, 5.22368437e+02],
    #            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

    # NOTE: adjust fl, k1 => level pitch, ground distance (altitude) == expected dist
    #        - +fl => +a0 alt, +b0 pitch
    #        - +k1 => -a1 alt, -b1 pitch
    #        - where b0 < b1, a0 > a1
    if 0:
        DIST_COEFFS = [-0.07,  0.,  0., 0., 0.]    # -0.06     (-0.047 - -0.54)
        CALIB_K = [[1600., 0., 969.],               # 1580, 960
                   [0., 1600., 515.],               # 1580, 400
                   [0., 0., 1.]]
    else:
        DIST_COEFFS = [0.125,  0.,  0., 0., 0.]      # 0.123
        CALIB_K = [[2474.3, 0., 969.],               # 1580, 960
                   [0., 2474.3, 515.],               # 1580, 400
                   [0., 0., 1.]]

else:
    assert False, 'wrong num for DIST_COEF_N'


class NokiaSensor(Mission):
    COORD0 = (60.08879, 24.35987, 0.629)    # reference (lat, lon, alt) from 2021_10_07_10_53_12_11_03_00.csv

    CAM_WIDTH = 1920
    CAM_HEIGHT = 1080

    # world frame: +z up, +x is east, +y is north
    # body frame: +z down, +x is fw towards north, +y is right wing (east)
    # camera frame: +z into the image plane, -y is up (north), +x is right (east)
    w2b = Pose(None, tools.eul_to_q((np.pi, -np.pi / 2), 'xz'))
    b2c = Pose(None, tools.eul_to_q((np.pi / 2,), 'z'))

    def __init__(self, *args, verbosity=2, high_quality=False, ori_off_q=None,
                 use_gimbal=False, cam_mx=None, cam_dist=None, undist_img=False, **kwargs):
        self.verbosity = verbosity
        self.high_quality = high_quality

        self.undist_img = undist_img
        self.use_gimbal = use_gimbal
        self.cam_mx = cam_mx or CALIB_K

        self.ori_off_q = ori_off_q

        tmp = cam_dist or DIST_COEFFS
        self.cam_dist_n = np.where(np.array(tmp) != 0)[0][-1] + 1
        self.cam_dist = [0.0] * max(5, self.cam_dist_n)
        for i in range(self.cam_dist_n):
            self.cam_dist[i] = tmp[i]

        super(NokiaSensor, self).__init__(*args, **kwargs)

    def init_data(self):
        t_time, *t_data = readTelemetryCsv(self.data_path, None, None)
        coord0 = t_data[0][0], t_data[1][0], t_data[2][0]
        time0 = t_time[0].astype(np.float64) / 1e6
        t_time = (t_time - t_time[0]).astype(np.float64) / 1e6
        t_data = list(zip(*t_data))

        # hardcoded for now
        m = re.search(r'(^|\\|/)([\w-]+)\.mp4$', self.video_path)
        tmp = {
            'HD_CAM_2020_12_17_14_50_40': [[5019, 9.0]],               # 14 (by eye: 6300-6700 & 7800-8100)
            'HD_CAM_2021_02_04_15_18_17': [[5447, 6.9]],               # 18 (by eye: 6300-6700 & 11750-12150)
            'HD_CAM_2021_03_04_11_50_25': [[5149, 3.0], [8152, 8.9]],  # 19 (by eye: 5150-5400; 8220-8500 & 9400-9650)
            'HD_CAM_2021_03_04_13_33_15': [[4965, 6.9], [8013, 1.3]],  # 20 (by eye: 6500-6900; 9150-9600)
        }.get(m[2], [])
        gap_offset_fids, gap_offsets = map(list, zip(*tmp)) if len(tmp) > 0 else ([], [])

        if self.undist_img:
            cam = self.init_cam()
            map_u, map_v = cv2.initUndistortRectifyMap(cam.cam_mx if self.cam_mx is None else np.array(self.cam_mx).reshape((3, 3)),
                                                       np.array(self.cam_dist), None,
                                                       cam.cam_mx, (cam.width, cam.height), cv2.CV_16SC2)

        def data_gen():
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            f_id, t_id, t_t, f_t_raw1 = 0, 0, 0, None
            missing_frames, last_measure, ret = False, False, True
            fts, fdt, inspect = [], [], False
            if inspect:
                pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            while cap.isOpened() and ret and not last_measure:
                f_id += 1
                ret, img = cap.read()
                if not ret or img is None:
                    continue

                f_t_raw0, f_t_raw1 = f_t_raw1, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

                if inspect and f_t_raw0 is not None:
                    pbar.update(1)
                    fdt.append((f_id, f_t_raw1 - f_t_raw0))

                # For some reason sometimes CAP_PROP_POS_MSEC returns zero. Othertimes, f_t_raw1 - f_t_raw0 is
                # very short (0.01-25ms or even negative, e.g. -100ms), even though frames seem to progress normally.
                # It does seem to give some information still, as delays ranging from 200ms to 2s are observed and they
                # seem to correspond to "jumps" in images.
                if f_t_raw0 is not None:
                    if f_t_raw1 == 0:
                        f_t_raw1 = f_t_raw0 + 1/fps
                    elif f_t_raw1 - f_t_raw0 < 0.5/fps:
                        self.video_toff += 1/fps - (f_t_raw1 - f_t_raw0)

                f_t = f_t_raw1 + self.video_toff
                if self.first_frame is not None and f_id < self.first_frame and not inspect:
                    continue
                elif self.last_frame is not None and f_id > self.last_frame and not inspect:
                    break
                elif np.all(img == img[0, 0, 0]):
                    missing_frames = True
                    if inspect:
                        fts.append((f_id, f_t))
                    continue

                if missing_frames:
                    missing_frames = False
                    err_msg = 'Missing frames before frame id %d, no custom time offset given' % f_id
                    if f_id not in gap_offset_fids:
                        if inspect:
                            print(err_msg)
                        else:
                            assert False, err_msg

                while len(gap_offset_fids) > 0 and f_id >= gap_offset_fids[0]:
                    gap_offset_fids.pop(0)
                    t_off = gap_offsets.pop(0)
                    self.video_toff += t_off
                    f_t += t_off

                if inspect:
                    fts.append((f_id, f_t))
                    continue

                # read a measurement
                meas = None
                if t_t <= f_t + 0.5/fps:
                    later_measures = np.where(t_time > f_t + 0.5 / fps)[0]
                    if len(later_measures) <= 1:
                        last_measure = True
                    t_id = later_measures[0] - 1
                    t_t = t_time[t_id]
                    lat, lon, alt, roll, pitch, yaw, *gimbal = t_data[t_id]
                    gimbal_roll, gimbal_pitch, gimbal_yaw = map(math.radians, gimbal)

                    incremental = False
                    nadir_pointing = True

                    if incremental:
                        if self.use_gimbal:
                            # seems that gimbal overrides regular (roll, pitch, yaw) instead of being relative to body
                            b_q = tools.ypr_to_q(yaw, pitch, roll)
                            c_q = tools.ypr_to_q(gimbal_yaw, gimbal_pitch, gimbal_roll)
                            gimbal_yaw, gimbal_pitch, gimbal_roll = tools.q_to_ypr(b_q.conj() * c_q)
                        else:
                            # enable for datasets with no gimbal information
                            gimbal_roll, gimbal_pitch, gimbal_yaw = 0, math.radians(-90 if nadir_pointing else -20), 0
                    else:
                        if self.use_gimbal:
                            # TODO: make sure pitch -20 deg => 70 deg
                            yaw, pitch, roll = gimbal_yaw, gimbal_pitch, gimbal_roll
                            # if 1:
                            #     gf_world_cam = Pose(None, tools.ypr_to_q(yaw, pitch, roll))
                            #     if 0:
                            #         off_q = tools.ypr_to_q(*map(math.radians, (-1.39653216, -16.87428421, 1.66370009)))
                            #         yaw, pitch, roll = tools.q_to_ypr(gf_world_cam.quat * off_q)
                            #     elif 0:
                            #         b2g = Pose(None, tools.ypr_to_q(*map(math.radians, (-3.56934931, 1.1271115, 2.04077157))))
                            #         yaw, pitch, roll = tools.q_to_ypr(gf_world_cam.to_global(b2g).quat)
                            # elif 1:
                            #     yaw, pitch, roll = np.array([yaw, pitch, roll]) + np.array(list(map(math.radians,
                            #                        (0.0, -5.5, 7.5))))
                        else:
                            pitch, roll = math.radians(0 if nadir_pointing else 70), 0
                        gimbal_yaw, gimbal_pitch, gimbal_roll = 0, 0, 0

                    meas = Measure(data=np.array([lat, lon, alt, roll, pitch, yaw,
                                                  gimbal_roll, gimbal_pitch, gimbal_yaw]), time_off=t_t - f_t)

                    if t_id >= len(t_time):
                        # last measure used
                        last_measure = True
                    else:
                        t_t = t_time[t_id + 1]

                if 0:
                    img, _ = self.preprocess_image(img, 2.0)

                if self.undist_img:
                    img = cv2.remap(img, map_u, map_v, interpolation=cv2.INTER_LINEAR)

                f_ts = datetime.fromtimestamp(f_t + time0 - self.video_toff).strftime('%H:%M:%S.%f')
                t_ts = datetime.fromtimestamp(t_t + time0).strftime('%H:%M:%S.%f')
                yield img, f_t, 'frame-%d (%s)' % (f_id, f_ts), meas, ('meas-%d (%s)' % (t_id, t_ts)) if meas else None, None

            cap.release()

            if inspect:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

                fdt = np.array(fdt)
                axs[0].plot(fdt[:-1, 0], fdt[1:, 1])
                axs[0].set_title('raw frame times')

                fts = np.array(fts)
                axs[1].plot(fts[:-1, 0], np.diff(fts[:, 1]))
                axs[1].set_title('corrected frame times')

                plt.show()

        return data_gen(), time0, NokiaSensor.COORD0

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
        w, h, p = NokiaSensor.CAM_WIDTH, NokiaSensor.CAM_HEIGHT, 0.00375
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
            dist_coefs=None if self.undist_img else self.cam_dist,
            cam_mx=np.array(self.cam_mx).reshape((3, 3)),
            undist_proj_mx=None if self.undist_img or self.cam_mx is None else np.array(self.cam_mx).reshape((3, 3)),
            **common_kwargs
        )

        return cam

    def init_odo(self):
        sc = 0.5

        params = {
            'use_ba': True,
            'new_keyframe_ba': False,
            'threaded_ba': False,    # TODO: debug adjustment to new keyframes & 3d points (!!) after ba completed

            'rolling_shutter': False,
            'rolling_shutter_axis': '-y',
            'rolling_shutter_delay': 30e-3,  # delay in secs between first and last scanned line

            'online_cam_calib': self.cam_dist_n if 0 else 0,
            'verify_feature_tracks': True,
            'max_keypoints': 320,                # 320
            'min_keypoint_dist': round(50 * sc),
            'min_tracking_quality': 0.0005,      # def 0.0001, was 0.0003 then 0.0005
            'repr_refine_kp_uv': False,
            'repr_refine_coef': 0.2,
            'refine_kp_uv': False,
            'max_kp_refine_dist': 5,

            'repr_err_fov_ratio': 0.0005,         # was 0.002 then 0.0005
            'max_repr_err_fov_ratio': 0.003,      # was 0.003
            'est_2d2d_prob': 0.9999,
            'pose_2d2d_quality_lim': 0.04,
            'check_2d2d_result': False,

            'use_3d2d_ransac': False,
            'opt_init_ransac': False,
            # 'est_3d2d_iter_count': 10000,
            'ini_kf_triangulation_trigger': 40,

            'new_kf_min_kp_ratio': 0.80,                    # remaining inliers from previous keyframe features
            'new_kf_min_displ_fov_ratio': 0.016,            # displacement relative to the fov for triangulation
            'new_kf_triangulation_trigger_ratio': 1.01 if 1 else 0.1,     # ratio of 2d points tracked that can be triangulated
            'new_kf_rot_angle': math.radians(10),           # new keyframe if orientation changed by this much
            'new_kf_min_kp_displ': 0.032,                    # fov relative displacement for significant viewpoint change
            'new_kf_kp_displ_ratio': 0.2,                   # new keyframe if ratio of keypoints with
                                                            # significant viewpoint change surpasses this

            'max_keyframes': 48,
            'ba_interval': 3,
            'window_fifo_len': 48,
            'max_ba_fun_eval': 100 * 10,
            'loc_err_sd': np.inf if 0 else np.array([3., 3., 3.]),  # y == alt (was [2, 10, 2])
            'ori_err_sd': np.inf if 0 else math.radians(60.0),
            'min_retain_obs': 4,

            # TODO: find out why dist coef and cam intrinsics optimization doesnt work (jacobian seem to be correct)
            'ba_dist_coef': False,         # optimize k1, k2
            'ba_n_cam_intr': 0,            # optimize 1) focal length only, 2) principal point only, 3) both fl and pp

            'enable_marginalization': True,
        }

        if self.high_quality:
            params.update({
                'max_keypoints': 1000,                # 320
                'min_keypoint_dist': round(25 * sc),
                'min_tracking_quality': 0.0003,      # def 0.0001, was 0.0003 then 0.0005
            })

        if 0:
            params.update({
                'detection_grid': (1, 1),
                'orb_feature_tracking': True,
                'orb_search_dist': 0.04,            # in fov ratio
                'max_keypoints': 500,
                'min_keypoint_dist': round(0 * sc),
                'repr_err_fov_ratio': 0.006,        # was 0.004
                'max_repr_err_fov_ratio': 0.012,    # was 0.006
            })

        logging.basicConfig(level=logging.INFO)
        odo = VisualGPSNav(self.cam, round(self.cam.width * sc),
                           geodetic_origin=self.coord0,
                           wf2bf=self.w2b, bf2cf=self.b2c, ori_off_q=self.ori_off_q,
                           verbose=self.verbosity, pause=False, **params)  # 1: logging, 2: tracks, 3: 3d-map, 4: poses
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
