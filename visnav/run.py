import os
import argparse
import re
from datetime import datetime
import logging

import matplotlib.pyplot as plt
import numpy as np
import quaternion
import cv2

from visnav.algo.image import ImageProc
from visnav.algo.model import Camera
from visnav.algo.odometry import VisualOdometry, Pose


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Run visual odometry on a set of images')
    parser.add_argument('--dir', '-d', metavar='DIR', help='path to the images')
    args = parser.parse_args()

    # init odometry
    odo, prior, time = init()

    # find images
    img_files = []
    for fname in os.listdir(args.dir):
        m = re.search(r"(\d+)\.(png|jpg|jpeg)$", fname)
        if m:
            img_files.append((int(m[1]), fname))
    img_files = sorted(img_files, key=lambda x: x[0])

    # run odometry
    loc = np.ones((len(img_files), 3)) * np.nan
    for i, (_, fname) in enumerate(img_files):
        logging.info('')
        logging.info(fname)
        img = cv2.imread(os.path.join(args.dir, fname), cv2.IMREAD_GRAYSCALE)[2:-2, :]

        # normalize brightness
        img = np.clip(img.astype(np.float) * 255 / np.percentile(img, 99.8), 0, 255).astype(np.uint8)

        #img = np.clip(img.astype(np.uint16)*2, 0, 255).astype(np.uint8)
        #img = ImageProc.adjust_gamma(img, 1.8)

        res = odo.process(img, datetime.fromtimestamp(time + i*1), prior, quaternion.one)
        if res and res[0] and res[0].post:
            prior = res[0].post
            if res[0].method == VisualOdometry.POSE_RANSAC_3D:
                loc[i, :] = res[0].post.loc

    logging.disable(logging.INFO)
    plt.plot(loc[:, 0], loc[:, 2], 'x-')
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()
    print('ok: %.1f%%, delta loc std: %.3e' % (
        100*(1 - np.mean(np.isnan(loc[:, 0]))),
        np.nanstd(np.linalg.norm(np.diff(loc, axis=0), axis=1)),
    ))


def init():
    params = {
        'use_ba': True,
        'threaded_ba': True,
        'max_ba_fun_eval': 20,
        'asteroid': False,
    }
    cam = get_cam()
    logging.basicConfig(level=logging.INFO)
    odo = VisualOdometry(cam, cam.width/2, verbose=1, pause=False,
                         use_scale_correction=False, est_cam_pose=False, **params)
    prior = Pose(np.array([0, 0, 0]), quaternion.one, np.ones((3,)) * 0.1, np.ones((3,)) * 0.01)
    time = datetime.strptime('2020-07-01 15:42:00', '%Y-%m-%d %H:%M:%S').timestamp()
    return odo, prior, time


def get_cam():
    w, h = 1280, 960
    common_kwargs_worst = {
        'sensor_size': (w * 0.00375, h * 0.00375),
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
        43.6,  # x fov in degrees  (could be 6 & 5.695, 5.15 & 4.89, 7.7 & 7.309)
        33.4,  # y fov in degrees
        f_stop=5,  # TODO: put better value here
        point_spread_fn=0.50,  # ratio of brightness in center pixel
        scattering_coef=2e-10,  # affects strength of haze/veil when sun shines on the lens
        dist_coefs=[-3.79489919e-01, 2.55784821e-01, 9.52433459e-04, 1.27543923e-04, -2.74301340e-01],   # by using calibrate.py
        cam_mx=np.array([[1.60665503e+03, 0.00000000e+00, 6.12522544e+02],
                        [0.00000000e+00, 1.60572265e+03, 4.57510418e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        **common_kwargs
    )

    return cam


if __name__ == '__main__':
    main()