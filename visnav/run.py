import argparse
import pickle
from datetime import datetime
import logging

import matplotlib.pyplot as plt
import numpy as np
import quaternion

from visnav.algo import tools
from visnav.algo.odometry import VisualOdometry
from visnav.missions.hwproto import HardwarePrototype
from visnav.missions.nokia import NokiaSensor


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Run visual odometry on a set of images')
    parser.add_argument('--data', '-d', metavar='DATA', help='path to the data folder')
    parser.add_argument('--mission', '-m', choices=('hwproto', 'nokia'), help='select mission')
    args = parser.parse_args()

    # init odometry and data
    if args.mission == 'hwproto':
        mission = HardwarePrototype(args.data, last_frame=(155, 321, None)[2])
    elif args.mission == 'nokia':
        mission = NokiaSensor(args.data, first_frame=(65, 3000)[0], last_frame=(6000, None)[0])
    else:
        assert False, 'bad mission given: %s' % args.mission

    # run odometry
    prior = mission.prior
    frame_names = []
    results = []
    ground_truth = []
    for i, (img, name, gt) in enumerate(mission.data):
        logging.info('')
        logging.info(name)
        frame_names.append(name)
        ground_truth.append(gt)

        res = mission.odo.process(img, datetime.fromtimestamp(mission.time0 + i*1), prior, quaternion.one)

        results.append(res)
        if res and res[0] and res[0].post:
            prior = res[0].post

    mission.odo.quit()
    plot_results(results, frame_names, ground_truth, '%s-result.pickle' % args.mission)


def plot_results(results=None, frame_names=None, ground_truth=None, file='result.pickle'):
    if results is None:
        with open(file, 'rb') as fh:
            results, frame_names, ground_truth = pickle.load(fh)
    else:
        with open(file, 'wb') as fh:
            pickle.dump((results, frame_names, ground_truth), fh)

    loc = np.ones((len(frame_names), 3)) * np.nan
    for i, res in enumerate(results):
        if res and res[0] and res[0].post:
            if res[0].method == VisualOdometry.POSE_RANSAC_3D:
                loc[i, :] = tools.q_times_v(res[0].post.quat.conj(), -res[0].post.loc)

    logging.disable(logging.INFO)

    fig, axs = plt.subplots(2, 1)
    axs[0].set_aspect('equal')
    rng = np.nanmax(loc[:, :2], axis=0) - np.nanmin(loc[:, :2], axis=0)
    mrg = 0.05 * np.max(rng)

    if rng[0] > rng[1]:
        line = axs[0].plot(loc[:, 0], loc[:, 1], '+-')
        axs[0].set_xlim(np.nanmin(loc[:, 0]) - mrg, np.nanmax(loc[:, 0]) + mrg)
        axs[0].set_ylim(np.nanmin(loc[:, 1]) - mrg, np.nanmax(loc[:, 1]) + mrg)
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
    else:
        line = axs[0].plot(loc[:, 1], loc[:, 0], '+-')
        axs[0].set_xlim(np.nanmin(loc[:, 1]) - mrg, np.nanmax(loc[:, 1]) + mrg)
        axs[0].set_ylim(np.nanmax(loc[:, 0]) + mrg, np.nanmin(loc[:, 0]) - mrg)
        axs[0].set_xlabel('y')
        axs[0].set_ylabel('x')

    tools.hover_annotate(fig, axs[0], line[0], frame_names)

    line = axs[1].plot(np.linspace(1, 100, len(loc[:, 2])), loc[:, 2], '+-')
    tools.hover_annotate(fig, axs[1], line[0], frame_names)
    axs[1].set_ylabel('z')
    axs[1].set_xlabel('t/T [%]')

    plt.tight_layout()
    plt.show()
    print('ok: %.1f%%, delta loc std: %.3e' % (
        100*(1 - np.mean(np.isnan(loc[:, 0]))),
        np.nanstd(np.linalg.norm(np.diff(loc, axis=0), axis=1)),
    ))


if __name__ == '__main__':
    if 1:
        main()
    elif 1:
        plot_results(file='hwproto-result.pickle')
    else:
        plot_results(file='nokia-result.pickle')
