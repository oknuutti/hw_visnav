import argparse
import copy
import pickle
from datetime import datetime
import logging

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import quaternion

from visnav.algo import tools
from visnav.algo.odo.base import VisualOdometry, Pose
from visnav.missions.hwproto import HardwarePrototype
from visnav.missions.nokia import NokiaSensor


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Run visual odometry on a set of images')
    parser.add_argument('--data', '-d', metavar='DATA', help='path to data')
    parser.add_argument('--meta', '-t', metavar='META', help='path to meta data')
    parser.add_argument('--video-toff', '--dt', type=float, metavar='dT', help='video time offset compared to metadata')
    parser.add_argument('--out', '-o', metavar='OUT', help='path to the output folder')
    parser.add_argument('--mission', '-m', choices=('hwproto', 'nokia'), help='select mission')
    parser.add_argument('--skip', '-s', type=int, default=1, help='use only every xth frame (default: 1)')
    args = parser.parse_args()

    # init odometry and data
    if args.mission == 'hwproto':
        mission = HardwarePrototype(args.data, last_frame=(155, 321, None)[0])
    elif args.mission == 'nokia':
        mission = NokiaSensor(args.data, data_path=args.meta, video_toff=args.video_toff,
                              first_frame=(350, 1650)[0], last_frame=(500, 1850)[0])
    else:
        assert False, 'bad mission given: %s' % args.mission

    if args.out:
        mission.odo._track_save_path = args.out

    # run odometry
    prior = Pose(np.array([0, 0, 0]), quaternion.one)
    frame_names0 = []
    meta_names0 = []
    results0 = []
    ground_truth0 = []
    kfid2img = {}
    started = datetime.now()
    ax = None

    for i, (img, t, name, meta, meta_name, gt) in enumerate(mission.data):
        if i % args.skip != 0:
            continue

        logging.info('')
        logging.info(name)
        frame_names0.append(name)
        meta_names0.append(meta_name)
        ground_truth0.append(gt)

        try:
            nf, *_ = mission.odo.process(img, datetime.fromtimestamp(mission.time0 + t), measure=meta)

            if 1 and nf is not None and nf.id is not None:
                kfid2img[nf.id] = i

                w2c_q = NokiaSensor.w2b_q * NokiaSensor.b2c_q
                post = np.zeros((len(mission.odo.state.keyframes), 7))
                k, prior = 0, np.zeros((len(mission.odo.state.keyframes), 7))
                for j, kf in enumerate([kf for kf in mission.odo.state.keyframes if kf.pose.post]):
                    post[j, :3] = tools.q_times_v(w2c_q * kf.pose.post.quat.conj(), -kf.pose.post.loc)
                    post[j, 3:] = quaternion.as_float_array(w2c_q.conj() * kf.pose.post.quat.conj() * w2c_q)
                    if kf.measure is not None:
                        prior[k, :3] = tools.q_times_v(w2c_q * kf.pose.prior.quat.conj(), -kf.pose.prior.loc)
                        prior[k, 3:] = quaternion.as_float_array(w2c_q.conj() * kf.pose.prior.quat.conj() * w2c_q)
                        k += 1
                if ax is not None:
                    ax.clear()
                ax = tools.plot_poses(post, axis=(0, 1, 0), up=(0, 0, 1), ax=ax, wait=False)
                tools.plot_poses(prior[:k, :], axis=(0, 1, 0), up=(0, 0, 1), ax=ax, wait=False,
                                 colors=map(lambda c: (c, 0, 0, 0.5), np.linspace(.3, 1.0, k)))

        except TypeError as e:
            if 0:
                nf, *_ = mission.odo.process(img, datetime.fromtimestamp(mission.time0 + t), prior, quaternion.one)
                if nf and nf.pose.post:
                    prior = nf.pose.post
            else:
                raise e

        results0.append(None if nf is None or nf.pose is None or nf.pose.post is None else
                        (nf.pose, getattr(nf, 'measure', None), nf.time, nf.id))

    try:
        mission.odo.flush_state()  # flush all keyframes and keypoints
        map3d = mission.odo.removed_keypoints
        results = [(kf.pose, kf.measure, kf.time, kf.id) for kf in mission.odo.removed_keyframes if kf.pose.post]
        frame_names = [frame_names0[kfid2img[r[3]]] for r in results]
        meta_names = [meta_names0[kfid2img[r[3]]] for r in results]
        ground_truth = [ground_truth0[kfid2img[r[3]]] for r in results]
    except AttributeError as e:
        if 0:
            map3d = None
            results = [(kf.pose, None, kf.time, kf.id) for kf in mission.odo.state.keyframes]
        else:
            raise e

    mission.odo.quit()
    if 0:
        # show results as given by the online version of the algorithm
        results, frame_names, meta_names, ground_truth = results0, frame_names0, meta_names0, ground_truth0

    logging.info('time spent: %.0fs' % (datetime.now() - started).total_seconds())
    plot_results(results, map3d, frame_names, meta_names, ground_truth, '%s-result.pickle' % args.mission)


def plot_results(results=None, map3d=None, frame_names=None, meta_names=None, ground_truth=None, file='result.pickle'):
    if results is None:
        with open(file, 'rb') as fh:
            results, map3d, frame_names, meta_names, ground_truth = pickle.load(fh)
    else:
        with open(file, 'wb') as fh:
            pickle.dump((results, map3d, frame_names, meta_names, ground_truth), fh)

    w2c_q = NokiaSensor.w2b_q * NokiaSensor.b2c_q

    pose = np.ones((len(results), 7)) * np.nan
    ori = np.ones((len(results), 3)) * np.nan
    pose2 = np.ones((len(results), 7)) * np.nan
    ori2 = np.ones((len(results), 3)) * np.nan
    for i, res in enumerate(results):
        if res and res[0] and res[0].post:
            if res[0].method == VisualOdometry.POSE_2D3D:
                pose[i, :3] = tools.q_times_v(w2c_q * res[0].post.quat.conj(), -res[0].post.loc)
                pose2[i, :3] = tools.q_times_v(w2c_q * res[0].prior.quat.conj(), -res[0].prior.loc)
                pose[i, 3:] = quaternion.as_float_array(w2c_q.conj() * res[0].post.quat.conj() * w2c_q)
                pose2[i, 3:] = quaternion.as_float_array(w2c_q.conj() * res[0].prior.quat.conj() * w2c_q)
                ori[i, :] = tools.q_to_ypr(NokiaSensor.b2c_q.conj() * res[0].post.quat.conj() * NokiaSensor.b2c_q)
                ori2[i, :] = tools.q_to_ypr(NokiaSensor.b2c_q.conj() * res[0].prior.quat.conj() * NokiaSensor.b2c_q)

    ori = ori / np.pi * 180
    ori2 = ori2 / np.pi * 180

    logging.disable(logging.INFO)

    fst = np.where(np.logical_not(np.isnan(pose[:, 0])))[0][0]
    idx = np.where([r is not None for r in results])[0]
    idx2 = np.where([r is not None and r[1] is not None for r in results])[0]
    t0 = results[idx[0]][2].timestamp()

    t = np.array([results[i][2].timestamp() - t0 for i in idx])
    t2 = np.array([results[i][2].timestamp() - t0 + results[i][1].time_off + results[i][1].time_adj for i in idx2])
    dt = np.array([results[i][1].time_adj for i in idx2])

    rng = np.nanmax(pose[idx, :2], axis=0) - np.nanmin(pose[idx, :2], axis=0)
    rng2 = 0 if len(idx2) == 0 else np.nanmax(pose2[idx2, :2], axis=0) - np.nanmin(pose2[idx2, :2], axis=0)
    mrg = 0.05 * max(np.max(rng), np.max(rng2))
    min_x = min(np.nanmin(pose[idx, 0]), 9e9 if len(idx2) == 0 else np.nanmin(pose2[idx2, 0])) - mrg
    max_x = max(np.nanmax(pose[idx, 0]), -9e9 if len(idx2) == 0 else np.nanmax(pose2[idx2, 0])) + mrg
    min_y = min(np.nanmin(pose[idx, 1]), 9e9 if len(idx2) == 0 else np.nanmin(pose2[idx2, 1])) - mrg
    max_y = max(np.nanmax(pose[idx, 1]), -9e9 if len(idx2) == 0 else np.nanmax(pose2[idx2, 1])) + mrg
    min_z = min(np.nanmin(pose[idx, 2]), 9e9 if len(idx2) == 0 else np.nanmin(pose2[idx2, 2])) - mrg
    max_z = max(np.nanmax(pose[idx, 2]), -9e9 if len(idx2) == 0 else np.nanmax(pose2[idx2, 2])) + mrg

    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 10), np.linspace(min_y, max_y, 10))
    zz = np.ones(xx.shape) * min(min_z + mrg/2, 0)
    min_z = min(0, min_z)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz, zorder=2)
    line = ax.plot(pose[idx, 0], pose[idx, 1], pose[idx, 2], 'C0', zorder=3)
    ax.scatter(pose[fst, 0], pose[fst, 1], pose[fst, 2], 'C2', zorder=4)
    line2 = ax.plot(pose2[idx2, 0], pose2[idx2, 1], pose2[idx2, 2], 'C1', zorder=5)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    tools.hover_annotate(fig, ax, line[0], frame_names)
    tools.hover_annotate(fig, ax, line2[0], [meta_names[i] for i in idx2])

    if 1:
        fig, axs = plt.subplots(1, 1)
        axs = [axs]
        axs[0].set_aspect('equal')
        rng_x = max_x - min_x
        rng_y = max_y - min_y

        if True or rng_x > rng_y:
            axs[0].plot(pose[fst, 0], pose[fst, 1], 'oC0', mfc='none')
            line = axs[0].plot(pose[idx, 0], pose[idx, 1], 'C0')  # , '+-')
            line2 = axs[0].plot(pose2[idx2, 0], pose2[idx2, 1], 'C1')  # , '+-')
            axs[0].set_xlim(min_x, max_x)
            axs[0].set_ylim(min_y, max_y)
            axs[0].set_xlabel('x')
            axs[0].set_ylabel('y')
        else:
            axs[0].plot(pose[fst, 1], pose[fst, 0], 'oC0', mfc='none')
            line = axs[0].plot(pose[idx, 1], pose[idx, 0], 'C0')  # , '+-')
            line2 = axs[0].plot(pose2[idx2, 1], pose2[idx2, 0], 'C1')  # , '+-')
            axs[0].set_xlim(min_y, max_y)
            axs[0].set_ylim(min_x, max_x)
            axs[0].set_xlabel('y')
            axs[0].set_ylabel('x')

        tools.hover_annotate(fig, axs[0], line[0], frame_names)
        tools.hover_annotate(fig, axs[0], line2[0], [meta_names[i] for i in idx2])

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(t[fst], pose[fst, 2], 'oC0', mfc='none')
        line = axs[0].plot(t, pose[idx, 2], 'C0')  # , '+-')
        line2 = axs[0].plot(t2, pose2[idx2, 2], 'C1')  # , '+-')
        tools.hover_annotate(fig, axs[0], line[0], frame_names)
        tools.hover_annotate(fig, axs[0], line2[0], [meta_names[i] for i in idx2])
        axs[0].set_ylabel('z')
        axs[0].set_xlabel('t')

        line = axs[1].plot(t2, dt, 'C0')  #, '+-')
        tools.hover_annotate(fig, axs[1], line[0], [meta_names[i] for i in idx2])
        axs[1].set_ylabel('dt')
        axs[1].set_xlabel('t')

        fig, axs = plt.subplots(3, 1)
        for i, title in enumerate(('yaw', 'pitch', 'roll')):
            axs[i].plot(t[fst], ori[fst, i], 'oC0', mfc='none')
            line = axs[i].plot(t, ori[idx, i], 'C0')  # , '+-')
            line2 = axs[i].plot(t2, ori2[idx2, i], 'C1')  # , '+-')
            tools.hover_annotate(fig, axs[i], line[0], frame_names)
            tools.hover_annotate(fig, axs[i], line2[0], [meta_names[j] for j in idx2])
            axs[i].set_ylabel(title)
            axs[i].set_xlabel('t')

    plt.tight_layout()

    tools.plot_poses(pose[idx, :], axis=(0, 1, 0), up=(0, 0, 1))

    plt.show()
    print('ok: %.1f%%, delta loc std: %.3e' % (
        100*(1 - np.mean(np.isnan(pose[:, 0]))),
        np.nanstd(np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)),
    ))


if __name__ == '__main__':
    if 1:
        main()
    elif 0:
        plot_results(file='hwproto-result.pickle')
    else:
        plot_results(file='nokia-result.pickle')
