import argparse
import copy
import pickle
from datetime import datetime
import logging
import os

import numpy as np
import quaternion

from visnav.algo import tools
from visnav.algo.tools import Pose
from visnav.algo.odo.base import VisualOdometry
from visnav.iotools.kapture import KaptureIO
from visnav.missions.hwproto import HardwarePrototype
from visnav.missions.nokia import NokiaSensor, interp_loc
from visnav.missions.toynokia import ToyNokiaSensor

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Run visual odometry on a set of images')
    parser.add_argument('--data', '-d', metavar='DATA', help='path to data')
    parser.add_argument('--meta', '-t', metavar='META', help='path to meta data')
    parser.add_argument('--video-toff', '--dt', type=float, metavar='dT', help='video time offset compared to metadata')
    parser.add_argument('--res', '-r', metavar='OUT', help='path to the result pickle')
    parser.add_argument('--debug-out', '-o', metavar='OUT', help='path to the debug output folder')
    parser.add_argument('--kapture', metavar='OUT', help='path to kapture-format export folder')
    parser.add_argument('--verbosity', '-v', type=int, default=2,
                        help='verbosity level (0-4, 0-1: text only, 2:+debug imgs, 3: +keypoints, 4: +poses)')
    parser.add_argument('--high-quality', action='store_true', help='high quality settings with more keypoints detected')

    parser.add_argument('--mission', '-m', choices=('hwproto', 'nokia', 'toynokia'), help='select mission')
    parser.add_argument('--undist-img', action='store_true', help='undistort image instead of keypoints')
    parser.add_argument('--use-gimbal', action='store_true', help='gimbal data is ok, use it')
    parser.add_argument('--nadir-looking', action='store_true', help='downwards looking cam')

    parser.add_argument('--cam-dist', type=float, nargs='*', help='cam distortion coeffs')
    parser.add_argument('--cam-fl-x', type=float, help='cam focal length x')
    parser.add_argument('--cam-fl-y', type=float, help='cam focal length y')
    parser.add_argument('--cam-pp-x', type=float, help='cam principal point x')
    parser.add_argument('--cam-pp-y', type=float, help='cam principal point y')

    parser.add_argument('--first-frame', '-f', type=int, default=0, help='first frame (default: 0; -1: hardcoded value)')
    parser.add_argument('--last-frame', '-l', type=int, help='last frame (default: None; -1: hardcoded end)')
    parser.add_argument('--skip', '-s', type=int, default=1, help='use only every xth frame (default: 1)')
    args = parser.parse_args()

    if args.verbosity > 1:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D

    # init odometry and data
    if args.mission == 'hwproto':
        mission = HardwarePrototype(args.data, last_frame=(155, 321, None)[0])
    elif args.mission == 'nokia':
        ff = (100, 250, 520, 750, 1250, 1500, 1750)[0] if args.first_frame == -1 else args.first_frame
        lf = (240, 400, 640, 1000, 1500, 1750, 2000)[1] if args.last_frame == -1 else args.last_frame

        cam_mx = None
        if args.cam_fl_x:
            fl_y = args.cam_fl_y or args.cam_fl_x
            pp_x = args.cam_pp_x or NokiaSensor.CAM_WIDTH / 2
            pp_y = args.cam_pp_y or NokiaSensor.CAM_HEIGHT / 2
            cam_mx = [[args.cam_fl_x, 0., pp_x],
                      [0.,          fl_y, pp_y],
                      [0.,            0., 1.]]
        mission = NokiaSensor(args.data, data_path=args.meta, video_toff=args.video_toff, use_gimbal=args.use_gimbal,
                              undist_img=args.undist_img, cam_mx=cam_mx, cam_dist=args.cam_dist,
                              verbosity=args.verbosity, high_quality=args.high_quality,
                              first_frame=ff, last_frame=lf)

    elif args.mission == 'toynokia':
        mission = ToyNokiaSensor(args.data, data_path=args.meta, video_toff=args.video_toff,
                                 first_frame=(100, 350, 850, 1650)[1], last_frame=(415, 500, 1250, 1850, 2000)[2])
    else:
        assert False, 'bad mission given: %s' % args.mission

    if args.debug_out:
        mission.odo._track_save_path = args.debug_out

    if args.nadir_looking:
        mission.odo._nadir_looking = True

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

        logger.info('')
        logger.info(name)
        frame_names0.append(name)
        meta_names0.append(meta_name)
        ground_truth0.append(gt)

        try:
            nf, *_ = mission.odo.process(img, datetime.fromtimestamp(mission.time0 + t), measure=meta)

            if nf is not None and nf.id is not None:
                kfid2img[nf.id] = i

                pts3d_w = np.array([pt.pt3d for pt in mission.odo.state.map3d.values() if pt.active])
                if len(pts3d_w) > 0:
                    pts3d_c = tools.q_times_mx(nf.pose.post.quat, pts3d_w) + nf.pose.post.loc
                    pts_d = np.quantile(pts3d_c[:, 2], (0.01, 0.5, 0.9))
                    logger.info('tot pts: %d, tree tops: %.1f m, median: %.1f m, ground: %.1f m' % (len(pts3d_w), *pts_d))

                if args.verbosity > 3:
                    post = np.zeros((len(mission.odo.state.keyframes), 7))
                    k, prior = 0, np.zeros((len(mission.odo.state.keyframes), 7))
                    for j, kf in enumerate([kf for kf in mission.odo.state.keyframes if kf.pose.post]):
                        post[j, :] = (-kf.pose.post).to_global(NokiaSensor.b2c).to_global(NokiaSensor.w2b).to_array()
                        if kf.measure is not None:
                            prior[k, :] = (-kf.pose.prior).to_global(NokiaSensor.b2c).to_global(NokiaSensor.w2b).to_array()
                            k += 1
                    if ax is not None:
                        ax.clear()
                    ax = tools.plot_poses(post, axis=(0, 1, 0), up=(0, 0, 1), ax=ax, wait=False)
                    tools.plot_poses(prior[:k, :], axis=(0, 1, 0), up=(0, 0, 1), ax=ax, wait=False,
                                     colors=map(lambda c: (c, 0, 0, 0.5), np.linspace(.3, 1.0, k)))
                    plt.pause(0.05)

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

        if 1:
            pts3d = np.array([pt.pt3d for pt in map3d])
            ground_alt = np.quantile(-pts3d[:, 1], 0.5)    # neg-y is altitude in cam frame
            drone_alt = -(-results[-1][0].post).loc[1]        # word
            expected_dist = drone_alt - mission.coord0[2]
            modeled_dist = drone_alt - ground_alt
            fl_used = (mission.cam.cam_mx[0, 0] + mission.cam.cam_mx[1, 1]) / 2
            logger.info('ground at %.1f m (%.1f m), drone alt %.1f m (%.1f m), est focal length: %.1f px' % (
                ground_alt, mission.coord0[2], modeled_dist, expected_dist, fl_used * expected_dist / modeled_dist
            ))

        if args.kapture:
            kapture = KaptureIO(args.kapture, reset=True, jpg_qlt=95, scale=0.5)
            kapture.set_camera(1, 'cam', mission.cam)
            kapture.add_frames(mission.odo.removed_keyframes, map3d)
            kapture.write_to_dir()
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

    logger.info('time spent: %.0fs' % (datetime.now() - started).total_seconds())

    repr_errs = np.concatenate([list(kf.repr_err.values()) for kf in mission.odo.removed_keyframes if len(kf.repr_err)])
    err_q95 = np.quantile(np.linalg.norm(repr_errs, axis=1), 0.95) if len(repr_errs) else np.nan
    logger.info('95%% percentile repr err: %.3fpx' % (err_q95,))

    interp = interp_loc(mission.odo.removed_keyframes, mission.time0)
    loc_est = np.array([np.ones((3,))*np.nan if kf.pose.post is None else (-kf.pose.post).loc for kf in mission.odo.removed_keyframes])
    loc_gps = np.array([interp(kf.time.timestamp() - mission.time0).flatten() for kf in mission.odo.removed_keyframes]).squeeze()
    mean_loc_err = np.nanmean(np.linalg.norm(loc_est - loc_gps, axis=1))
    logger.info('mean loc err: %.3fm' % (mean_loc_err,))

    if 0:
        plt.figure(10)
        plt.plot(loc_est[:, 0], loc_est[:, 1])
        plt.plot(loc_gps[:, 0], loc_gps[:, 1])
        plt.show()

    if args.verbosity > 3:
        plt.show()  # stop to show last trajectory plot

    file = args.res or ('%s-result.pickle' % args.mission)
    res_path = os.path.dirname(file)
    if res_path:
        os.makedirs(res_path, exist_ok=True)
    with open(file, 'wb') as fh:
        pickle.dump((results, map3d, frame_names, meta_names, ground_truth), fh)

    if args.verbosity > 1:
        plot_results(results, map3d, frame_names, meta_names, ground_truth, file, nadir_looking=args.nadir_looking)


def plot_results(results=None, map3d=None, frame_names=None, meta_names=None, ground_truth=None, file='result.pickle',
                 nadir_looking=False):

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    if results is None:
        with open(file, 'rb') as fh:
            results, map3d, frame_names, meta_names, ground_truth = pickle.load(fh)

    w2b, b2c = NokiaSensor.w2b, NokiaSensor.b2c
    nl_dq = tools.eul_to_q((-np.pi / 2,), 'y') if nadir_looking else quaternion.one

    est_loc = np.ones((len(results), 3)) * np.nan
    est_ori = np.ones((len(results), 3)) * np.nan
    meas_loc = np.ones((len(results), 3)) * np.nan
    meas_ori = np.ones((len(results), 3)) * np.nan
    for i, res in enumerate(results):
        if res and res[0] and res[0].post:
            if res[0].method == VisualOdometry.POSE_2D3D:
                est_loc[i, :] = ((-res[0].post).to_global(b2c).to_global(w2b)).loc
                meas_loc[i, :] = ((-res[0].prior).to_global(b2c).to_global(w2b)).loc
                est_ori[i, :] = tools.q_to_ypr(nl_dq.conj() * ((-res[0].post).to_global(b2c)).quat)
                meas_ori[i, :] = tools.q_to_ypr(nl_dq.conj() * ((-res[0].prior).to_global(b2c)).quat)

    est_ori = est_ori / np.pi * 180
    meas_ori = meas_ori / np.pi * 180

    if nadir_looking:
        # TODO: better way, now somehow works heuristically
        est_ori = est_ori[:, (2, 0, 1)]
        meas_ori = meas_ori[:, (2, 0, 1)]

    fst = np.where(np.logical_not(np.isnan(est_loc[:, 0])))[0][0]
    idx = np.where([r is not None for r in results])[0]
    idx2 = np.where([r is not None and r[1] is not None for r in results])[0]
    t0 = results[idx[0]][2].timestamp()

    t = np.array([results[i][2].timestamp() - t0 for i in idx])
    t2 = np.array([results[i][2].timestamp() - t0 + results[i][1].time_off + results[i][1].time_adj for i in idx2])
    dt = np.array([results[i][1].time_adj for i in idx2])

    rng = np.nanmax(est_loc[idx, :2], axis=0) - np.nanmin(est_loc[idx, :2], axis=0)
    rng2 = 0 if len(idx2) == 0 else np.nanmax(meas_loc[idx2, :2], axis=0) - np.nanmin(meas_loc[idx2, :2], axis=0)
    mrg = 0.05 * max(np.max(rng), np.max(rng2))
    min_x = min(np.nanmin(est_loc[idx, 0]), 9e9 if len(idx2) == 0 else np.nanmin(meas_loc[idx2, 0])) - mrg
    max_x = max(np.nanmax(est_loc[idx, 0]), -9e9 if len(idx2) == 0 else np.nanmax(meas_loc[idx2, 0])) + mrg
    min_y = min(np.nanmin(est_loc[idx, 1]), 9e9 if len(idx2) == 0 else np.nanmin(meas_loc[idx2, 1])) - mrg
    max_y = max(np.nanmax(est_loc[idx, 1]), -9e9 if len(idx2) == 0 else np.nanmax(meas_loc[idx2, 1])) + mrg
    min_z = min(np.nanmin(est_loc[idx, 2]), 9e9 if len(idx2) == 0 else np.nanmin(meas_loc[idx2, 2])) - mrg
    max_z = max(np.nanmax(est_loc[idx, 2]), -9e9 if len(idx2) == 0 else np.nanmax(meas_loc[idx2, 2])) + mrg

    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 10), np.linspace(min_y, max_y, 10))
    zz = np.ones(xx.shape) * min(min_z + mrg/2, 0)
    min_z = min(0, min_z)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz, zorder=2)
    line = ax.plot(est_loc[idx, 0], est_loc[idx, 1], est_loc[idx, 2], 'C0', zorder=3)
    ax.scatter(est_loc[fst, 0], est_loc[fst, 1], est_loc[fst, 2], 'C2', zorder=4)
    line2 = ax.plot(meas_loc[idx2, 0], meas_loc[idx2, 1], meas_loc[idx2, 2], 'C1', zorder=5)

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
            axs[0].plot(est_loc[fst, 0], est_loc[fst, 1], 'oC0', mfc='none')
            line = axs[0].plot(est_loc[idx, 0], est_loc[idx, 1], 'C0')  # , '+-')
            line2 = axs[0].plot(meas_loc[idx2, 0], meas_loc[idx2, 1], 'C1')  # , '+-')
            axs[0].set_xlim(min_x, max_x)
            axs[0].set_ylim(min_y, max_y)
            axs[0].set_xlabel('x')
            axs[0].set_ylabel('y')
        else:
            axs[0].plot(est_loc[fst, 1], est_loc[fst, 0], 'oC0', mfc='none')
            line = axs[0].plot(est_loc[idx, 1], est_loc[idx, 0], 'C0')  # , '+-')
            line2 = axs[0].plot(meas_loc[idx2, 1], meas_loc[idx2, 0], 'C1')  # , '+-')
            axs[0].set_xlim(min_y, max_y)
            axs[0].set_ylim(min_x, max_x)
            axs[0].set_xlabel('y')
            axs[0].set_ylabel('x')

        tools.hover_annotate(fig, axs[0], line[0], frame_names)
        tools.hover_annotate(fig, axs[0], line2[0], [meta_names[i] for i in idx2])

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(t[fst], est_loc[fst, 2], 'oC0', mfc='none')
        line = axs[0].plot(t, est_loc[idx, 2], 'C0')  # , '+-')
        line2 = axs[0].plot(t2, meas_loc[idx2, 2], 'C1')  # , '+-')
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
            axs[i].plot(t[fst], est_ori[fst, i], 'oC0', mfc='none')
            line = axs[i].plot(t, est_ori[idx, i], 'C0')  # , '+-')
            line2 = axs[i].plot(t2, meas_ori[idx2, i], 'C1')  # , '+-')
            tools.hover_annotate(fig, axs[i], line[0], frame_names)
            tools.hover_annotate(fig, axs[i], line2[0], [meta_names[j] for j in idx2])
            axs[i].set_ylabel(title)
            axs[i].set_xlabel('t')

    plt.tight_layout()

#    tools.plot_poses(pose[idx, :], axis=(0, 1, 0), up=(0, 0, 1))

    plt.show()
    print('ok: %.1f%%, delta loc std: %.3e' % (
        100*(1 - np.mean(np.isnan(est_loc[:, 0]))),
        np.nanstd(np.linalg.norm(np.diff(est_loc[:, :3], axis=0), axis=1)),
    ))


if __name__ == '__main__':
    if 1:
        main()
    elif 0:
        plot_results(file='hwproto-result.pickle')
    else:
        plot_results(file='nokia-result.pickle')
