import argparse
import pickle
from datetime import datetime
import os
import math
from typing import List
import re

import numpy as np
import quaternion

from visnav.algo import tools
from visnav.algo.model import Camera
from visnav.algo.tools import Pose
from visnav.algo.odo.base import VisualOdometry, Frame, Keypoint
from visnav.iotools.kapture import KaptureIO
from visnav.missions.hwproto import HardwarePrototype
from visnav.missions.nokia import NokiaSensor, interp_loc
from visnav.missions.toynokia import ToyNokiaSensor

logger = tools.get_logger("main")


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Run visual odometry on a set of images')
    parser.add_argument('--data', '-d', metavar='DATA', help='path to data')
    parser.add_argument('--meta', '-t', metavar='META', help='path to meta data')
    parser.add_argument('--video-toff', '--dt', type=float, metavar='dT', help='video time offset compared to metadata')
    parser.add_argument('--alt-scale', '--as', type=float, default=1.0, metavar='sc', help='scale telemetry altitude with this value')
    parser.add_argument('--res', '-r', metavar='OUT', help='path to the result pickle')
    parser.add_argument('--debug-out', '-o', metavar='OUT', help='path to the debug output folder')
    parser.add_argument('--kapture', metavar='OUT', help='path to kapture-format export folder')
    parser.add_argument('--verbosity', '-v', type=int, default=2,
                        help='verbosity level (0-4, 0-1: text only, 2:+debug imgs, 3: +keypoints, 4: +poses)')
    parser.add_argument('--high-quality', action='store_true', help='high quality settings with more keypoints detected')

    parser.add_argument('--mission', '-mr', choices=('hwproto', 'nokia', 'toynokia'), help='select mission')
    parser.add_argument('--undist-img', action='store_true', help='undistort image instead of keypoints')
    parser.add_argument('--use-gimbal', action='store_true', help='gimbal data is ok, use it')
    parser.add_argument('--drifting-gimbal', action='store_true',
                        help='gimbal orientation measure drifts, update orientation offset with each odometry result')
    parser.add_argument('--nadir-looking', action='store_true', help='downwards looking cam')

    parser.add_argument('--ori-off-ypr', type=float, nargs=3,
                        help='orientation offset in ypr (deg) to be applied in body frame to orientation measurements')

    parser.add_argument('--cam-dist', type=float, nargs='*', help='cam distortion coeffs')
    parser.add_argument('--cam-fl-x', type=float, help='cam focal length x')
    parser.add_argument('--cam-fl-y', type=float, help='cam focal length y')
    parser.add_argument('--cam-pp-x', type=float, help='cam principal point x')
    parser.add_argument('--cam-pp-y', type=float, help='cam principal point y')

    parser.add_argument('--first-frame', '-f', type=int, default=0, help='first frame (default: 0; -1: hardcoded value)')
    parser.add_argument('--last-frame', '-l', type=int, help='last frame (default: None; -1: hardcoded end)')
    parser.add_argument('--skip', '-s', type=int, default=1, help='use only every xth frame (default: 1)')
    parser.add_argument('--plot-only', action='store_true', help='only plot the result')
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
                              verbosity=args.verbosity, high_quality=args.high_quality, alt_scale=args.alt_scale,
                              ori_off_q=tools.ypr_to_q(*map(math.radians, args.ori_off_ypr)) if args.ori_off_ypr else None,
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


    res_file = args.res or ('%s-result.pickle' % args.mission)
    if args.plot_only:
        plot_results(file=res_file, nadir_looking=args.nadir_looking)
        replay_keyframes(mission.odo.cam, file=res_file)
        return

    # run odometry
    prior = Pose(np.array([0, 0, 0]), quaternion.one)
    frame_names0 = []
    meta_names0 = []
    results0 = []
    ground_truth0 = []
    ori_offs = []
    ba_errs = []
    kfid2img = {}
    started = datetime.now()
    ax = None

    def ba_err_logger(frame_id, per_frame_ba_errs):
        per_frame_ba_errs = np.stack((per_frame_ba_errs[:, 0],
                                      np.linalg.norm(per_frame_ba_errs[:, 1:4], axis=1),
                                      np.linalg.norm(per_frame_ba_errs[:, 4:7], axis=1) / np.pi * 180), axis=1)
        ba_errs.append([frame_id, *np.nanmean(per_frame_ba_errs, axis=0)])
    mission.odo.ba_err_logger = ba_err_logger
    vid_id = re.search(r'(^|\\|/)([\w-]+)\.mp4$', mission.video_path)

    for i, (img, t, name, meta, meta_name, gt) in enumerate(mission.data):
        if i % args.skip != 0:
            continue

        logger.info('')
        logger.info(name)
        frame_names0.append(name)
        meta_names0.append(meta_name)
        ground_truth0.append(gt)

        if vid_id[2] == 'HD_CAM-1__514659341_03_12_2020_16_12_44' and t > 1050:
            # hardcoding just to get dataset 10 to work
            mission.odo.ori_off_q = tools.ypr_to_q(math.radians(0), 0, 0)

        try:
            nf, *_ = mission.odo.process(img, datetime.fromtimestamp(mission.time0 + t), measure=meta)

            if nf is not None and nf.id is not None:
                kfid2img[nf.id] = i

                if args.drifting_gimbal and nf.pose.post and meta:
                    est_w2c_bf_q = (-nf.pose.post).to_global(mission.b2c).quat
                    meas_w2c_bf_q = (-nf.pose.prior).to_global(mission.b2c).quat * mission.odo.ori_off_q.conj()
                    new_ori_off_q = meas_w2c_bf_q.conj() * est_w2c_bf_q
                    filtered_ori_off_q = tools.mean_q([mission.odo.ori_off_q, new_ori_off_q], [0.8, 0.2])
                    if 0:
                        # TODO: debug, something wrong here as roll and pitch explode
                        mission.odo.ori_off_q = filtered_ori_off_q
                    else:
                        y, p, r = tools.q_to_ypr(filtered_ori_off_q)
                        mission.odo.ori_off_q = tools.ypr_to_q(y, 0, 0)
                    ori_offs.append(mission.odo.ori_off_q)
                    logger.info('new gimbal offset: %s' % (
                        list(map(lambda x: round(math.degrees(x), 2), tools.q_to_ypr(mission.odo.ori_off_q))),))

                pts3d = np.array([pt.pt3d for pt in mission.odo.state.map3d.values() if pt.active])
                if len(pts3d) > 0 and nf.pose.post:
                    ground_alt = np.quantile(-pts3d[:, 1], 0.2)  # neg-y is altitude in cam frame
                    drone_alt = -(-nf.pose.post).loc[1]
                    expected_dist = drone_alt - mission.coord0[2]
                    modeled_dist = drone_alt - ground_alt
                    logger.info('ground at %.1f mr (%.1f mr), drone alt %.1f mr (%.1f mr)' % (
                        ground_alt, mission.coord0[2], modeled_dist, expected_dist
                    ))

                if args.verbosity > 3:
                    keyframes = mission.odo.all_keyframes()
                    post = np.zeros((len(keyframes), 7))
                    k, prior = 0, np.zeros((len(keyframes), 7))
                    for j, kf in enumerate([kf for kf in keyframes if kf.pose.post]):
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
#        results = [(kf.pose, kf.measure, kf.time, kf.id) for kf in mission.odo.removed_keyframes if kf.pose.post]
        keyframes = [dict(pose=kf.pose, meas=kf.measure, time=kf.time, id=kf.id, kps_uv=kf.kps_uv)
                     for kf in mission.odo.removed_keyframes if kf.pose.post]
        frame_names = [frame_names0[kfid2img[kf['id']]] for kf in keyframes]
        meta_names = [meta_names0[kfid2img[kf['id']]] for kf in keyframes]
        ground_truth = [ground_truth0[kfid2img[kf['id']]] for kf in keyframes]
        ba_errs = np.array(ba_errs)

        if 1:
            pts3d = np.array([pt.pt3d for pt in map3d])
            ground_alt = np.quantile(-pts3d[:, 1], 0.5)    # neg-y is altitude in cam frame
            drone_alt = -(-keyframes[-1]['pose'].post).loc[1]        # word
            expected_dist = drone_alt - mission.coord0[2]
            modeled_dist = drone_alt - ground_alt
            logger.info('ground at %.1f m (%.1f m), drone alt %.1f m (%.1f m)' % (
                ground_alt, mission.coord0[2], modeled_dist, expected_dist
            ))

        if args.kapture:
            kapture = KaptureIO(args.kapture, reset=True, jpg_qlt=95, scale=0.5)
            kapture.set_camera(1, 'cam', mission.cam)
            kapture.add_frames(mission.odo.removed_keyframes, map3d)
            kapture.write_to_dir()
    except AttributeError as e:
        if 0:
            map3d = None
            results = [(kf.pose, None, kf.time, kf.id) for kf in mission.odo.all_keyframes()]
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
    logger.info('mean ba errs (repr, loc, ori): %.3fpx %.3fm %.3fdeg' % (*np.nanmean(ba_errs[:, 1:], axis=0),))
    logger.info('latest ba errs (repr, loc, ori): %.3fpx %.3fm %.3fdeg' % (*ba_errs[-1, 1:],))

    if 0:
        plt.figure(10)
        plt.plot(loc_est[:, 0], loc_est[:, 1])
        plt.plot(loc_gps[:, 0], loc_gps[:, 1])
        plt.show()

    if args.verbosity > 3:
        plt.show()  # stop to show last trajectory plot

    res_folder = os.path.dirname(res_file)
    if res_folder:
        os.makedirs(res_folder, exist_ok=True)
    with open(res_file, 'wb') as fh:
        pickle.dump((keyframes, map3d, frame_names, meta_names, ground_truth, ba_errs), fh)

    if args.verbosity > 1:
        plot_results(keyframes, map3d, frame_names, meta_names, ground_truth, ba_errs, res_file, nadir_looking=args.nadir_looking)


def plot_results(keyframes=None, map3d=None, frame_names=None, meta_names=None, ground_truth=None, ba_errs=None,
                 file='result.pickle', nadir_looking=False):

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # TODO: remove override
    nadir_looking = False

    if keyframes is None:
        with open(file, 'rb') as fh:
            keyframes, map3d, frame_names, meta_names, ground_truth, *ba_errs = pickle.load(fh)
            ba_errs = ba_errs[0] if len(ba_errs) else None

    w2b, b2c = NokiaSensor.w2b, NokiaSensor.b2c
    nl_dq = tools.eul_to_q((-np.pi / 2,), 'y') if nadir_looking else quaternion.one

    est_loc = np.ones((len(keyframes), 3)) * np.nan
    est_ori = np.ones((len(keyframes), 3)) * np.nan
    meas_loc = np.ones((len(keyframes), 3)) * np.nan
    meas_ori = np.ones((len(keyframes), 3)) * np.nan
    for i, kf in enumerate(keyframes):
        if kf and kf['pose'] and kf['pose'].post:
            if kf['pose'].method == VisualOdometry.POSE_2D3D:
                est_loc[i, :] = ((-kf['pose'].post).to_global(b2c).to_global(w2b)).loc
                meas_loc[i, :] = ((-kf['pose'].prior).to_global(b2c).to_global(w2b)).loc
                est_ori[i, :] = tools.q_to_ypr(nl_dq.conj() * ((-kf['pose'].post).to_global(b2c)).quat)
                meas_ori[i, :] = tools.q_to_ypr(nl_dq.conj() * ((-kf['pose'].prior).to_global(b2c)).quat)

    est_ori = est_ori / np.pi * 180
    meas_ori = meas_ori / np.pi * 180

    if nadir_looking:
        # TODO: better way, now somehow works heuristically
        est_ori = est_ori[:, (2, 0, 1)]
        meas_ori = meas_ori[:, (2, 0, 1)]

    fst = np.where(np.logical_not(np.isnan(est_loc[:, 0])))[0][0]
    idx = np.where([kf['pose'] is not None for kf in keyframes])[0]
    idx2 = np.where([kf['pose'] is not None and kf['meas'] is not None for kf in keyframes])[0]
    t0 = keyframes[idx[0]]['time'].timestamp()

    t = np.array([keyframes[i]['time'].timestamp() - t0 for i in idx])
    t2 = np.array([keyframes[i]['time'].timestamp() - t0 + keyframes[i]['meas'].time_off + keyframes[i]['meas'].time_adj for i in idx2])
    dt = np.array([keyframes[i]['meas'].time_adj for i in idx2])

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

        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(t[fst], est_loc[fst, 2], 'oC0', mfc='none')
        line = axs[0].plot(t, est_loc[idx, 2], 'C0')  # , '+-')
        line2 = axs[0].plot(t2, meas_loc[idx2, 2], 'C1')  # , '+-')
        tools.hover_annotate(fig, axs[0], line[0], frame_names)
        tools.hover_annotate(fig, axs[0], line2[0], [meta_names[i] for i in idx2])
        axs[0].set_ylabel('z')
        axs[0].set_xlabel('t')

        if ba_errs is None:
            line = axs[1].plot(t2, dt, 'C0')  #, '+-')
            tools.hover_annotate(fig, axs[1], line[0], [meta_names[i] for i in idx2])
            axs[1].set_ylabel('dt')
        else:
            id2idx = {kf['id']: i for i, kf in enumerate(keyframes)}
            t3 = [keyframes[id2idx[int(id)]]['time'].timestamp() - t0 for id in ba_errs[:, 0]]

            axs[1].plot(t3, ba_errs[:, 1], 'C0', label='repr [px]')
            axs[1].plot(t3, ba_errs[:, 2], 'C1', label='loc [m]')
            axs[1].plot(t3, ba_errs[:, 3], 'C2', label='ori [deg]')
            axs[1].set_title('BA errors')
            axs[1].legend()
        axs[1].set_xlabel('t')

        fig, axs = plt.subplots(3, 1, sharex=True)
        for i, title in enumerate(('yaw', 'pitch', 'roll')):
            axs[i].plot(t[fst], est_ori[fst, i], 'oC0', mfc='none')
            line = axs[i].plot(t, est_ori[idx, i], 'C0')  # , '+-')
            line2 = axs[i].plot(t2, meas_ori[idx2, i], 'C1')  # , '+-')
            tools.hover_annotate(fig, axs[i], line[0], frame_names)
            tools.hover_annotate(fig, axs[i], line2[0], [meta_names[j] for j in idx2])
            axs[i].set_ylabel(title)
            axs[i].set_xlabel('t')

    plt.tight_layout()

    if map3d is not None and len(map3d) > 0:
        x, y, z = np.array([tools.q_times_v(w2b.quat * b2c.quat, pt.pt3d) for pt in map3d]).T

        if 1:
            fig, ax = plt.subplots(1, 1)
            ax.set_aspect('equal')
            ax.set_xlabel("east", fontsize=12)
            ax.set_ylabel("north", fontsize=12)
            line = ax.scatter(x, y, s=20, c=z, marker='o', vmin=-5., vmax=100., cmap=cm.get_cmap('jet'))  #, vmax=20.)
            fig.colorbar(line)
            tools.hover_annotate(fig, ax, line, ['%.1f' % v for v in z])
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(x, y, z, '.')
            ax.set_xlabel("east", fontsize=12)
            ax.set_ylabel("north", fontsize=12)
            ax.set_zlabel("alt", fontsize=12)

    plt.show()
    print('ok: %.1f%%, delta loc std: %.3e' % (
        100*(1 - np.mean(np.isnan(est_loc[:, 0]))),
        np.nanstd(np.linalg.norm(np.diff(est_loc[:, :3], axis=0), axis=1)),
    ))


def replay_keyframes(cam: Camera, keyframes: List[Frame] = None, map3d: List[Keypoint] = None, file: str = 'results.pickle'):
    import cv2

    if keyframes is None:
        with open(file, 'rb') as fh:
            keyframes, map3d, frame_names, meta_names, ground_truth, *ba_errs = pickle.load(fh)
            ba_errs = ba_errs[0] if len(ba_errs) else None

    if isinstance(keyframes[0], Frame):
        keyframes = [dict(pose=kf.pose, meas=kf.measure, time=kf.time, id=kf.id, kps_uv=kf.kps_uv, image=kf.image)
                     for kf in keyframes]

    kp_size, kp_color = 5, (200, 0, 0)
    kp_ids = set(kf.id for kf in map3d if not kf.bad_qlt and kf.inlier_count > 2 and kf.inlier_count / kf.total_count > 0.2)
    map3d = {kf.id: kf for kf in map3d}
    img_scale = 0.5

    for kf in keyframes:
        if kf['pose'] is None or kf['pose'].post is None:
            continue

        obs_ids = list(kp_ids.intersection(kf['kps_uv'].keys()))
        if len(obs_ids) == 0:
            continue

        image = kf['image'].copy() if 'image' in kf else np.zeros((int(cam.height*img_scale), int(cam.width*img_scale), 3), dtype=np.uint8)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        p_pts2d = (np.array([kf['kps_uv'][id] for id in obs_ids]) + 0.5).astype(int).squeeze()
        p_pts3d = np.array([map3d[id].pt3d for id in obs_ids])

        pts3d_cf = tools.q_times_mx(kf['pose'].post.quat, p_pts3d) + kf['pose'].post.loc
        pts2d_proj = (cam.project(pts3d_cf.astype(np.float32)) * img_scale + 0.5).astype(int)

        for (x, y), (xp, yp) in zip(p_pts2d, pts2d_proj):
            image = cv2.circle(image, (x, y), kp_size, kp_color, 1)   # negative thickness => filled circle
            image = cv2.rectangle(image, (xp-2, yp-2), (xp+2, yp+2), kp_color, 1)
            image = cv2.line(image, (xp, yp), (x, y), kp_color, 1)

        cv2.imshow('keypoint reprojection', image)
        cv2.waitKey()


if __name__ == '__main__':
    main()
