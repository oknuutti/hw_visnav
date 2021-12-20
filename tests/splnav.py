from datetime import datetime
import os
import math
import logging
import warnings

from visnav.algo.tools import PositioningException

warnings.filterwarnings("ignore", module='quaternion', lineno=21)

import numpy as np
import cv2
import sys

from visnav.render.render import RenderEngine
from visnav.algo import tools
from visnav.algo.keypoint import KeypointAlgo
from visnav.missions.rosetta import RosettaSystemModel

logger = logging.getLogger(__name__)


def main(outfile='spl-test.log', img_path='67p-imgs', config=None):
    logger.info('Setting up renderer and loading the 3d model...')
    tconfig = config or {}
    config = {'verbose': 1, 'feat': KeypointAlgo.AKAZE, 'v16k': False, 'view_width': 384}
    config.update(tconfig)
    v16k = config.pop('v16k')
    view_width = config.pop('view_width')

    sm = RosettaSystemModel(hi_res_shape_model=False, skip_obj_load=True, res_mult=512/1024, view_width=view_width)

    re = RenderEngine(sm.view_width, sm.view_height, antialias_samples=0)
    re.set_frustum(sm.cam.x_fov, sm.cam.y_fov, 0.1 * sm.min_distance, 1.1 * sm.max_distance)
    obj_path = os.path.join(os.path.dirname(__file__), 'data', '67p-16k.obj' if v16k else '67p-4k.obj')
    obj_idx = re.load_object(obj_path)

    spl = KeypointAlgo(sm, re, obj_idx)
    spl.RENDER_TEXTURES = False

    img_path = os.path.join(os.path.dirname(__file__), 'data', img_path)
    imgfiles = sorted([fname for fname in os.listdir(img_path) if fname[-4:].lower() in ('.jpg', '.png')])

    with open(outfile, 'w') as file:
        file.write(' '.join(sys.argv) + '\n' + '\t'.join(log_columns()) + '\n')

    # loop through some test images
    for i, fname in enumerate(imgfiles):
        imgfile = os.path.join(img_path, fname)

        lblfile = imgfile[:-4] + '.lbl'
        if not os.path.exists(lblfile):
            lblfile = imgfile[:-6] + '.lbl'

        # load a noisy system state
        sm.load_state(lblfile, sc_ast_vertices=False)
        initial = {
            'time': sm.time.value,
            'ast_axis': sm.asteroid_axis,
            'sc_rot': sm.spacecraft_rot,
            'sc_pos': sm.spacecraft_pos,
            'ast_pos': sm.asteroid.real_position,
        }

        # get result and log result stats
        try:
            spl.solve_pnp(imgfile, None, scale_cam_img=True, **config)
            ok = True
        except PositioningException as e:
            ok = False
        rtime = spl.timer.elapsed

        # calculate results
        results = calculate_result(sm, spl, fname, ok, initial)

        # write log entry
        write_log_entry(outfile, i, rtime, 0.0, *results)


def calculate_result(sm, spl, fname, ok, initial):
    # save function values from optimization
    fvals = getattr(spl, 'extra_values', None)
    final_fval = fvals[-1] if fvals else None

    real_rel_rot = tools.q_to_lat_lon_roll(sm.real_sc_asteroid_rel_q())
    elong, direc = sm.solar_elongation(real=True)
    r_ast_axis = sm.real_asteroid_axis

    # real system state
    params = (sm.time.real_value, *r_ast_axis,
              *sm.real_spacecraft_rot, math.degrees(elong), math.degrees(direc),
              *sm.real_spacecraft_pos, sm.real_spacecraft_altitude, *map(math.degrees, real_rel_rot),
              fname, final_fval)

    # calculate added noise
    #
    time_noise = initial['time'] - sm.time.real_value

    ast_rot_noise = (
        initial['ast_axis'][0] - r_ast_axis[0],
        initial['ast_axis'][1] - r_ast_axis[1],
        360 * time_noise / sm.asteroid.rotation_period
        + (initial['ast_axis'][2] - r_ast_axis[2])
    )
    sc_rot_noise = tuple(np.subtract(initial['sc_rot'], sm.real_spacecraft_rot))

    dev_angle = math.degrees(tools.angle_between_lat_lon_roll(map(math.radians, ast_rot_noise),
                                                              map(math.radians, sc_rot_noise)))

    sc_loc_noise = ('', '', '')
    noise = sc_loc_noise + (time_noise,) + ast_rot_noise + sc_rot_noise + (dev_angle,)

    if np.all(ok):
        ok_pos, ok_rot = True, True
    elif not np.any(ok):
        ok_pos, ok_rot = False, False
    else:
        ok_pos, ok_rot = ok

    if ok_pos:
        pos = sm.spacecraft_pos
        pos_err = tuple(np.subtract(pos, sm.real_spacecraft_pos))
    else:
        pos = float('nan') * np.ones(3)
        pos_err = tuple(float('nan') * np.ones(3))

    if ok_rot:
        rel_rot = tools.q_to_lat_lon_roll(sm.sc_asteroid_rel_q())
        rot_err = (math.degrees(tools.wrap_rads(tools.angle_between_lat_lon_roll(rel_rot, real_rel_rot))),)
    else:
        rel_rot = float('nan') * np.ones(3)
        rot_err = (float('nan'),)

    alt = float('nan')
    if ok_pos and ok_rot:
        est_vertices = sm.sc_asteroid_vertices()
        max_shift = float('nan') if est_vertices is None else \
            tools.sc_asteroid_max_shift_error(est_vertices, sm.asteroid.real_sc_ast_vertices)
        alt = sm.spacecraft_altitude or float('nan')
        both_err = (max_shift, alt - (sm.real_spacecraft_altitude or float('nan')))
    else:
        both_err = (float('nan'), float('nan'),)

    err = pos_err + rot_err + both_err
    return params, noise, pos, alt, map(math.degrees, rel_rot), fvals, err


def log_columns():
    return (
        'iter', 'date', 'execution time',
        'time', 'ast lat', 'ast lon', 'ast rot',
        'sc lat', 'sc lon', 'sc rot',
        'sol elong', 'light dir', 'x sc pos', 'y sc pos', 'z sc pos', 'sc altitude',
        'rel yaw', 'rel pitch', 'rel roll',
        'imgfile', 'extra val', 'shape model noise',
        'sc pos x dev', 'sc pos y dev', 'sc pos z dev',
        'time dev', 'ast lat dev', 'ast lon dev', 'ast rot dev',
        'sc lat dev', 'sc lon dev', 'sc rot dev', 'total dev angle',
        'x est sc pos', 'y est sc pos', 'z est sc pos', 'altitude est sc',
        'yaw rel est', 'pitch rel est', 'roll rel est',
        'x err sc pos', 'y err sc pos', 'z err sc pos', 'rot error',
        'shift error km', 'altitude error', 'lat error (mr/km)', 'dist error (mr/km)', 'rel shift error (mr/km)',
    )


def write_log_entry(logfile, i, rtime, sm_noise, params, noise, pos, alt, rel_rot, fvals, err):

    # # save execution time
    # self.run_times.append(rtime)

    # calculate errors
    dist = abs(params[-7])
    if not math.isnan(err[0]):
        lerr = 1000*math.sqrt(err[0]**2 + err[1]**2) / dist     # mr/km
        derr = 1000*err[2] / dist                               # mr/km
        rerr = abs(err[3])                                      # deg
        serr = 1000*err[4] / dist                               # mr/km
        fail = 0
    else:
        lerr = derr = rerr = serr = float('nan')
        fail = 1
    # self.laterrs.append(lerr)
    # self.disterrs.append(abs(derr))
    # self.roterrs.append(rerr)
    # self.shifterrs.append(serr)
    # self.fails.append(fail)

    # log all parameter values, timing & errors into a file
    with open(logfile, 'a') as file:
        file.write('\t'.join(map(str, (
            i, datetime.now().strftime("%Y-%mr-%d %H:%M:%S"), rtime, *params,
            sm_noise, *noise, *pos, alt, *rel_rot, *err, lerr, derr, serr
        )))+'\n')

    # log opt fun values in other file
    if fvals:
        with open('fval-'+logfile, 'a') as file:
            file.write(str(i)+'\t'+'\t'.join(map(str, fvals))+'\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    config = [
        ('spl-orb-4k-384.log', {
            'feat': KeypointAlgo.ORB,
            'v16k': False,
            'view_width': 384,
        }),
        ('spl-akaze-4k-384.log', {
            'feat': KeypointAlgo.AKAZE,
            'v16k': False,
            'view_width': 384,
        }),
        ('spl-orb-4k-512.log', {
            'feat': KeypointAlgo.ORB,
            'v16k': False,
            'view_width': 512,
        }),
        ('spl-akaze-4k-512.log', {
            'feat': KeypointAlgo.AKAZE,
            'v16k': False,
            'view_width': 512,
        }),
        ('spl-orb-16k-512.log', {
            'feat': KeypointAlgo.ORB,
            'v16k': True,
            'view_width': 512,
        }),
        ('spl-akaze-16k-512.log', {
            'feat': KeypointAlgo.AKAZE,
            'v16k': True,
            'view_width': 512,
        }),
    ]

    test_ids = list(map(str, range(1, len(config) + 1)))
    if len(sys.argv) != 2 or sys.argv[1] not in test_ids:
        logger.error('USAGE: %s %s' % (sys.argv[0], '|'.join(test_ids)))

    test_id = int(sys.argv[1]) - 1
    main(config[test_id][0], config=config[test_id][1])
