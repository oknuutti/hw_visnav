import os
import argparse
import subprocess
import re
import logging

import numpy as np
import cv2
import tqdm

from kapture.io.csv import kapture_from_dir
from kapture.io.records import get_record_fullpath
from kapture.converter.colmap.export_colmap import export_colmap

from visnav.algo import tools
from visnav.algo.tools import Pose


def main():
    parser = argparse.ArgumentParser(description='Estimate depthmaps using COLMAP')
    parser.add_argument('-i', '--kapture', required=True,
                        help='input path to kapture data root directory')
    parser.add_argument('-s', '--sensor', default='cam',
                        help='input kapture image sensor name')
    parser.add_argument('-k', '--keypoint', default='gftt',
                        help='input kapture keypoint type name')
    parser.add_argument('-p', '--path', required=True,
                        help='output base path')
    parser.add_argument('-t', '--txt', default='txt',
                        help='output text folder name')
    parser.add_argument('-d', '--dense', default='dense',
                        help='output dense folder name')
    parser.add_argument('-e', '--export',
                        help='export depth maps here')
    parser.add_argument('-c', '--cmd',
                        help='path to colmap command')
    parser.add_argument('--composite-cmd',
                        help='composite cmd to invoke colmap command, e.g. "singularity exec colmap.sif colmap"')
    parser.add_argument('--gpu', default='0',
                        help='gpu indices to use, e.g. 0 (default) or 0,0,0,0 or 0,1,2')
    parser.add_argument('--mem', default=32, type=int,
                        help='max mem usage in GB ')
    parser.add_argument('--min-depth', type=float, default=10,
                        help='min depth for depth map estimation')
    parser.add_argument('--max-depth', type=float, default=200,
                        help='max depth for depth map estimation')
    parser.add_argument('--win-rad', type=int, default=5,
                        help='window radius for colmap depth map estimation (default=5)')
    parser.add_argument('--win-step', type=int, default=1,
                        help='window step size for colmap depth map estimation (default=1)')

    parser.add_argument('--filter-min-ncc', type=float, default=0.1,
                        help='--PatchMatchStereo.filter_min_ncc  arg (=0.1)')
    parser.add_argument('--filter-min-triangulation-angle', type=float, default=3.0,
                        help='--PatchMatchStereo.filter_min_triangulation_angle  arg (=3.0)')
    parser.add_argument('--filter-min-num-consistent', type=int, default=2,
                        help='--PatchMatchStereo.filter_min_num_consistent  arg (=2)')
    parser.add_argument('--filter-geom-consistency-max-cost', type=float, default=1.0,
                        help='--PatchMatchStereo.filter_geom_consistency_max_cost  arg (=1.0)')

    parser.add_argument('--skip-import', action='store_true',
                        help='skip importing kapture to colmap format')
    parser.add_argument('--skip-depth-est', action='store_true',
                        help='skip depth map estimation')
    parser.add_argument('--skip-export', action='store_true',
                        help='skip exporting depth maps to exr format')

    args = parser.parse_args()
    txt_rec = os.path.join(args.path, args.txt)
    db_path = os.path.join(args.path, 'colmap.db')
    img_path = get_record_fullpath(args.kapture)
    dense_path = os.path.join(args.path, args.dense)
    os.makedirs(os.path.join(dense_path, 'images', args.sensor), exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    if not args.export:
        args.export = os.path.join(args.kapture, 'reconstruction')

    if args.composite_cmd:
        cmd = args.composite_cmd.split(' ')
    else:
        assert args.cmd, 'either --cmd or --composite-cmd argument needs to be given'
        cmd = [args.cmd]

    if not args.skip_import:
        export_colmap(args.kapture, db_path, txt_rec, keypoints_type=args.keypoint, force_overwrite_existing=True)
        image_undistorter_args = ["image_undistorter",
                                  "--image_path", img_path,
                                  "--input_path", txt_rec,
                                  "--output_path", dense_path,
                                  "--blank_pixels", "1",
                                  ]
        exec_cmd(cmd + image_undistorter_args)

        for f in ('consistency_graphs', 'depth_maps', 'normal_maps'):
            os.makedirs(os.path.join(dense_path, 'stereo', f, args.sensor), exist_ok=True)

    if not args.skip_depth_est:
        patch_match_stereo_args = ["patch_match_stereo",
                                   "--workspace_path", dense_path,
                                   "--PatchMatchStereo.depth_min", str(args.min_depth),
                                   "--PatchMatchStereo.depth_max", str(args.max_depth),
                                   "--PatchMatchStereo.window_radius", str(args.win_rad),
                                   "--PatchMatchStereo.window_step", str(args.win_step),
                                   "--PatchMatchStereo.gpu_index", args.gpu,
                                   "--PatchMatchStereo.cache_size", str(args.mem),
                                   "--PatchMatchStereo.filter_min_ncc", str(args.filter_min_ncc),
                                   "--PatchMatchStereo.filter_min_triangulation_angle",
                                        str(args.filter_min_triangulation_angle),
                                   "--PatchMatchStereo.filter_min_num_consistent",
                                        str(args.filter_min_num_consistent),
                                   "--PatchMatchStereo.filter_geom_consistency_max_cost",
                                        str(args.filter_geom_consistency_max_cost),
                                   ]
        exec_cmd(cmd + patch_match_stereo_args)

    if not args.skip_export:
        depth_path = os.path.join(dense_path, 'stereo', 'depth_maps', args.sensor)
        os.makedirs(os.path.join(args.export, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(args.export, 'geometry'), exist_ok=True)
        kapt = kapture_from_dir(args.kapture)
        sensor_id, width, height, fl_x, fl_y, pp_x, pp_y, *dist_coefs = get_cam_params(kapt, args.sensor)
        file2id = {fn[sensor_id]: id for id, fn in kapt.records_camera.items()}

        exr_params_d = exr_params_xyz = (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT)
        if hasattr(cv2, 'IMWRITE_EXR_COMPRESSION'):
            # supported in OpenCV 4, see descriptions at
            #   https://rainboxlab.org/downloads/documents/EXR_Data_Compression.pdf
            exr_params_d += (cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24)    # zip 24bit floats
            exr_params_xyz += (cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP)    # zip 32bit floats

        logging.info('Exporting geometric depth maps in EXR format...')
        for fname in tqdm.tqdm(os.listdir(depth_path), mininterval=3):
            m = re.search(r'(.*?)(\.jpg|\.png|\.jpeg)?\.geometric\.bin', fname)
            if m:
                depth0 = read_colmap_array(os.path.join(depth_path, fname))
                depth0[depth0 <= args.min_depth] = np.nan
                depth0[depth0 >= args.max_depth] = np.nan
                depth = filter_depth(depth0, args)

                if width != depth.shape[1] or height != depth.shape[0]:
                    logging.warning('Depth map for image %s is different size than the camera resolution %s vs %s' % (
                        m[1] + m[2], depth.shape, (height, width)))

                outfile = os.path.join(args.export, 'depth', m[1] + '.d.exr')
                cv2.imwrite(outfile, depth, exr_params_d)

                frame_id = file2id.get(args.sensor + '/' + m[1] + m[2], file2id.get(args.sensor + '\\' + m[1] + m[2], None))
                cf_cam_world_v, cf_cam_world_q = kapt.trajectories[frame_id][sensor_id].t, kapt.trajectories[frame_id][sensor_id].r
                cf_world_cam = -Pose(cf_cam_world_v, cf_cam_world_q)

                px_u = cam_px_u(depth.shape[1], depth.shape[0], fl_x, fl_y, pp_x, pp_y)

                # the depth is actually the z-component, not the distance from the camera to the surface
                dist = depth.flatten()/px_u[:, 2]

                px_u = tools.q_times_mx(cf_world_cam.quat, px_u * dist[:, None])
                xyz = px_u.reshape(depth.shape + (3,)) + cf_world_cam.loc.reshape((1, 1, 3))

                outfile = os.path.join(args.export, 'geometry', m[1] + '.xyz.exr')
                cv2.imwrite(outfile, xyz.astype(np.float32), exr_params_xyz)

                if 0 and m[1] == 'frame000370':
                    _, depth2 = filter_depth(depth0, args, return_interm=True)
                    xyz = xyz.reshape((-1, 3))

                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D

                    plt.figure(1)
                    plt.imshow(depth0)

                    plt.figure(2)
                    plt.imshow(depth2)

                    plt.figure(3)
                    plt.imshow(depth)

                    f = plt.figure(4)
                    a = f.add_subplot(111, projection='3d')
                    a.set_xlabel('x')
                    a.set_ylabel('y')
                    a.set_zlabel('z')
                    a.plot(xyz[::5, 0], xyz[::5, 1], xyz[::5, 2], '.')

                    plt.show()

                    # mask = np.logical_not(np.isnan(depth)).astype(np.uint8)*255
                    # k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    # mask2 = cv2.erode(mask, k)


def filter_depth(depth, args, return_interm=False):
    lim_val = (args.max_depth - args.min_depth) * 0.1
    depth2 = filter_outliers(depth, (15, 15), (7, 7), lim_val, 0.2, 0.7)
    depth3 = nan_blur(depth2, krn=(3, 3), lim=3, onlynans=True)
    depth3 = remove_borders(depth3, margin=args.win_rad)
    if return_interm:
        return depth3, depth2
    return depth3


def filter_outliers(arr, krn_bg, krn_nan, lim_val, lim_nan, max_nan, use_and=True):
    bg_val = nan_blur(arr, krn_bg, 0)
    nans = nan_blur(np.isnan(arr).astype(float), krn_nan, 0)
    if use_and:
        is_outlier = np.logical_and(np.abs(arr - bg_val) > lim_val, nans > lim_nan)
        is_outlier = np.logical_or(is_outlier, nans > max_nan)
    else:
        is_outlier = np.abs(arr - bg_val) * nans > lim_val * lim_nan
    is_outlier[np.isnan(is_outlier)] = False
    arr2 = arr.copy()
    arr2[is_outlier] = np.nan
    return arr2


def nan_blur(arr, krn=(3, 3), lim=3, onlynans=False):
    arr2 = arr.copy()
    arr2[np.isnan(arr)] = 0
    filt_arr = cv2.blur(arr2, krn)
    filt_mask = cv2.blur(np.logical_not(np.isnan(arr)).astype(np.float32), krn)
    filt_ok = filt_mask > lim / np.prod(krn)
    if onlynans:
        filt_ok = np.logical_and(filt_ok, np.isnan(arr))
    arr2[filt_ok] = filt_arr[filt_ok] / filt_mask[filt_ok]
    arr2[filt_mask <= lim / np.prod(krn)] = np.nan
    return arr2


def remove_borders(arr, margin):
    arr2 = arr.copy()
    arr2[:margin, :] = np.nan
    arr2[-margin:, :] = np.nan
    arr2[:, :margin] = np.nan
    arr2[:, -margin:] = np.nan
    return arr2


def get_cam_params(kapt, sensor_name):
    sid, sensor = None, None
    for id, s in kapt.sensors.items():
        if s.name == sensor_name:
            sid, sensor = id, s
            break
    sp = sensor.sensor_params
    return (sid,) + tuple(map(int, sp[1:3])) + tuple(map(float, sp[3:]))


def unit_aflow(W, H):
    return np.stack(np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)), axis=2)


def cam_px_u(width, height, fl_x, fl_y, pp_x, pp_y):
    xy = unit_aflow(width, height).reshape((-1, 2))
    u = tools.normalize_mx(np.stack((
        (xy[:, 0] - pp_x) / fl_x,
        (xy[:, 1] - pp_y) / fl_y,
        np.ones((xy.shape[0],), dtype=np.float32)
    ), axis=1))
    return u


def read_colmap_array(path):
    # from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_dense.py
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def exec_cmd(args):
    proc = subprocess.Popen(args)
    proc.wait()

    if proc.returncode != 0:
        raise ValueError(
            '\nSubprocess Error (Return code:'
            f' {proc.returncode} )')


if __name__ == '__main__':
    main()
