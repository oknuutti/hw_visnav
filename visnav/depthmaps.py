import os
import argparse

import numpy as np

from kapture.io.records import get_record_fullpath
from kapture.converter.colmap.export_colmap import export_colmap
from kapture_localization.colmap.colmap_command import run_image_undistorter, run_colmap_command


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
    parser.add_argument('--skip-import', action='store_true',
                        help='skip importing kapture to colmap format')
    parser.add_argument('--skip-depth-est', action='store_true',
                        help='skip depth map estimation')

    args = parser.parse_args()
    txt_rec = os.path.join(args.path, args.txt)
    db_path = os.path.join(args.path, 'colmap.db')
    img_path = get_record_fullpath(args.kapture)
    dense_path = os.path.join(args.path, args.dense)
    os.makedirs(os.path.join(dense_path, 'images', args.sensor), exist_ok=True)

    if not args.skip_import:
        export_colmap(args.kapture, db_path, txt_rec, keypoints_type=args.keypoint, force_overwrite_existing=True)
        run_image_undistorter(args.cmd, img_path, txt_rec, dense_path)

        for f in ('consistency_graphs', 'depth_maps', 'normal_maps'):
            os.makedirs(os.path.join(dense_path, 'stereo', f, args.sensor), exist_ok=True)

    if not args.skip_depth_est:
        patch_match_stereo_args = ["patch_match_stereo", "--workspace_path", dense_path,
                                   "--PatchMatchStereo.depth_min", str(args.min_depth),
                                   "--PatchMatchStereo.depth_max", str(args.max_depth),
                                   "--PatchMatchStereo.window_radius", str(args.win_rad),
                                   "--PatchMatchStereo.window_step", str(args.win_step),
                                   "--PatchMatchStereo.gpu_index", args.gpu,
                                   "--PatchMatchStereo.cache_size", str(args.mem),
                                   ]
        if args.composite_cmd:
            extras = args.composite_cmd.split(' ')
            cmd, patch_match_stereo_args = extras[0], extras[1:] + patch_match_stereo_args
        else:
            cmd = args.cmd

        run_colmap_command(cmd, patch_match_stereo_args)


def read_colmap_array(path):
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


if __name__ == '__main__':
    main()
