import os
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

from visnav.algo import tools


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Calibrate camera an image taken of a checker board')
    parser.add_argument('--path', '-f', action='append', help='path to the calibration images / video(s)')
    parser.add_argument('--skip', '-i', type=int, default=1, help='frame interval (1=no skipping, 2=use every other frame, etc)')
    parser.add_argument('--offset', '-o', type=int, default=0, help='skip these many images from the start')
    parser.add_argument('--preproc', type=int, default=0, help='rescale image brightness before processing')
    parser.add_argument('--rot', type=int, choices=[0, 90, -90, 180], default=0, help='rotate images with this many degrees')
    parser.add_argument('--nx', '-x', type=int, help='checker board corner count on x-axis')
    parser.add_argument('--ny', '-y', type=int, help='checker board corner count on y-axis')
    parser.add_argument('--cell-size', '-s', type=float, help='cell width and height in mm')
    parser.add_argument('--dist-coef-n', '-n', type=int, default=5, help='how many distortion coeffs to estimate [0-5]')
    parser.add_argument('--fix-pp', action='store_true', help='fix principal point')
    parser.add_argument('--fix-aspect', action='store_true', help='fix aspect ratio')
    parser.add_argument('--pause', action='store_true', help='pause after each image during corner detection')
    parser.add_argument('--max-err', type=float, help='run calibration a second time after dropping out frames'
                                                      ' with max repr err surpassing this value')
    args = parser.parse_args()

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    detect_corner_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((args.nx * args.ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.nx, 0:args.ny].T.reshape(-1, 2) * args.cell_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    names = []
    imgs = []
    shape = None

    def process_img(img, name):
        if args.rot:
            rot = {180: cv2.ROTATE_180, -90: cv2.ROTATE_90_COUNTERCLOCKWISE, 90: cv2.ROTATE_90_CLOCKWISE}[args.rot]
            img = cv2.rotate(img, rot)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if args.preproc:
            # min_v, max_v = np.quantile(gray, (0.03, 0.999))
            # gray = np.clip(255*(gray.astype(float) - min_v)/(max_v - min_v) + 0.5, 0, 255).astype(np.uint8)
            gray = cv2.resize(gray, None, fx=0.5, fy=0.5)
            img = cv2.resize(img, None, fx=0.5, fy=0.5)

        shape = gray.shape

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (args.nx, args.ny), None, detect_corner_flags)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            imgs.append(img)
            names.append(name)

            # Draw and display the corners
            img_show = img.copy()
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(img_show, (args.nx, args.ny), corners2, ret)
            sc = 1024/np.max(img_show.shape)
            cv2.imshow('detected corners', cv2.resize(img_show, None, fx=sc, fy=sc))
        else:
            print('cant find the corners from %s' % name)
            sc = 1024 / np.max(img.shape)
            cv2.imshow('detected corners', cv2.resize(img, None, fx=sc, fy=sc))
        cv2.setWindowTitle('detected corners', name)
        cv2.waitKey(0 if args.pause else 500)

        return shape

    for path in args.path:
        if os.path.isdir(path):
            files = [fname for fname in os.listdir(path) if fname[-4:] in ('.bmp', '.jpg', '.png')]
            files = sorted(files)
            files = [fname for i, fname in enumerate(files) if i >= args.offset and (i - args.offset) % args.skip == 0]
            for fname in tqdm(files):
                img = cv2.imread(os.path.join(path, fname))
                shape = process_img(img, fname)

        else:
            cap = cv2.VideoCapture(path)
            i, ret = 0, True
            while cap.isOpened() and ret:
                ret, img = cap.read()
                if i >= args.offset and (i - args.offset) % args.skip == 0 and ret:
                    shape = process_img(img, 'frame-%d' % i)
                i += 1

            cap.release()

    cv2.destroyAllWindows()

    if len(imgs) == 0:
        print('too few images found at %s (%d)' % (args.path, len(imgs)))
        return

    flags = 0
    if args.dist_coef_n > 5:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if args.dist_coef_n < 5 and args.dist_coef_n != 3:
        flags |= cv2.CALIB_FIX_K3
    if args.dist_coef_n < 4:
        flags |= cv2.CALIB_FIX_TANGENT_DIST
    if args.dist_coef_n < 2:
        flags |= cv2.CALIB_FIX_K2
    if args.dist_coef_n < 1:
        flags |= cv2.CALIB_FIX_K1

    if args.fix_pp:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

    if args.fix_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO

    print('estimating camera parameters using a total of %d images...' % (len(imgs),))

    for iter in range(2):
        rms, K, dist, rvecs, tvecs, stds, *_ = cv2.calibrateCameraExtended(objpoints, imgpoints, shape[::-1],
                                                                           None, None, flags=flags)
        intr = tuple(np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))
        dist = tuple(dist.flatten())
        stds = stds.flatten()
        intr_lbl = ('f_x', 'f_y', 'c_x', 'c_y')
        dist_lbl = ('k_1', 'k_2', 'p_1', 'p_2', 'k_3')[:len(dist)]

        img_errs = []
        img_projs = []
        for i in range(len(objpoints)):
            if 1:
                img_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            else:
                from visnav.algo.odo.vis_gps_bundleadj import project
                img_proj = project(objpoints[i], np.array([*rvecs[i], *tvecs[i]]).reshape((-1, 6)), K, dist)[:, None, :]

            errors = np.linalg.norm(imgpoints[i] - img_proj, axis=2).flatten()
            img_projs.append(img_proj)
            img_errs.append(errors)

        max_errs = np.max(np.array(img_errs), axis=1)
        I = np.argsort(-max_errs)
        for i in I:
            print('%s max repr err [px]: %.3f' % (names[i], max_errs[i]))

        if args.max_err is not None and iter == 0 and np.any(max_errs >= args.max_err):
            print('recalibrating after dropping %d frames because their max repr err exceeded %f (%s)...' % (
                np.sum(max_errs >= args.max_err), args.max_err, max_errs[max_errs >= args.max_err]))

            J = np.where(max_errs < args.max_err)[0]
            objpoints = [objpoints[j] for j in J]
            imgpoints = [imgpoints[j] for j in J]
            imgs = [imgs[j] for j in J]
            names = [names[j] for j in J]
        else:
            break

    def to_str(lbls, vals, unit='', list_sep=', ', lbl_sep='): ', prec=3):
        return list_sep.join(lbls) + lbl_sep + list_sep.join([tools.fixed_precision(val, prec, True) + unit
                                                              for lbl, val in zip(lbls, vals)])
#        return list_sep.join([lbl + lbl_sep + tools.fixed_precision(val, prec, True) for lbl, val in zip(lbls, vals)])

    print('intrinsics (' + to_str(intr_lbl, intr, prec=5))
    print('dist coefs (' + to_str(dist_lbl, dist, prec=5))
    print('stds (' + to_str(intr_lbl + dist_lbl, stds[:len(intr) + len(dist)]))
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='invalid value encountered in true_divide')
        print('relative (' + to_str(intr_lbl + dist_lbl, 100*stds[:len(intr) + len(dist)]/np.abs(np.array(intr + dist)),
                                    unit='%', prec=2))

    y = np.array(imgpoints).reshape((-1, 2)) - np.array(intr)[None, 2:4]
    yh = np.array(img_projs).reshape((-1, 2)) - np.array(intr)[None, 2:4]
    e = y - yh

    u = y / np.linalg.norm(y, axis=1)[:, None]
    e_radial = np.einsum('ij,ij->i', e, u)
    e_cross = np.linalg.norm(e - e_radial[:, None] * u, axis=1)

    r = np.linalg.norm(y, axis=1)
    J = np.argsort(r)

    print('repr err across all imgs q=0.99: %.3f, e_r: %.3f, e_c: %.3f' % (
        np.quantile(np.array(img_errs).flatten(), 0.99),
        np.quantile(e_radial, 0.99), np.quantile(e_cross, 0.99),
    ))

    from scipy import stats

    plt.figure()
    plt.plot(r[J], e_radial[J], '.')
    bin_means, bin_edges, binnumber = stats.binned_statistic(r[J], e_radial[J], statistic='mean', bins=20)
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], lw=5, color='C1')
    plt.title('radial errors vs distance from principal point')
    plt.tight_layout()

    plt.figure()
    plt.plot(r[J], e_cross[J], '.')
    bin_means, bin_edges, binnumber = stats.binned_statistic(r[J], e_cross[J], statistic='mean', bins=20)
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], lw=5, color='C1')
    plt.title('tangential errors vs distance from principal point')
    plt.tight_layout()

    if 1:
        # project
        for i in I:
            name, pts, img = names[i], img_projs[i], imgs[i]
            plt.figure(figsize=(20, 14))
            plt.imshow(np.flip(img, axis=2))
            for pt in pts.squeeze():
                plt.plot(*pt, 'oC1', mfc='none')
            plt.title(name + ', max-err %.3f px' % max_errs[i])
            plt.tight_layout()
            plt.show()

    else:
        # undistort
        h, w = imgs[0].shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        print('new camera mx (P):\n%s' % (newcameramtx,))
        for img in imgs:
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
        #            x, y, w, h = roi
        #            dst = dst[y:y + h, x:x + w]
            cv2.imshow('undistorted', dst)
            cv2.waitKey()


if __name__ == '__main__':
    main()
