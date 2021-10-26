import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import cv2


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Calibrate camera an image taken of a checker board')
    parser.add_argument('--path', '-f', action='append', help='path to the calibration images / video(s)')
    parser.add_argument('--skip', '-i', type=int, default=1, help='frame interval (1=no skipping, 2=use every other frame, etc)')
    parser.add_argument('--offset', '-o', type=int, default=0, help='skip these many images from the start')
    parser.add_argument('--nx', '-x', type=int, help='checker board corner count on x-axis')
    parser.add_argument('--ny', '-y', type=int, help='checker board corner count on y-axis')
    parser.add_argument('--cell-size', '-s', type=float, help='cell width and height in mm')
    parser.add_argument('--dist-coef-n', '-n', type=int, default=5, help='how many distortion coeffs to estimate [0-5]')
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
    imgs = []
    shape = None

    def process_img(img, name):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (args.nx, args.ny), None, detect_corner_flags)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            imgs.append(img)

            # Draw and display the corners
            img_show = img.copy()
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(img_show, (args.nx, args.ny), corners2, ret)
            sc = 1024/np.max(img_show.shape)
            cv2.imshow('calibration', cv2.resize(img_show, None, fx=sc, fy=sc))
            cv2.waitKey(500)
        else:
            print('cant find the corners from %s' % name)

        return shape

    for path in args.path:
        if os.path.isdir(path):
            files = [fname for fname in os.listdir(path) if fname in ('.bmp', '.jpg', '.jpeg', '.png')]
            files = sorted(files)
            for i, fname in enumerate(files):
                if i >= args.offset and (i - args.offset) % args.skip == 0:
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

    print('estimating camera parameters using a total of %d images...' % (len(imgs),))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape[::-1], None, None, flags=flags)

    print('camera mx (K):\n%s' % (mtx,))
    print('\ndist coefs (D): %s' % (dist,))

    img_errs = []
    img_projs = []
    for i in range(len(objpoints)):
        img_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        errors = np.linalg.norm(imgpoints[i] - img_proj, axis=2).flatten()
        print('img #%d reprojection errors [px] q=0.99: %.3f' % (i, np.quantile(errors, 0.99)))
        img_errs.append(errors)
        img_projs.append(img_proj)

    print('repr err across all imgs q=0.99: %.3f' % np.quantile(np.array(img_errs).flatten(), 0.99))
    cv2.destroyAllWindows()

    if 1:
        # project
        for pts, img in zip(img_projs, imgs):
            plt.figure(figsize=(9, 5))
            plt.imshow(img)
            for pt in pts.squeeze():
                plt.plot(*pt, 'oC1', mfc='none')
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
