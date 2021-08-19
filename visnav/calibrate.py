import os
import argparse

import numpy as np
import cv2


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Calibrate camera an image taken of a checker board')
    parser.add_argument('--path', '-f', help='path to the calibration images')
    parser.add_argument('--nx', '-x', type=int, help='checker board corner count on x-axis')
    parser.add_argument('--ny', '-y', type=int, help='checker board corner count on y-axis')
    parser.add_argument('--cell-size', '-s', type=float, help='cell width and height in mm')
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

    for fname in os.listdir(args.path):
        if fname[-4:] in ('.bmp', '.jpg', '.jpeg', '.png'):
            img = cv2.imread(os.path.join(args.path, fname))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
                print('cant find the corners from %s' % fname)

    if len(imgs) == 0:
        print('too few images found at %s (%d)' % (args.path, len(imgs)))
        return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('camera mx (K):\n%s' % (mtx,))
    print('\ndist coefs (D): %s' % (dist,))

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        errors = np.linalg.norm(imgpoints[i] - imgpoints2, axis=2).flatten()
        print('img #%d reprojection errors [px]: %.3f' % (i, np.mean(errors)))

    # undistort
    if 1:
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
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
