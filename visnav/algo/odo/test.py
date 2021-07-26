import math

import numpy as np
import quaternion

import cv2
import matplotlib.pyplot as plt

from visnav.algo.odo.base import Pose, Frame, PoseEstimate
from visnav.algo import tools


def main():
    cam_mx = np.array([[1580.356552, 0.000000, 994.026697],
                       [0.000000, 1580.553177, 518.938726],
                       [0.000000, 0.000000, 1.000000]])

    sc_v0, sc_v1 = np.array([0, 0, -100]),  np.array([0, -5, -100])
    sc_q0, sc_q1 = tools.ypr_to_q(0, 0, 0), tools.ypr_to_q(0, 0, math.radians(-2))

    w_q0, w_q1 = sc_q0.conj(), sc_q1.conj()
    w_v0, w_v1 = tools.q_times_v(w_q0, -sc_v0), tools.q_times_v(w_q1, -sc_v1)

    f0 = Frame(0, None, 1.0, PoseEstimate(None, Pose(w_v0, w_q0), 0))
    f1 = Frame(1, None, 1.0, PoseEstimate(None, Pose(w_v1, w_q1), 0))

    ref_pts4d = np.concatenate((np.random.uniform(-30, 30, (100, 2)), np.zeros((100, 1)), np.ones((100, 1))), axis=1)

    P0, P1 = cam_mx.dot(f0.to_mx()), cam_mx.dot(f1.to_mx())
    uvs0, uvs1 = P0.dot(ref_pts4d.T).T, P1.dot(ref_pts4d.T).T
    uvs0, uvs1 = uvs0[:, :2] / uvs0[:, 2:], uvs1[:, :2] / uvs1[:, 2:]

    plt.figure(1)
    for uv0, uv1 in zip(uvs0, uvs1):
        plt.plot([uv0[0], uv1[0]], [uv0[1], uv1[1]])
    plt.plot(uvs1[:, 0], uvs1[:, 1], '.')
    plt.gca().invert_yaxis()

    pts3d = []
    for uv0, uv1 in zip(uvs0, uvs1):
        kp4d = cv2.triangulatePoints(P0, P1, uv0.reshape((-1, 1, 2)), uv1.reshape((-1, 1, 2)))
        pt3d = (kp4d.T[:, :3] / kp4d.T[:, 3:])[0]
        pts3d.append(pt3d)

    pts3d = np.array(pts3d)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], '.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    print('done')


if __name__ == '__main__':
    main()
