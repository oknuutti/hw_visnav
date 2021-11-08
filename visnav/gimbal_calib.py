import argparse
import pickle

import numpy as np
import quaternion
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from visnav.algo import tools
from visnav.missions.nokia import NokiaSensor


def main():
    parser = argparse.ArgumentParser(description='Estimate orientation offset between body and cam on gimbal')
    parser.add_argument('--res', required=True, help='path to result pickle')
    parser.add_argument('--skip', default=0, type=int, help='skip first x frames')
    parser.add_argument('--huber-coef', default=0, type=float, help='use pseudo huber loss')
    parser.add_argument('--nadir-looking', action='store_true', help='better plot for nadir looking cam')
    parser.add_argument('--init-ypr', default=[0.0, 0.0, 0.0], type=float, nargs=3)
    args = parser.parse_args()

    with open(args.res, 'rb') as fh:
        results, map3d, frame_names, meta_names, ground_truth = pickle.load(fh)

    def cf(pose):
        return (-pose).to_global(NokiaSensor.b2c).quat

    # (kf.pose, kf.measure, kf.time, kf.id)
    meas_q = np.array([tools.ypr_to_q(*meas.data[3:6]) for pose, meas, _, _ in results if meas and pose.post])
    est_q = np.array([(-pose.post).to_global(NokiaSensor.b2c).quat for pose, meas, _, _ in results if meas and pose.post])

    meas_q = meas_q[args.skip:]
    est_q = est_q[args.skip:]

    nl_dq = tools.eul_to_q((-np.pi / 2,), 'y') if args.nadir_looking else quaternion.one
    est_ypr = np.array([tools.q_to_ypr(nl_dq.conj() * q) for q in est_q]) / np.pi * 180

    print('optimizing orientation offset based on %d measurements...' % (len(meas_q),))

    def costfun(x, meas_q, est_q):
        off_q = np.quaternion(*x).normalized()
        delta_angles = tools.angle_between_q_arr(est_q, meas_q * off_q)
        return delta_angles

    x0 = tools.ypr_to_q(*(np.array(args.init_ypr)/180*np.pi)).components
    res = least_squares(costfun, x0, verbose=2, ftol=1e-5, xtol=1e-5, gtol=1e-8, max_nfev=1000,
                        loss='huber' if args.huber_coef > 0 else 'linear', f_scale=args.huber_coef or 1.0,  #huber_coef,
                        args=(meas_q, est_q))

    off_q = np.quaternion(*res.x).normalized() if 1 else quaternion.one
    corr_ypr = np.array([tools.q_to_ypr(nl_dq.conj() * q * off_q) for q in meas_q]) / np.pi * 180

    print('offset q: %s, ypr: %s' % (off_q, np.array(tools.q_to_ypr(off_q)) / np.pi * 180,))

    plt.plot(est_ypr[:, 0], 'C0:')
    plt.plot(est_ypr[:, 1], 'C1:')
    plt.plot(est_ypr[:, 2], 'C2:')
    plt.plot(corr_ypr[:, 0], 'C0-')
    plt.plot(corr_ypr[:, 1], 'C1-')
    plt.plot(corr_ypr[:, 2], 'C2-')
    plt.show()


if __name__ == '__main__':
    main()
