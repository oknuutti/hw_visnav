from dataclasses import dataclass
import logging
import tracemalloc

import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as spl

#from memory_profiler import profile

from visnav.algo import tools
from visnav.algo.odo.linqr import InnerLinearizerQR
from visnav.algo.tools import Stopwatch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class IterationStats:
    tr_rad: float = np.nan
    l_diff: float = np.nan
    step_quality: float = np.nan
    delta_f: float = np.nan
    delta_r: float = np.nan
    delta_x: float = np.nan
    cost: float = np.nan
    time: float = np.nan
    valid: bool = False
    success: bool = False


class RootBundleAdjuster:
    """
    Based on Square Root Marginalization for Sliding-Window Bundle Adjustment
    by N Demmel, D Schubert, C Sommer, D Cremers and V Usenko,
    in IEEE International Conference on Computer Vision (ICCV), 2021.
    c-code at https://github.com/NikolausDemmel/rootba
    preprint at https://arxiv.org/abs/2109.02182

    Notable modifications:
        - Support external absolute location and orientation measures
        - Support pose independent camera parameter optimization (e.g. optimize cam params shared by all poses)
    """

    MAX_INNER_ITERS = np.inf

    def __init__(self, ini_tr_rad, min_tr_rad, max_tr_rad, ini_vee, vee_factor, thread_n, max_iters, max_time,
                 min_step_quality, xtol, rtol, ftol, max_repr_err, jacobi_scaling_eps,
                 lin_cg_maxiter, lin_cg_tol, preconditioner_type, huber_coefs, use_weighted_residuals):
        self.ini_lambda = 1 / ini_tr_rad
        self.min_lambda = 1 / max_tr_rad
        self.max_lambda = 1 / min_tr_rad
        self.ini_vee = ini_vee
        self.vee_factor = vee_factor
        self.thread_n = thread_n
        self.max_iters = max_iters
        self.max_time = max_time
        self.min_step_quality = min_step_quality
        self.xtol = xtol
        self.rtol = rtol
        self.ftol = ftol
        self.max_repr_err = max_repr_err
        self.lin_opt = dict(
            jacobi_scaling_eps=jacobi_scaling_eps,
            lin_cg_maxiter=lin_cg_maxiter,
            lin_cg_tol=lin_cg_tol,
            preconditioner_type=preconditioner_type,
            huber_coefs=huber_coefs,
            use_weighted_residuals=use_weighted_residuals,
        )

        self._timer = None
        self._lambda = None
        self._lambda_vee = None
        self._linearizer = None
        self._prev_x = None

    def solve(self, prob, callback=None):
        self._timer = Stopwatch(start=True)
        self._lambda = self.ini_lambda
        self._lambda_vee = self.ini_vee
        self._linearizer = LinearizerQR(prob, **self.lin_opt)

        if logger.level == logging.DEBUG:
            tracemalloc.start()

        if 0 in self.max_repr_err:
            n_obs = len(prob.pts2d)
            obs_idxs = prob.filter(self.max_repr_err[0])
            logger.info('Before starting, filtered out %.3f%% of all observations as initial repr err > %.0f' % (
                len(obs_idxs) / n_obs * 100, self.max_repr_err[0]))

        iter_stats = []

        for i in range(self.max_iters):
            stats = self._step(i)
            iter_stats.append(stats)

            logger.info('Iter #%d (%.1fs): f: %.5f, df: %.5f, dl: %.5f, df/dl: %.5f, tr: %s' % (
                i, stats.time, stats.cost, stats.delta_f, stats.l_diff, stats.step_quality,
                tools.fixed_precision(stats.tr_rad, 3, True)))

            if callback is not None:
                callback(prob)

            if i == 0:
                continue
            if stats.delta_f < self.ftol:
                logger.info('ftol reached: %f < %f' % (stats.delta_f, self.ftol))
                break
            if stats.delta_r < self.rtol:
                logger.info('rtol reached: %f < %f' % (stats.delta_r, self.rtol))
                break
            if stats.delta_x < self.xtol:
                logger.info('xtol reached: %f < %f' % (stats.delta_x, self.xtol))
                break
            if self._timer.elapsed > self.max_time > 0:
                logger.info('max time reached: %.0fs > %.0fs' % (self._timer.elapsed, self.max_time))
                break

        if i + 1 == self.max_iters:
            logger.info('max iters (%d) reached' % self.max_iters)

        # TODO: return stats or something?
        return

    def _step(self, i) -> [IterationStats, np.ndarray]:
        timer = Stopwatch(start=True)
        stats = IterationStats(tr_rad=1/self._lambda)

        # is first step or not?
        if i == 0:
            # self._linearizer.start_iter()
            stats.cost = self._linearizer.current_cost()
            # self._linearizer.finish_iter()

            stats.time = timer.stop()
            stats.success = True
            stats.valid = True
            return stats

        if logger.level == logging.DEBUG:
            logger.debug('...calling _linearizer.linearize (mem use %.1f GB, peak %.1f GB)' % (
                         *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
        self._linearizer.linearize()

        j = 0
        while True:
            cost1 = self._inner_step(stats, i)
            j += 1
            if j >= self.MAX_INNER_ITERS:
                break
            if self._lambda > self.max_lambda:
                break
            if stats.success:
                assert stats.valid, 'if not valid cannot be successful'
                break

        stats.cost = cost1
        stats.tr_rad = 1 / self._lambda
        stats.time = timer.stop()
        return stats

    def _inner_step(self, stats: IterationStats, i):
        # solve damped linear system
        if logger.level == logging.DEBUG:
            logger.debug('...calling _linearizer.solve (mem use %.1f GB, peak %.1f GB)' % (
                         *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
        delta_xbp = stats.delta_xbp = self._linearizer.solve(self._lambda)

        if not np.all(np.isfinite(delta_xbp)):
            self._lambda *= self._lambda_vee
            self._lambda_vee *= self.vee_factor
            stats.success = False
            stats.valid = False
            return None

        if logger.level == logging.DEBUG:
            logger.debug('...calling _linearizer.backup_x (mem use %.1f GB, peak %.1f GB)' % (
                         *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
        self._linearizer.backup_x()

        if logger.level == logging.DEBUG:
            logger.debug('...calling _linearizer.update_x (mem use %.1f GB, peak %.1f GB)' % (
                         *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
        l_diff = self._linearizer.update_x(delta_xbp)

        if logger.level == logging.DEBUG:
            logger.debug('...calling _linearizer.current_cost (mem use %.1f GB, peak %.1f GB)' % (
                         *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
        cost1 = self._linearizer.current_cost()

        if logger.level == logging.DEBUG:
            logger.debug('...setting stats object (mem use %.1f GB, peak %.1f GB)' % (
                         *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))

        if cost1 is None:
            stats.success = False
            stats.valid = False
        else:
            stats.cost = cost1
            stats.l_diff = l_diff
            stats.delta_f = self._linearizer.initial_cost() - cost1
            stats.step_quality = stats.delta_f / l_diff
            stats.delta_r = stats.delta_f / self._linearizer.initial_cost()
            stats.valid = l_diff > 0
            stats.success = stats.valid and stats.step_quality > self.min_step_quality

        if stats.success:
            self._lambda = max(self.min_lambda, self._lambda * max(1/3, 1-(2*stats.step_quality-1)**3))
            self._lambda_vee = self.ini_vee
            # self._linearizer.finish_iter()
            if i in self.max_repr_err:
                filtered_ratio, cost1 = self._linearizer.filter(self.max_repr_err[i])
                logger.info('...filtered out %.3f%% of all observations as repr err > %.0f, cost %.5f => %.5f' % (
                            filtered_ratio * 100, self.max_repr_err[i], stats.cost, cost1))
        else:
            self._lambda *= self._lambda_vee
            self._lambda_vee *= self.vee_factor
            # self._linearizer.finish_iter()
            if logger.level == logging.DEBUG:
                logger.debug('...calling _linearizer.restore_x (mem use %.1f GB, peak %.1f GB)' % (
                             *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))

            self._linearizer.restore_x()
            logger.info('...failed with: f: %.5f, df: %.5f, dl: %.5f, df/dl: %.5f => new tr: %s' % (
                        stats.cost, stats.delta_f, stats.l_diff, stats.step_quality,
                        tools.fixed_precision(1 / self._lambda, 3, True),))

        return cost1


class LinearizerQR:
    (
        PRECONDITIONER_TYPE_JACOBI,
        PRECONDITIONER_TYPE_SCHUR_JACOBI,
    ) = range(2)

    def __init__(self, problem, jacobi_scaling_eps=0, lin_cg_maxiter=500, lin_cg_tol=1e-5,
                 preconditioner_type=PRECONDITIONER_TYPE_SCHUR_JACOBI, staged_execution=True,
                 huber_coefs=None, use_weighted_residuals=False):
        self.problem = problem
        self.dtype = self.problem.dtype

        # default tracked to https://github.com/strasdat/Sophus/blob/master/sophus/common.hpp
        self.jacobi_scaling_eps = jacobi_scaling_eps or np.sqrt(1e-10 if self.dtype == np.float64 else 1e-5)

        self.lin_cg_maxiter = lin_cg_maxiter
        self.lin_cg_tol = lin_cg_tol
        self.preconditioner_type = preconditioner_type
        self.staged_execution = staged_execution     # only True implemented
        self.check_projection_validity = True        # TODO: implement

        self._pose_jac_scaling = None
        self._new_linearization_point = True
        self._precond_mx = None                 # precond_blocks_
        self._lqr = InnerLinearizerQR(self.problem, jacobi_scaling_eps=self.jacobi_scaling_eps,
                                      huber_coefs=huber_coefs, use_weighted_residuals=use_weighted_residuals)
    # def start_iter(self):
    #     pass
    #
    # def finish_iter(self):
    #     pass

    def linearize(self):
        # Stage 1: outside lm solver inner loop
        #  - linearization
        #  - scale landmark jacobians
        #  - compute pose jacobian scale
        #  - marginalization of landmarks
        #  - compute jacobi preconditioner

        use_jacobi_precond = self.preconditioner_type == LinearizerQR.PRECONDITIONER_TYPE_JACOBI

        if self.staged_execution:
            self._lqr.linearize()

            if logger.level == logging.DEBUG:
                logger.debug('...calling _lqr.get_Jbp_diag2 (mem use %.1f GB, peak %.1f GB)' % (
                             *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
            d2 = self._lqr.get_Jbp_diag2()

            if logger.level == logging.DEBUG:
                logger.debug('...calling _lqr.scale_Jl_cols (mem use %.1f GB, peak %.1f GB)' % (
                             *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
            self._lqr.scale_Jl_cols()

            if use_jacobi_precond:
                if logger.level == logging.DEBUG:
                    logger.debug('...calling _lqr.get_Jbp_T_Jbp_blockdiag (mem use %.1f GB, peak %.1f GB)' % (
                                 *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
                self._precond_mx = self._lqr.get_Jbp_T_Jbp_blockdiag()

            if logger.level == logging.DEBUG:
                logger.debug('...calling _lqr.marginalize (mem use %.1f GB, peak %.1f GB)' % (
                             *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
            self._lqr.marginalize()
        else:
            d2 = self._lqr.get_stage1(self._precond_mx if use_jacobi_precond else None)

        if logger.level == logging.DEBUG:
            logger.debug('...setting _linearizer._pose_jac_scaling (mem use %.1f GB, peak %.1f GB)' % (
                         *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
        self._pose_jac_scaling = 1 / (self.jacobi_scaling_eps + np.sqrt(d2))
        self._new_linearization_point = True

    def solve(self, _lambda):
        # Stage 2: inside lm solver inner loop
        #  - scale pose jacobians (1st inner iteration)
        #  - dampen
        #  - compute rhs b
        #  - compute schur_jacobi preconditioner

        use_schur_jacobi_precond = self.preconditioner_type == LinearizerQR.PRECONDITIONER_TYPE_SCHUR_JACOBI

        self._lqr.set_pose_damping(_lambda)

        if self.staged_execution:
            if self._new_linearization_point:
                if logger.level == logging.DEBUG:
                    logger.debug('...calling _lqr.scale_Jbp_cols (mem use %.1f GB, peak %.1f GB)' % (
                                 *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
                self._lqr.scale_Jbp_cols(self._pose_jac_scaling)

            if logger.level == logging.DEBUG:
                logger.debug('...calling _lqr.set_landmark_damping (mem use %.1f GB, peak %.1f GB)' % (
                             *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
            self._lqr.set_landmark_damping(_lambda)

            if use_schur_jacobi_precond:
                if logger.level == logging.DEBUG:
                    logger.debug('...calling _lqr.get_Q2TJbp_T_Q2TJbp_blockdiag (mem use %.1f GB, peak %.1f GB)' % (
                                 *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
                self._precond_mx = self._lqr.get_Q2TJbp_T_Q2TJbp_blockdiag()

            if logger.level == logging.DEBUG:
                logger.debug('...calling _lqr.get_Q2TJbp_T_Q2Tr (mem use %.1f GB, peak %.1f GB)' % (
                             *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))
            b = self._lqr.get_Q2TJbp_T_Q2Tr()
        else:
            b = self._lqr.get_stage2(_lambda,
                                     self._pose_jac_scaling if self._new_linearization_point else None,
                                     self._precond_mx if use_schur_jacobi_precond else None)

        # solving by:
        # - inverting preconditioner
        # - running preconditioned conjugate gradient algo

        n = self._precond_mx.shape[0]
        if self.preconditioner_type == LinearizerQR.PRECONDITIONER_TYPE_JACOBI:
            if logger.level == logging.DEBUG:
                logger.debug('...setting _lqr._precond_mx (mem use %.1f GB, peak %.1f GB)' % (
                             *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))

            if self._new_linearization_point:
                self._precond_mx = self._pose_jac_scaling.dot(self._precond_mx).dot(self._pose_jac_scaling)
            P = self._precond_mx + _lambda * sp.eye(n, format='csc', dtype=self.dtype)

        elif self.preconditioner_type == LinearizerQR.PRECONDITIONER_TYPE_SCHUR_JACOBI:
            # jacobian scaling and damping diagonal is already applied
            P = self._precond_mx

        else:
            assert False, 'invalid preconditioner type'

        # TODO: try different ways, see which one is fastest
        if 0:
            M_x = lambda x: spl.spsolve(P, x)
            M = spl.LinearOperator((n, n), M_x, dtype=self.dtype)
        elif 1:
            if logger.level == logging.DEBUG:
                logger.debug('...creating M (mem use %.1f GB, peak %.1f GB)' % (
                             *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))

            M_x = spl.factorized(P)
            M = spl.LinearOperator((n, n), M_x, dtype=self.dtype)
        else:
            M = spl.inv(P)

        if self.preconditioner_type == LinearizerQR.PRECONDITIONER_TYPE_SCHUR_JACOBI:
            Hpp = P     # precond_mx == P == Hpp (!)
        elif 1:
            # TODO: try using LinearOperator instead
            if logger.level == logging.DEBUG:
                logger.debug('...calling _lqr.get_Q2TJbp_T_Q2TJbp_blockdiag (mem use %.1f GB, peak %.1f GB)' % (
                             *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))

            Hpp = self._lqr.get_Q2TJbp_T_Q2TJbp_blockdiag()
        else:
            Hpp_x = lambda x: self._lqr.right_multiply(x)
            Hpp = spl.LinearOperator((n, n), Hpp_x, dtype=self.dtype)

        if logger.level == logging.DEBUG:
            logger.debug('...calling spl.cg (mem use %.1f GB, peak %.1f GB)' % (
                         *map(lambda x: x / 1024 / 1024 / 1024, tracemalloc.get_traced_memory()),))

        delta_xbp, info = spl.cg(Hpp, b, M=M, tol=self.lin_cg_tol, maxiter=self.lin_cg_maxiter)  # TODO: Hpp == P ??

        # TODO: log timing, messages, iterations etc
        # // negate the pose increment, since we solve H(-x) = b
        delta_xbp *= -1

        self._new_linearization_point = False
        return delta_xbp[:, None]    # includes delta_xb (delta_xp[:nb])

    def update_x(self, delta_xbp):
        delta_xl, l_diff = self._lqr.backsub_xl(delta_xbp)

        # unscale pose increments
        delta_xbp *= self._pose_jac_scaling.reshape((-1, 1))

        # update batch (camera) params
        self._lqr.problem.xb = self._lqr.problem.xb + delta_xbp[:self._lqr.nb]

        # update poses
        self._lqr.problem.xp = self._lqr.problem.xp + delta_xbp[self._lqr.nb:]

        # update landmarks
        self._lqr.problem.xl = self._lqr.problem.xl + delta_xl.reshape((-1, 1))

        return l_diff

    def backup_x(self):
        self._prev_x = self._lqr.problem.x

    def restore_x(self):
        self._lqr.problem.x = self._prev_x

    def initial_cost(self):
        return self._lqr.initial_cost()

    def current_cost(self):
        return self._lqr.current_cost()

    def filter(self, max_repr_err):
        return self._lqr.filter(max_repr_err)
