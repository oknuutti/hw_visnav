import math
import time
import dateutil.parser as dparser

import numpy as np
#import numba as nb
import quaternion  # adds to numpy  # noqa # pylint: disable=unused-import
import sys

# import scipy
# from astropy.coordinates import SkyCoord
# from scipy.interpolate import RectBivariateSpline
# from scipy.interpolate import NearestNDInterpolator
# from scipy.spatial.ckdtree import cKDTree

from visnav.settings import *


class PositioningException(Exception):
    pass


class Stopwatch:
    # from https://www.safaribooksonline.com/library/view/python-cookbook-3rd/9781449357337/ch13s13.html

    def __init__(self, elapsed=0.0, func=time.perf_counter):
        self._elapsed = elapsed
        self._func = func
        self._start = None

    @property
    def elapsed(self):
        return self._elapsed + ((self._func() - self._start) if self.running else 0)

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self._elapsed += end - self._start
        self._start = None

    def reset(self):
        self._elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class Time:
    def __init__(self, datestr, **kwargs):
        self.unix = dparser.parse(datestr).timestamp() if isinstance(datestr, str) else float(datestr)

    @property
    def sec(self):
        return self.unix

    def __sub__(self, other: 'Time'):
        return Time(self.unix - other.unix)


def sphere_angle_radius(loc, r):
    return np.arcsin(r / np.linalg.norm(loc, axis=1))


def dist_across_and_along_vect(A, b):
    """ A: array of vectors, b: axis vector """
    lat, lon, r = cartesian2spherical(*b)
    q = lat_lon_roll_to_q(lat, lon, 0).conj()
    R = quaternion.as_rotation_matrix(q)
    Ab = R.dot(A.T).T
    d = Ab[:, 0:1]
    r = np.linalg.norm(Ab[:, 1:3], axis=1).reshape((-1, 1))
    return r, d


def point_vector_dist(A, B, dist_along_v=False):
    """ A: point, B: vector """

    # (length of b)**2
    normB2 = (B ** 2).sum(-1).reshape((-1, 1))

    # a dot b vector product (project a on b but also times length of b)
    diagAB = (A * B).sum(-1).reshape((-1, 1))

    # A projected along B (projection = a dot b/||b|| * b/||b||)
    A_B = (diagAB / normB2) * B

    # vector from projected A to A, it is perpendicular to B
    AB2A = A - A_B

    # diff vector lengths
    normD = np.sqrt((AB2A ** 2).sum(-1)).reshape((-1, 1))
    return (normD, diagAB / np.sqrt(normB2)) if dist_along_v else normD


def sc_asteroid_max_shift_error(A, B):
    """
    Calculate max error between two set of vertices when projected to camera,
    A = estimated vertex positions
    B = true vertex positions
    Error is a vector perpendicular to B, i.e. A - A||
    """

    # diff vector lengths
    normD = point_vector_dist(A, B)

    # max length of diff vectors
    return np.max(normD)


#@nb.njit(nb.f8[:](nb.f8[:], nb.f8[:]))
def cross3d(left, right):
    # for short vectors cross product is faster in pure python than with numpy.cross
    x = ((left[1] * right[2]) - (left[2] * right[1]))
    y = ((left[2] * right[0]) - (left[0] * right[2]))
    z = ((left[0] * right[1]) - (left[1] * right[0]))
    return np.array((x, y, z))


def normalize_v(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


#@nb.njit(nb.types.f8[:](nb.types.f8[:]))
def normalize_v_f8(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def normalize_mx(mx):
    norm = np.linalg.norm(mx, axis=1)
    mx = mx.copy()
    mx[norm > 0, :] = mx[norm > 0, :] / norm[norm > 0, None]
    mx[norm == 0, :] = 0
    return mx


def generate_field_fft(shape, sd=(0.33, 0.33, 0.34), len_sc=(0.5, 0.5 / 4, 0.5 / 16)):
    from visnav.algo.image import ImageProc
    sds = sd if getattr(sd, '__len__', False) else [sd]
    len_scs = len_sc if getattr(len_sc, '__len__', False) else [len_sc]
    assert len(shape) == 2, 'only 2d shapes are valid'
    assert len(sds) == len(len_scs), 'len(sd) differs from len(len_sc)'
    n = np.prod(shape)

    kernel = np.sum(
        np.stack([1 / len_sc * sd * n * ImageProc.gkern2d(shape, 1 / len_sc) for sd, len_sc in zip(sds, len_scs)],
                 axis=2), axis=2)
    f_img = np.random.normal(0, 1, shape) + np.complex(0, 1) * np.random.normal(0, 1, shape)
    f_img = np.real(np.fft.ifft2(np.fft.fftshift(kernel * f_img)))
    return f_img


#@nb.njit(nb.types.f8[:](nb.types.f8[:], nb.types.f8[:], nb.types.f8[:]))
def _surf_normal(x1, x2, x3):
#    a, b, c = np.array(x1, dtype=np.float64), np.array(x2, dtype=np.float64), np.array(x3, dtype=np.float64)
    return normalize_v_f8(cross3d(x2-x1, x3-x1))


def surf_normal(x1, x2, x3):
    a, b, c = np.array(x1, dtype=np.float64), np.array(x2, dtype=np.float64), np.array(x3, dtype=np.float64)
    return _surf_normal(a, b, c)


#    return normalize_v_f8(cross3d(b-a, c-a))

def vector_projection(a, b):
    return a.dot(b) / b.dot(b) * b


def vector_rejection(a, b):
    return a - vector_projection(a, b)


def parallax(f0, f1, pt):
    if len(pt.shape) == 1:
        return angle_between_v(f0-pt, f1-pt)
    return angle_between_mx(f0-pt, f1-pt)


def to_cartesian(lat, lon, alt, lat0, lon0, alt0):
    from pygeodesy.ltp import LocalCartesian
    lc = LocalCartesian(lat0, lon0, alt0)
    xyz = lc.forward(lat, lon, alt)
    return np.array(xyz.xyz)


def angle_between_v(v1, v2):
    # Notice: only returns angles between 0 and 180 deg

    try:
        v1 = np.reshape(v1, (1, -1))
        v2 = np.reshape(v2, (-1, 1))

        n1 = normalize_v(v1)
        n2 = normalize_v(v2)

        cos_angle = n1.dot(n2)
    except TypeError as e:
        raise Exception('Bad vectors:\n\tv1: %s\n\tv2: %s' % (v1, v2)) from e

    return math.acos(np.clip(cos_angle, -1, 1))


def angle_between_v_mx(a, B, normalize=True):
    Bn = B / np.linalg.norm(B, axis=1).reshape((-1, 1)) if normalize else B
    an = normalize_v(a).reshape((-1, 1)) if normalize else a
    return np.arccos(np.clip(Bn.dot(an), -1.0, 1.0))


def angle_between_mx(A, B):
    return angle_between_rows(A, B)


def angle_between_rows(A, B, normalize=True):
    assert A.shape[1] == 3 and B.shape[1] == 3, 'matrices need to be of shape (n, 3) and (m, 3)'
    if A.shape[0] == B.shape[0]:
        # from https://stackoverflow.com/questions/50772176/calculate-the-angle-between-the-rows-of-two-matrices-in-numpy/50772253
        cos_angles = np.einsum('ij,ij->i', A, B)
        if normalize:
            p2 = np.einsum('ij,ij->i', A, A)
            p3 = np.einsum('ij,ij->i', B, B)
            cos_angles /= np.sqrt(p2 * p3)
    else:
        if normalize:
            A = A / np.linalg.norm(A, axis=1).reshape((-1, 1))
            B = B / np.linalg.norm(B, axis=1).reshape((-1, 1))
        cos_angles = B.dot(A.T)

    return np.arccos(np.clip(cos_angles, -1.0, 1.0))


def rand_q(angle):
    r = normalize_v(np.random.normal(size=3))
    return angleaxis_to_q(np.hstack((angle, r)))


def angle_between_q(q1, q2):
    # from  https://chrischoy.github.io/research/measuring-rotation/
    qd = q1.conj() * q2
    return abs(wrap_rads(2 * math.acos(qd.normalized().w)))


def angle_between_q_arr(q1, q2):
    qd = quaternion.as_float_array(q1.conj() * q2)
    qd = qd / np.linalg.norm(qd, axis=1).reshape((-1, 1))
    return np.abs(wrap_rads(2 * np.arccos(qd[:, 0])))


def angle_between_lat_lon_roll(llr1, llr2):
    q1 = lat_lon_roll_to_q(*llr1)
    q2 = lat_lon_roll_to_q(*llr2)
    return angle_between_q(q1, q2)


def distance_mx(A, B=None):
    if B is None:
        B = A
    assert A.shape[1] == B.shape[1], 'matrices must have same amount of columns'
    k = A.shape[1]
    O = np.repeat(A.reshape((-1, 1, k)), B.shape[0], axis=1) - np.repeat(B.reshape((1, -1, k)), A.shape[0], axis=0)
    D = np.linalg.norm(O, axis=2)
    return D


def q_to_unitbase(q):
    U0 = quaternion.as_quat_array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])
    Uq = q * U0 * q.conj()
    return quaternion.as_float_array(Uq)[:, 1:]


def equatorial_to_ecliptic(ra, dec):
    """ translate from equatorial ra & dec to ecliptic ones """
    sc = SkyCoord(ra, dec, unit='deg', frame='icrs', obstime='J2000') \
        .transform_to('barycentrictrueecliptic')
    return sc.lat.value, sc.lon.value


def q_to_angleaxis(q, compact=False):
    theta = math.acos(np.clip(q.w, -1, 1)) * 2.0
    v = normalize_v(np.array([q.x, q.y, q.z]))
    if compact:
        return theta * v
    else:
        return np.array((theta,) + tuple(v))


def angleaxis_to_q(rv):
    """ first angle, then axis """
    if len(rv) == 4:
        theta = rv[0]
        v = normalize_v(np.array(rv[1:]))
    elif len(rv) == 3:
        theta = math.sqrt(sum(x ** 2 for x in rv))
        v = np.array(rv) / (1 if theta == 0 else theta)
    else:
        raise Exception('Invalid angle-axis vector: %s' % (rv,))

    w = math.cos(theta / 2)
    v = v * math.sin(theta / 2)
    return np.quaternion(w, *v).normalized()


def lat_lon_roll_to_q(lat, lon, roll):
    # intrinsic euler rotations z-y'-x'' for lat lon and roll
    return (
            np.quaternion(math.cos(lon / 2), 0, 0, math.sin(lon / 2))
            * np.quaternion(math.cos(-lat / 2), 0, math.sin(-lat / 2), 0)
            * np.quaternion(math.cos(roll / 2), math.sin(roll / 2), 0, 0)
    )


def ypr_to_q(yaw, pitch, roll):
    # Tait-Bryan angles, aka yaw-pitch-roll, nautical angles, cardan angles
    # intrinsic euler rotations z-y'-x'', pitch=-lat, yaw=lon
    return (
            np.quaternion(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
            * np.quaternion(math.cos(pitch / 2), 0, math.sin(pitch / 2), 0)
            * np.quaternion(math.cos(roll / 2), math.sin(roll / 2), 0, 0)
    )


def eul_to_q(angles, order='xyz', reverse=False):
    assert len(angles) == len(order), 'len(angles) != len(order)'
    q = quaternion.one
    idx = {'x': 0, 'y': 1, 'z': 2}
    for angle, axis in zip(angles, order):
        w = math.cos(angle / 2)
        v = [0, 0, 0]
        v[idx[axis]] = math.sin(angle / 2)
        dq = np.quaternion(w, *v)
        q = (dq * q) if reverse else (q * dq)
    return q


def q_to_lat_lon_roll(q):
    # from https://math.stackexchange.com/questions/687964/getting-euler-tait-bryan-angles-from-quaternion-representation
    q0, q1, q2, q3 = quaternion.as_float_array(q)
    roll = np.arctan2(q2 * q3 + q0 * q1, .5 - q1 ** 2 - q2 ** 2)
    lat = -np.arcsin(np.clip(-2 * (q1 * q3 - q0 * q2), -1, 1))
    lon = np.arctan2(q1 * q2 + q0 * q3, .5 - q2 ** 2 - q3 ** 2)
    return lat, lon, roll


def q_to_ypr(q):
    # from https://math.stackexchange.com/questions/687964/getting-euler-tait-bryan-angles-from-quaternion-representation
    q0, q1, q2, q3 = quaternion.as_float_array(q)
    roll = np.arctan2(q2 * q3 + q0 * q1, .5 - q1 ** 2 - q2 ** 2)
    pitch = np.arcsin(np.clip(-2 * (q1 * q3 - q0 * q2), -1, 1))
    yaw = np.arctan2(q1 * q2 + q0 * q3, .5 - q2 ** 2 - q3 ** 2)
    return yaw, pitch, roll


def qarr_to_ypr(qarr):
    # from https://math.stackexchange.com/questions/687964/getting-euler-tait-bryan-angles-from-quaternion-representation
    roll = np.arctan2(qarr[:, 2] * qarr[:, 3] + qarr[:, 0] * qarr[:, 1], .5 - qarr[:, 1] ** 2 - qarr[:, 2] ** 2)
    pitch = np.arcsin(np.clip(-2 * (qarr[:, 1] * qarr[:, 3] - qarr[:, 0] * qarr[:, 2]), -1, 1))
    yaw = np.arctan2(qarr[:, 1] * qarr[:, 2] + qarr[:, 0] * qarr[:, 3], .5 - qarr[:, 2] ** 2 - qarr[:, 3] ** 2)
    return np.stack((yaw, pitch, roll), axis=1)


def mean_q(qs, ws=None):
    """
    returns a (weighted) mean of a set of quaternions
    idea is to rotate a bit in the direction of new quaternion from the sum of previous rotations
    NOTE: not tested properly, might not return same mean quaternion if order of input changed
    """
    wtot = 0
    qtot = quaternion.one
    for q, w in zip(qs, np.ones((len(qs),)) if ws is None else ws):
        ddaa = q_to_angleaxis(qtot.conj() * q)
        ddaa[0] = wrap_rads(ddaa[0]) * w / (w + wtot)
        qtot = angleaxis_to_q(ddaa) * qtot
        wtot += w
    return qtot


# from https://gist.github.com/davegreenwood/e1d2227d08e24cc4e353d95d0c18c914
# ~3.5x slower than cv2.triangulatePoints, seems to be less accurate?
def triangulate_nviews(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]


def naive_triangulate_points(P, uv):
    A = np.vstack(P)
    y = np.vstack([np.vstack((u.reshape((2, -1)), 1)) for u in uv])
    x = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y)
    return x


def q_times_img_coords(q, pts2d, cam, distort=True, opengl=False):
    # project to unit sphere
    vo = cam.to_unit_sphere(pts2d, undistort=distort, opengl=opengl)

    # rotate vector
    vn = q_times_mx(q, vo)

    # project back to image space
    pt2d_n = np.ones((len(pts2d), 2)) * np.nan

    # still in front of camera
    if opengl:
        mask = vn[:, 2] < 0
    else:
        mask = vn[:, 2] > 0

    pt2d_n[mask, :] = cam.calc_img_R(vn[mask, :], distort=distort, legacy=not opengl) - 0.5
    return pt2d_n


def q_times_v(q, v):
    qv = np.quaternion(0, *(v.flatten()))
    qv2 = q * qv * q.conj()
    return np.array([qv2.x, qv2.y, qv2.z])


def q_times_mx(q, mx):
    qqmx = q * mx2qmx(mx) * q.conj()
    aqqmx = quaternion.as_float_array(qqmx)
    return aqqmx[:, 1:]


def mx2qmx(mx):
    qmx = np.zeros((mx.shape[0], 4))
    qmx[:, 1:] = mx
    return quaternion.as_quat_array(qmx)


def wrap_rads(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_degs(a):
    return (a + 180) % 360 - 180


def eccentric_anomaly(eccentricity, mean_anomaly, tol=1e-6):
    # from http://www.jgiesen.de/kepler/kepler.html

    E = mean_anomaly if eccentricity < 0.8 else math.pi
    F = E - eccentricity * math.sin(mean_anomaly) - mean_anomaly;
    for i in range(30):
        if abs(F) < tol:
            break
        E = E - F / (1.0 - eccentricity * math.cos(E))
        F = E - eccentricity * math.sin(E) - mean_anomaly

    return round(E / tol) * tol


def solar_elongation(ast_v, sc_q):
    sco_x, sco_y, sco_z = q_to_unitbase(sc_q)

    if USE_ICRS:
        try:
            sc = SkyCoord(x=ast_v[0], y=ast_v[1], z=ast_v[2], frame='icrs',
                          unit='m', representation_type='cartesian', obstime='J2000') \
                .transform_to('hcrs') \
                .represent_as('cartesian')
            ast_v = np.array([sc.x.value, sc.y.value, sc.z.value])
        except NameError:
            # if SkyCoord not present, do not correct solar system barycenter -> center of the sun
            pass

    # angle between camera axis and the sun, 0: right ahead, pi: behind
    elong = angle_between_v(-ast_v, sco_x)

    # direction the sun is at when looking along camera axis
    nvec = np.cross(sco_x, ast_v)
    direc = angle_between_v(nvec, sco_z)

    # decide if direction needs to be negative or not
    if np.cross(nvec, sco_z).dot(sco_x) < 0:
        direc = -direc

    return elong, direc


def find_nearest_lesser(array, value):
    I = np.where(array < value)
    idx = (np.abs(array - value)).argmin()
    return array[I[idx]], I[idx]


def find_nearest_greater(array, value):
    I = np.where(array > value)
    idx = (np.abs(array - value)).argmin()
    return array[I[idx]], I[idx]


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def find_nearest_arr(array, value, ord=None, fun=None):
    diff = array - value
    idx = np.linalg.norm(diff if fun is None else list(map(fun, diff)), ord=ord, axis=1).argmin()
    return array[idx], idx


def find_nearest_n(array, value, r, ord=None, fun=None):
    diff = array - value
    d = np.linalg.norm(diff if fun is None else list(map(fun, diff)), ord=ord, axis=1)
    idxs = np.where(d < r)
    return idxs[0]


def find_nearest_each(haystack, needles, ord=None):
    assert len(haystack.shape) == 2 and len(needles.shape) == 2 and haystack.shape[1] == needles.shape[1], \
        'wrong shapes for haystack and needles, %s and %s, respectively' % (haystack.shape, needles.shape)
    c = haystack.shape[1]
    diff_mx = np.repeat(needles.reshape((-1, 1, c)), haystack.shape[0], axis=1) - np.repeat(
        haystack.reshape((1, -1, c)), needles.shape[0], axis=0)
    norm_mx = np.linalg.norm(diff_mx, axis=2, ord=ord)
    idxs = norm_mx.argmin(axis=1)
    return haystack[idxs], idxs


def cartesian2spherical(x, y, z):
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    lat = math.pi / 2 - theta
    lon = phi
    return np.array([lat, lon, r])


def spherical2cartesian(lat, lon, r):
    theta = math.pi / 2 - lat
    phi = lon
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return np.array([x, y, z])


def spherical2cartesian_arr(A, r=None):
    theta = math.pi / 2 - A[:, 0]
    phi = A[:, 1]
    r = (r or A[:, 2])
    x = r * np.sin(theta)
    y = x * np.sin(phi)
    x *= np.cos(phi)
    # x = r * np.sin(theta) * np.cos(phi)
    # y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.vstack([x, y, z]).T


def discretize_v(v, tol=None, lat_range=(-math.pi / 2, math.pi / 2), points=None):
    """
    simulate feature database by giving closest light direction with given tolerance
    """

    if tol is not None and points is not None or tol is None and points is None:
        assert False, 'Give either tol or points'
    elif tol is not None:
        points = bf2_lat_lon(tol, lat_range=lat_range)

    lat, lon, r = cartesian2spherical(*v)

    (nlat, nlon), idx = find_nearest_arr(
        points,
        np.array((lat, lon)),
        ord=2,
        fun=wrap_rads,
    )

    ret = spherical2cartesian(nlat, nlon, r)
    return ret, idx


def discretize_q(q, tol=None, lat_range=(-math.pi / 2, math.pi / 2), points=None):
    """
    simulate feature database by giving closest lat & roll with given tolerance
    and set lon to zero as feature detectors are rotation invariant (in opengl coords)
    """

    if tol is not None and points is not None or tol is None and points is None:
        assert False, 'Give either tol or points'
    elif tol is not None:
        points = bf2_lat_lon(tol, lat_range=lat_range)

    lat, lon, roll = q_to_lat_lon_roll(q)
    (nlat, nroll), idx = find_nearest_arr(
        points,
        np.array((lat, roll)),
        ord=2,
        fun=wrap_rads,
    )
    nq0 = lat_lon_roll_to_q(nlat, 0, nroll)
    return nq0, idx


def bf_lat_lon(tol, lat_range=(-math.pi / 2, math.pi / 2)):
    # tol**2 == (step/2)**2 + (step/2)**2   -- 7deg is quite nice in terms of len(lon)*len(lat) == 1260
    step = math.sqrt(2) * tol
    lat_steps = np.linspace(*lat_range, num=math.ceil((lat_range[1] - lat_range[0]) / step), endpoint=False)[1:]
    lon_steps = np.linspace(-math.pi, math.pi, num=math.ceil(2 * math.pi / step), endpoint=False)
    return lat_steps, lon_steps


def bf2_lat_lon(tol, lat_range=(-math.pi / 2, math.pi / 2)):
    # tol**2 == (step/2)**2 + (step/2)**2   -- 7deg is quite nice in terms of len(lon)*len(lat) == 1260
    step = math.sqrt(2) * tol
    lat_steps = np.linspace(*lat_range, num=math.ceil((lat_range[1] - lat_range[0]) / step), endpoint=False)[1:]

    # similar to https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    points = []
    for lat in lat_steps:
        Mphi = math.ceil(2 * math.pi * math.cos(lat) / step)
        lon_steps = np.linspace(-math.pi, math.pi, num=Mphi, endpoint=False)
        points.extend(zip([lat] * len(lon_steps), lon_steps))

    return points


def robust_mean(arr, discard_percentile=0.2, ret_n=False, axis=None):
    J = np.logical_not(np.isnan(arr))
    if axis is not None:
        J = np.all(J, axis=1 if axis == 0 else 0)
    if axis == 0:
        arr = arr[J, :]
    elif axis == 1:
        arr = arr[:, J]
    else:
        arr = arr[J]

    low = np.percentile(arr, discard_percentile, axis=axis)
    high = np.percentile(arr, 100 - discard_percentile, axis=axis)
    I = np.logical_and(low < arr, arr < high)
    if axis is not None:
        I = np.all(I, axis=1 if axis == 0 else 0)
    m = np.mean(arr[:, I] if axis == 1 else arr[I], axis=axis)
    return (m, np.sum(I, axis=axis)) if ret_n else m


def robust_std(arr, discard_percentile=0.2, mean=None, axis=None):
    corr = 1
    if mean is None:
        mean, n = robust_mean(arr, discard_percentile=discard_percentile, ret_n=True, axis=axis)
        corr = n / (n - 1)
    return np.sqrt(robust_mean((arr - mean) ** 2, discard_percentile=discard_percentile, axis=axis) * corr)


def mv_normal(mean, cov=None, L=None, size=None):
    if size is None:
        final_shape = []
    elif isinstance(size, (int, np.integer)):
        final_shape = [size]
    else:
        final_shape = size
    final_shape = list(final_shape[:])
    final_shape.append(mean.shape[0])

    if L is None and cov is None \
            or L is not None and cov is not None:
        raise ValueError("you must provide either cov or L (cholesky decomp result)")
    if len(mean.shape) != 1:
        raise ValueError("mean must be 1 dimensional")

    if L is not None:
        if (len(L.shape) != 2) or (L.shape[0] != L.shape[1]):
            raise ValueError("L must be 2 dimensional and square")
        if mean.shape[0] != L.shape[0]:
            raise ValueError("mean and L must have same length")

    if cov is not None:
        if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
            raise ValueError("cov must be 2 dimensional and square")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError("mean and cov must have same length")
        L = np.linalg.cholesky(cov)

    from numpy.random import standard_normal
    z = standard_normal(final_shape).reshape(mean.shape[0], -1)

    x = L.dot(z).T
    x += mean
    x.shape = tuple(final_shape)

    return x, L


def point_cloud_vs_model_err(points: np.ndarray, model) -> np.ndarray:
    faces = np.array([f[0] for f in model.faces], dtype='uint')
    vertices = np.array(model.vertices)
    errs = get_model_errors(points, vertices, faces)
    return errs


##@nb.njit(nb.f8[:](nb.f8[:, :], nb.f8[:, :]), nogil=True)
#@nb.njit(nb.f8(nb.f8[:, :], nb.f8[:, :]), nogil=True, cache=True)
def poly_line_intersect(poly, line):
    #    extend_line = True
    eps = 1e-6
    none = np.inf  # np.zeros(1)

    v0v1 = poly[1, :] - poly[0, :]
    v0v2 = poly[2, :] - poly[0, :]

    dir = line[1, :] - line[0, :]
    line_len = math.sqrt(np.sum(dir ** 2))
    if line_len < eps:
        return none

    dir = dir / line_len
    pvec = cross3d(dir, v0v2).ravel()
    det = np.dot(v0v1, pvec)
    if abs(det) < eps:
        return none

    # backface culling
    if False and det < 0:
        return none

    # frontface culling
    if False and det > 0:
        return none

    inv_det = 1.0 / det
    tvec = line[0, :] - poly[0, :]
    u = tvec.dot(pvec) * inv_det

    if u + eps < 0 or u - eps > 1:
        return none

    qvec = cross3d(tvec, v0v1).ravel()
    v = dir.dot(qvec) * inv_det

    if v + eps < 0 or u + v - eps > 1:
        return none

    t = v0v2.dot(qvec) * inv_det
    if True:
        # return error directly
        return t - line_len
    else:
        # return actual 3d intersect point
        if not extend_line and t - eps > line_len:
            return none
        return line[0, :] + t * dir


# INVESTIGATE: parallel = True does not speed up at all (or marginally) for some reason even though all cores are in use
#@nb.njit(nb.f8(nb.u4[:, :], nb.f8[:, :], nb.f8[:, :]), nogil=True, parallel=False, cache=True)
def intersections(faces, vertices, line):
    # pts = np.zeros((10, 3))
    # i = 0
    min_err = np.ones(faces.shape[0]) * np.inf
    for k in nb.prange(1, faces.shape[0]):
        err = poly_line_intersect(vertices[faces[k, :], :], line)
        min_err[k] = err
#        if abs(err) < min_err:
#            min_err = err
        # if len(pt) == 3:
        #     pts[i, :] = pt
        #     i += 1
        #     if i >= pts.shape[0]:
        #         print('too many intersects')
        #         i -= 1

    i = np.argmin(np.abs(min_err))
    return min_err[i]  # pts[0:i, :]


#@nb.jit(nb.f8[:](nb.f8[:, :], nb.f8[:, :], nb.i4[:, :]), nogil=True, parallel=False)
def get_model_errors(points, vertices, faces):
    count = len(points)
    show_progress(count // 10, 0)
    j = 0

    devs = np.empty(points.shape[0])
    for i in nb.prange(count):
        vx = points[i, :]
        err = intersections(faces, vertices, np.array(((0, 0, 0), vx)))
        if math.isinf(err):  # len(pts) == 0:
            print('no intersections!')
            continue

        if False:
            idx = np.argmin([np.linalg.norm(pt - vx) for pt in pts])
            err = np.linalg.norm(pts[idx]) - np.linalg.norm(vx)

        devs[i] = err
        if j < i // 10:
            show_progress(count // 10, i // 10)
            j = i // 10

    return devs


# class NearestKernelNDInterpolator(NearestNDInterpolator):
# -- commented out because of scipy dependency


def foreground_idxs(array, max_val=None):
    iy, ix = np.where(array < max_val)
    idxs = np.concatenate(((iy,), (ix,)), axis=0).T
    return idxs


def interp2(array, x, y, max_val=None, max_dist=30, idxs=None, discard_bg=False):
    assert y < array.shape[0] and x < array.shape[1], 'out of bounds %s: %s' % (array.shape, (y, x))

    v = array[int(y):int(y) + 2, int(x):int(x) + 2]
    xf = x - int(x)
    yf = y - int(y)
    w = np.array((
        ((1 - yf) * (1 - xf), (1 - yf) * xf),
        (yf * (1 - xf), yf * xf),
    ))

    # ignore background depths
    if max_val is not None:
        idx = v.reshape(1, -1) < max_val * 0.999
    else:
        idx = ~np.isnan(v.reshape(1, -1))

    w_sum = np.sum(w.reshape(1, -1)[idx])
    if w_sum > 0:
        # ignore background values
        val = np.sum(w.reshape(1, -1)[idx] * v.reshape(1, -1)[idx]) / w_sum

    elif discard_bg:
        return float('nan')

    else:
        # no foreground values in 2x2 matrix, find nearest foreground value
        if idxs is None:
            idxs = foreground_idxs(array, max_val)

        fallback = len(idxs) == 0
        if not fallback:
            dist = np.linalg.norm(idxs - np.array((y, x)), axis=1)
            i = np.argmin(dist)
            val = array[idxs[i, 0], idxs[i, 1]]
            # print('\n%s, %s, %s, %s, %s, %s, %s'%(v, x,y,dist[i],idxs[i,1],idxs[i,0],val))
            fallback = dist[i] > max_dist

        if fallback:
            val = np.sum(w * v) / np.sum(w)

    return val


def solve_rotation(src_q, dst_q):
    """ q*src_q*q.conj() == dst_q, solve for q """
    # based on http://web.cs.iastate.edu/~cs577/handouts/quaternion.pdf
    # and https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Pairs_of_unit_quaternions_as_rotations_in_4D_space

    # NOTE: not certain if works..

    M = np.zeros((4, 4))
    for i in range(len(src_q)):
        si = src_q[i]
        Pi = np.array((
            (si.w, -si.x, -si.y, -si.z),
            (si.x, si.w, si.z, -si.y),
            (si.y, -si.z, si.w, si.x),
            (si.z, si.y, -si.x, si.w),
        ))

        qi = dst_q[i]
        Qi = np.array((
            (qi.w, -qi.x, -qi.y, -qi.z),
            (qi.x, qi.w, -qi.z, qi.y),
            (qi.y, qi.z, qi.w, -qi.x),
            (qi.z, -qi.y, qi.x, qi.w),
        ))

        M += Pi.T * Qi

    w, v = np.linalg.eig(M)
    i = np.argmax(w)
    res_q = np.quaternion(*v[:, i])
    #    alt = v.dot(w)
    #    print('%s,%s'%(res_q, alt))
    #    res_q = np.quaternion(*alt).normalized()
    return res_q


def solve_q_bf(src_q, dst_q):
    qs = []
    d = []
    for res_q in (
            np.quaternion(0, 0, 0, 1).normalized(),
            np.quaternion(0, 0, 1, 0).normalized(),
            np.quaternion(0, 0, 1, 1).normalized(),
            np.quaternion(0, 0, -1, 1).normalized(),
            np.quaternion(0, 1, 0, 0).normalized(),
            np.quaternion(0, 1, 0, 1).normalized(),
            np.quaternion(0, 1, 0, -1).normalized(),
            np.quaternion(0, 1, 1, 0).normalized(),
            np.quaternion(0, 1, -1, 0).normalized(),
            np.quaternion(0, 1, 1, 1).normalized(),
            np.quaternion(0, 1, 1, -1).normalized(),
            np.quaternion(0, 1, -1, 1).normalized(),
            np.quaternion(0, 1, -1, -1).normalized(),
            np.quaternion(1, 0, 0, 1).normalized(),
            np.quaternion(1, 0, 0, -1).normalized(),
            np.quaternion(1, 0, 1, 0).normalized(),
            np.quaternion(1, 0, -1, 0).normalized(),
            np.quaternion(1, 0, 1, 1).normalized(),
            np.quaternion(1, 0, 1, -1).normalized(),
            np.quaternion(1, 0, -1, 1).normalized(),
            np.quaternion(1, 0, -1, -1).normalized(),
            np.quaternion(1, 1, 0, 0).normalized(),
            np.quaternion(1, -1, 0, 0).normalized(),
            np.quaternion(1, 1, 0, 1).normalized(),
            np.quaternion(1, 1, 0, -1).normalized(),
            np.quaternion(1, -1, 0, 1).normalized(),
            np.quaternion(1, -1, 0, -1).normalized(),
            np.quaternion(1, 1, 1, 0).normalized(),
            np.quaternion(1, 1, -1, 0).normalized(),
            np.quaternion(1, -1, 1, 0).normalized(),
            np.quaternion(1, -1, -1, 0).normalized(),
            np.quaternion(1, 1, 1, -1).normalized(),
            np.quaternion(1, 1, -1, 1).normalized(),
            np.quaternion(1, 1, -1, -1).normalized(),
            np.quaternion(1, -1, 1, 1).normalized(),
            np.quaternion(1, -1, 1, -1).normalized(),
            np.quaternion(1, -1, -1, 1).normalized(),
            np.quaternion(1, -1, -1, -1).normalized(),
    ):
        tq = res_q * src_q * res_q.conj()
        qs.append(res_q)
        # d.append(1-np.array((tq.w, tq.x, tq.y, tq.z)).dot(np.array((dst_q.w, dst_q.x, dst_q.y, dst_q.z)))**2)
        d.append(angle_between_q(tq, dst_q))
    i = np.argmin(d)
    return qs[i]


def hover_annotate(fig, ax, line, annotations):
    annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        idx = ind["ind"][0]
        try:
            # for regular plots
            x, y = line.get_data()
            annot.xy = (x[idx], y[idx])
        except AttributeError:
            # for scatter plots
            annot.xy = tuple(line.get_offsets()[idx])
        text = ", ".join([annotations[n] for n in ind["ind"]])
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = line.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


def plot_vectors(pts3d, scatter=True, conseq=True, neg_z=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)

    if scatter:
        ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
    else:
        if conseq:
            ax.set_prop_cycle('color', map(lambda c: '%f' % c, np.linspace(1, 0, len(pts3d))))
        for i, v1 in enumerate(pts3d):
            if v1 is not None:
                ax.plot((0, v1[0]), (0, v1[1]), (0, v1[2]))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if neg_z:
        ax.view_init(90, -90)
    else:
        ax.view_init(-90, -90)
    plt.show()


def numeric(s):
    try:
        float(s)
    except ValueError:
        return False
    return True


def pseudo_huber_loss(a, delta):
    # from https://en.wikipedia.org/wiki/Huber_loss
    # first +1e-15 is to avoid divide by zero, second to avoid loss becoming zero if delta > 1e7 due to float precision
    return delta ** 2 * (np.sqrt(1 + a ** 2 / (delta ** 2 + 1e-15)) - 1 + 1e-15)


def fixed_precision(val, precision, as_str=False):
    if val == 0:
        return ('%%.%df' % precision) % val if as_str else val
    d = math.ceil(math.log10(abs(val))) - precision
    c = 10 ** d
    fp_val = round(val / c) * c
    return ('%%.%df' % max(0, -d)) % fp_val if as_str else fp_val


def plot_quats(quats, conseq=True, wait=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if conseq:
        ax.set_prop_cycle('color', map(lambda c: '%f' % c, np.linspace(1, 0, len(quats))))
    for i, q in enumerate(quats):
        if q is not None:
            lat, lon, _ = q_to_lat_lon_roll(q)
            v1 = spherical2cartesian(lat, lon, 1)
            v2 = (v1 + normalize_v(np.cross(np.cross(v1, np.array([0, 0, 1])), v1)) * 0.1) * 0.85
            v2 = q_times_v(q, v2)
            ax.plot((0, v1[0], v2[0]), (0, v1[1], v2[1]), (0, v1[2], v2[2]))

    while (wait and not plt.waitforbuttonpress()):
        pass


def plot_poses(poses, conseq=True, wait=True, arrow_len=1):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if conseq:
        plt.hsv()
        # ax.set_prop_cycle('color', map(lambda c: '%f' % c, np.linspace(.7, 0, len(poses))))
    for i, pose in enumerate(poses):
        if pose is not None:
            q = np.quaternion(*pose[3:])
            lat, lon, _ = q_to_lat_lon_roll(q)
            v1 = spherical2cartesian(lat, lon, 1) * arrow_len
            v2 = (v1 + normalize_v(np.cross(np.cross(v1, np.array([0, 0, 1])), v1)) * 0.1 * arrow_len) * 0.85
            v2 = q_times_v(q, v2)
            ax.plot((pose[0], v1[0], v2[0]), (pose[1], v1[1], v2[1]), (pose[2], v1[2], v2[2]))

    while (wait and not plt.waitforbuttonpress()):
        pass


def show_progress(tot, i):
    digits = int(math.ceil(math.log10(tot + 1)))
    if i == 0:
        print('%s/%d' % ('0' * digits, tot), end='', flush=True)
    else:
        print(('%s%0' + str(digits) + 'd/%d') % ('\b' * (digits * 2 + 1), i + 1, tot), end='', flush=True)


def smooth1d(xt, x, Y, weight_fun=lambda d: 0.9 ** abs(d)):
    if xt.ndim != 1 or x.ndim != 1:
        raise ValueError("smooth1d only accepts 1 dimension arrays for location")
    if x.shape[0] != Y.shape[0]:
        raise ValueError("different lenght x and Y")

    D = np.repeat(np.expand_dims(xt, 1), len(x), axis=1) - np.repeat(np.expand_dims(x, 0), len(xt), axis=0)
    weights = np.array(list(map(weight_fun, D.flatten()))).reshape(D.shape)
    Yt = np.sum(Y * weights, axis=1) / np.sum(weights, axis=1)

    return Yt
