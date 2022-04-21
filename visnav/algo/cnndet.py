import os
import math

import numpy as np
import cv2

from navex.extract import Extractor
try:
    from navex.models.r2d2orig import R2D2
    r2d2_imported = True
except:
    r2d2_imported = False

from visnav.algo import tools


class CNN_Detector:
    def __init__(self, type='r2d2', model_path=None, max_feats=2000, gpu=False):
        self.type = type
        if self.type == 'r2d2':
            assert r2d2_imported, 'Type r2d2 was selected, however, import of the original r2d2 codebase failed'
            model_path = R2D2.DEFAULT_MODEL
        else:
            assert type == 'own', 'feature of type "%s" not supported' % (type,)
            assert os.path.exists(model_path), 'no file found at %s' % model_path

        self.extractor = Extractor(model_path, gpu=gpu, top_k=max_feats, border=None, feat_d=0.001, scale_f=2**(1/4),
                                   min_size=256, max_size=1024, min_scale=0.0, max_scale=1.0, det_lim=0.7, qlt_lim=0.7)

        self._latest_img = None
        self._latest_desc = None

    def getMaxFeatures(self):
        return self.extractor.top_k

    def setMaxFeatures(self, max_feats):
        self.extractor.top_k = max_feats

    def detect(self, img, mask=None):
        kps, desc = self.detectAndCompute(img, mask)
        self._latest_img = img
        self._latest_desc = desc
        return kps

    def compute(self, img, kps):
        assert self._latest_img is img and len(kps) == len(self._latest_desc), \
            'different image or keypoints than what was calculated during detect'
        return self._latest_desc

    def detectAndCompute(self, img, mask=None):
        assert mask is None, 'mask support not implemented'
        kps, desc, score = self.extractor.extract(img)
        kps = [cv2.KeyPoint(x, y, s, 0, r) for (x, y, s), r in zip(kps, score)]
        return kps, desc

    def normalizeDetectAndCompute(self, img, sc_q):
        angle = north_to_up(sc_q)
        r_img = rotate_image(img, angle)
        kps, desc = self.detectAndCompute(r_img)
        kps, desc = rotate_keypoints(kps, desc, img.shape[1], img.shape[0], angle, self.extractor.border)
        return kps, desc


def north_to_up(sc_q):
    # assume -y axis is towards north, camera borehole is along +z axis, and cam up is towards -y axis
    north_v, cam_axis, cam_up = np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([0, -1, 0])

    # north vector in image frame
    sc_north = tools.q_times_v(sc_q, north_v)

    # project to image plane
    img_north = tools.vector_rejection(sc_north, cam_axis)

    # calculate angle between projected north vector and image up
    angle = tools.angle_between_v(cam_up, img_north, direction=cam_axis)

    return angle


def rotate_image(img, angle, fullsize=False):
    img = np.array(img).squeeze()
    h, w = img.shape[:2]

    if fullsize:
        rw = int((h * abs(math.sin(angle))) + (w * abs(math.cos(angle))))
        rh = int((h * abs(math.cos(angle))) + (w * abs(math.sin(angle))))
    else:
        rw, rh = w, h

    cx, cy = w / 2, h / 2
    mx = cv2.getRotationMatrix2D((cx, cy), math.degrees(angle), 1)
    mx[0, 2] += rw / 2 - cx
    mx[1, 2] += rh / 2 - cy
    rimg = cv2.warpAffine(img, mx, (rw, rh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    return rimg


def rotate_keypoints(kps, descs, w, h, angle, margin):
    kps_arr = np.array([kp.pt for kp in kps]) - np.array([[w/2, h/2]])
    rot_mx = np.array([[math.cos(angle), -math.sin(angle)],
                       [math.sin(angle),  math.cos(angle)]])

    kps_arr = rot_mx.dot(kps_arr.T).T + np.array([[w/2, h/2]])
    for kp, pt in zip(kps, kps_arr):
        kp.pt = tuple(pt)

    I = np.array([kp.pt[0] > margin and kp.pt[1] > margin and kp.pt[0] < w-margin and kp.pt[1] < h-margin
                  for kp in kps], dtype=np.bool)
    return np.array(kps)[I], descs[I, :]
