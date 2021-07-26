import logging

import numpy as np
import quaternion
import cv2

from visnav.algo.odo.base import VisualOdometry
from visnav.algo.image import ImageProc


class VisualGPSNav(VisualOdometry):
    def check_features(self, image, in_kp2d):
        a_mask = self._feature_detection_mask(image)
        mask = np.ones((len(in_kp2d),), dtype=np.bool)

        h, w = image.shape[:2]
        for i, pt in enumerate(in_kp2d):
            d = 2
            y0, y1 = max(0, int(pt[0, 1]) - d), min(h, int(pt[0, 1]) + d)
            x0, x1 = max(0, int(pt[0, 0]) - d), min(w, int(pt[0, 0]) + d)

            if x1 <= x0 or y1 <= y0 \
                    or np.max(image[y0:y1, x0:x1]) < self.min_feature_intensity \
                    or np.min(image[y0:y1, x0:x1]) > self.max_feature_intensity \
                    or a_mask[min(int(pt[0, 1]), h - 1), min(int(pt[0, 0]), w - 1)] == 0:
                mask[i] = False

        return mask

    def _feature_detection_mask(self, image):
        _, mask = cv2.threshold(image, self.min_feature_intensity, 255, cv2.THRESH_BINARY)
        kernel = ImageProc.bsphkern(round(6*image.shape[0]/512)*2 + 1)

        # exclude asteroid limb from feature detection
        mask = cv2.erode(mask, ImageProc.bsphkern(7), iterations=1)    # remove stars
        mask = cv2.dilate(mask, kernel, iterations=1)   # remove small shadows inside asteroid
        mask = cv2.erode(mask, kernel, iterations=2)    # remove asteroid limb

        # exclude overexposed parts
        _, mask_oe = cv2.threshold(image, self.max_feature_intensity, 255, cv2.THRESH_BINARY)
        mask_oe = cv2.dilate(mask_oe, kernel, iterations=1)
        mask_oe = cv2.erode(mask_oe, kernel, iterations=1)
        mask[mask_oe > 0] = 0

        if 0:
            cv2.imshow('mask', ImageProc.overlay_mask(image, mask))
            cv2.waitKey()

        return mask
