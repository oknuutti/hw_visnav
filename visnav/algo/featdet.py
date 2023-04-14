import numpy as np
import cv2

# TODO: implement better non max suppression? e.g. like this: https://github.com/BAILOOL/ANMS-Codes


def detect_gridded(detector, img, mask, rows, cols, k, refine=True, margin=32):
    assert mask is None or img.shape[:2] == mask.shape, 'wrong size mask'
    ks = k // (rows * cols)
    if getattr(detector, 'setMaxFeatures', None) is not None:
        old_k = detector.getMaxFeatures()
        detector.setMaxFeatures(ks)

    if rows == 1 and cols == 1:
        keypoints = detector.detect(img, mask=mask)
    else:
        h, w = img.shape[:2]
        xs = np.uint32(np.rint(np.linspace(0, w, num=cols+1)))
        ys = np.uint32(np.rint(np.linspace(0, h, num=rows+1)))

        # expand xs and ys by the detector margin
        ystarts, yends = np.clip(ys[:-1] - margin, 0, None), np.clip(ys[1:] + margin, None, h)
        xstarts, xends = np.clip(xs[:-1] - margin, 0, None), np.clip(xs[1:] + margin, None, w)
        keypoints = []
        for y1, y2 in zip(ystarts, yends):
            for x1, x2 in zip(xstarts, xends):
                kps = detector.detect(img[y1:y2, x1:x2], mask=None if mask is None else mask[y1:y2, x1:x2])
                if len(kps) == 0:
                    continue

                # sort if too many
                if len(kps) > ks:  # and kps[0].response != 0.0:
                    kps = sorted(kps, key=lambda x: -x.response)
                    kps = kps[:ks]

                kp_arr = np.array([kp.pt for kp in kps], dtype=np.float32).reshape((-1, 1, 2))

                if refine:
                    win_size = (5, 5)
                    zero_zone = (-1, -1)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
                    kp_arr = cv2.cornerSubPix(img[y1:y2, x1:x2], kp_arr, win_size, zero_zone, criteria)

                kp_arr += np.array([x1, y1]).reshape((1, 1, 2))

                for i, kp in enumerate(kps):
                    kp.pt = tuple(kp_arr[i, 0, :])

                keypoints.extend(kps)

    if getattr(detector, 'setMaxFeatures', None) is not None:
        detector.setMaxFeatures(old_k)

    return keypoints

