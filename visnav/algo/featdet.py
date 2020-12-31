import numpy as np


# TODO: implement better non max suppression? e.g. like this: https://github.com/BAILOOL/ANMS-Codes


def detect_gridded(detector, img, mask, rows, cols, k):
    assert mask is None or img.shape[:2] == mask.shape, 'wrong size mask'
    ks = k // (rows * cols)
    if getattr(detector, 'setMaxFeatures', None) is not None:
        old_k = detector.getMaxFeatures()
        detector.setMaxFeatures(ks)

    h, w = img.shape[:2]
    xs = np.uint32(np.rint(np.linspace(0, w, num=cols+1)))
    ys = np.uint32(np.rint(np.linspace(0, h, num=rows+1)))
    ystarts, yends = ys[:-1], ys[1:]
    xstarts, xends = xs[:-1], xs[1:]
    keypoints = []
    for y1, y2 in zip(ystarts, yends):
        for x1, x2 in zip(xstarts, xends):
            kps = detector.detect(img[y1:y2, x1:x2], mask=None if mask is None else mask[y1:y2, x1:x2])
            if len(kps) == 0:
                continue

            # sort if too many
            if len(kps) > ks:  # and kps[0].response != 0.0:
                kps = sorted(kps, key=lambda x: -x.response)

            for kp in kps[:ks]:
                kp.pt = kp.pt[0]+x1, kp.pt[1]+y1

            keypoints.extend(kps[:ks])

    if getattr(detector, 'setMaxFeatures', None) is not None:
        detector.setMaxFeatures(old_k)

    return keypoints

