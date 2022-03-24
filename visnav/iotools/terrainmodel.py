import numpy as np
import cv2

import rasterio as rs
from rasterio.warp import transform
from rasterio.crs import CRS
from pygeodesy.ltp import LocalCartesian

from visnav.algo import tools
from visnav.depthmaps import nan_blur


class TerrainModel:
    def __init__(self, dem_file, orto_file, coord0, w2c, bounds=None):
        self.coord0 = coord0
        self.w2c = w2c
        self.bounds = bounds  # (top, left, bottom, right) in [0,1]
        dem = self.load_dem(dem_file)
        self.pts3d, self.px_vals, self.ground_res = self.load_orto(dem, orto_file)

    def load_dem(self, dem_file):
        # install using `conda install -c conda-forge rastrio`
        wgs84 = CRS.from_epsg(4326)

        with rs.open(dem_file) as ds:
            top, left, bottom, right = self.bounds
            top, bottom = int(ds.height * top), int(ds.height * bottom)
            left, right = int(ds.width * left), int(ds.width * right)

            xx, yy = np.meshgrid(np.arange(left, right), np.arange(top, bottom))
            alt = ds.read(1)[top:bottom, left:right]
            lon, lat = ds.xy(yy, xx)
            lon, lat, alt = transform(ds.crs, wgs84, *map(lambda x: np.array(x).flatten(), (lon, lat, alt)))

        # from https://proj.org/operations/conversions/topocentric.html
        # should work since proj version 8, gives errors here though
        #
        # lc = CRS.from_proj4(('cct -d 3 +proj=pipeline +step +proj=cart +ellps=WGS84 '
        #                     + '+step +proj=topocentric +ellps=WGS84 +lat_0=%f +lon_0=%f +h_0=%f') % self.coord0)

        # now going first to wgs84 and then using pygeodesy, which is very slow but luckily need to do only once
        lc = LocalCartesian(*self.coord0)
        dem = np.zeros((len(lon), 3), dtype=np.float32)
        for i, (lat_, lon_, alt_) in enumerate(zip(lat, lon, alt)):
            xyz = lc.forward(lat_, lon_, alt_)
            dem[i, :] = xyz.xyz

        # transform to correct coordinate frame (now x east, y north, z up => x east, y south, z down)
        dem = tools.q_times_mx(self.w2c.quat.conj(), dem)
        dem = dem.reshape((bottom - top, right - left, 3))
        return dem

    def load_orto(self, dem, orto_file):
        H, W, _ = dem.shape

        with rs.open(orto_file) as fh:
            img = fh.read()
        img = np.moveaxis(img, 0, -1)
        h, w, _ = img.shape

        top, left, bottom, right = self.bounds
        top, bottom = int(h * top), int(h * bottom)
        left, right = int(w * left), int(w * right)
        img = img[top:bottom, left:right, :]
        h, w = bottom - top, right - left

        if h > H or w > W:
            dem = cv2.resize(dem, (w, h))
            H, W, _ = dem.shape
        elif h < H or w < W:
            img = cv2.resize(img, (W, H))
            h, w, _ = img.shape
        assert (h, w) == (H, W), 'img and dem are different sizes: (%d, %d) == (%d, %d)' % (h, w, H, W)
        # h, w = min(h, H), min(w, W)
        # img, dem = img[:h, :w, :], dem[:h, :w, :]

        ground_res = (np.mean(np.abs(np.diff(dem[:, :, 0], axis=1))) + np.mean(np.abs(np.diff(dem[:, :, 1], axis=0)))) / 2
        return dem.reshape((-1, 3)), img.reshape((-1, 3)), ground_res

    def project(self, cam, pose):
        # project 3d points to 2d, filter out non-visible
        q_cf = tools.angleaxis_to_q(pose[:3])
        pts3d_cf = tools.q_times_mx(q_cf, self.pts3d) + pose[3:]
        I = pts3d_cf[:, 2] > 0   # in front of cam

        # project without distortion first to remove far away points that would be warped into the image
        uvph = cam.cam_mx.dot(pts3d_cf.T).T
        pts2d = uvph[:, :2] / uvph[:, 2:]
        margin = cam.width * 0.2
        I = np.logical_and.reduce((I, pts2d[:, 0] >= -margin, pts2d[:, 0] < cam.width + margin,
                                      pts2d[:, 1] >= -margin, pts2d[:, 1] < cam.height + margin))

        if np.sum(I) > 0:
            pts2d = np.atleast_2d(cam.project(pts3d_cf[I, :].astype(np.float32)) + 0.5).astype(int)

            J = np.logical_and.reduce((pts2d[:, 0] >= 0, pts2d[:, 0] < cam.width,
                                       pts2d[:, 1] >= 0, pts2d[:, 1] < cam.height))
            pts2d = pts2d[J, :]
            px_vals = self.px_vals[I, :][J, :]

            # calculate suitable blob radius
            median_dist = np.median(np.linalg.norm(pts3d_cf[I, :], axis=1))
            radius = round((self.ground_res/2 / median_dist) / (1 / cam.cam_mx[0, 0]))  # or ceil?
            diam = radius * 2 + 1
        else:
            diam, pts2d, px_vals = 1, np.zeros((0, 2), dtype=int), np.zeros((0, 3), dtype=int)

        # draw image
        image = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        image[pts2d[:, 1], pts2d[:, 0], :] = px_vals
        if diam >= 3:
            image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diam, diam)))

        # fill in gaps
        image = image.reshape((-1, 3)).astype(np.float32)
        image[np.all(image == 0, axis=1), :] = np.nan
        image = nan_blur(image.reshape((cam.height, cam.width, 3)), (5, 5), onlynans=True).astype(np.uint8)

        return image
