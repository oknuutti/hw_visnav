import os
from typing import List, Dict
import shutil

import numpy as np
import cv2

import kapture as kt
from kapture import CameraType
from kapture.io.csv import kapture_from_dir, kapture_to_dir
from kapture.io.features import get_keypoints_fullpath, image_keypoints_to_file
from kapture.io.records import get_record_fullpath

from visnav.algo.odo.base import Frame, Keypoint
from visnav.algo.model import Camera


class KaptureIO:
    _IMAGE_FOLDER = 'cam'
    IMG_FORMAT_PNG = 'png'
    IMG_FORMAT_JPG = 'jpg'

    def __init__(self, path, reset=False, img_format=IMG_FORMAT_JPG, min_pt3d_obs=3, min_pt3d_ratio=0.2,
                 jpg_qlt=95, scale=1.0):
        self.path = path
        self.default_cam = ('1', self._IMAGE_FOLDER)
        self.default_kp_type = 'gftt'
        self.img_format = img_format
        self.min_pt3d_obs = min_pt3d_obs
        self.min_pt3d_ratio = min_pt3d_ratio
        self.jpg_qlt = jpg_qlt
        self.scale = scale

        if os.path.exists(self.path):
            self.kapture = kapture_from_dir(self.path)
            if reset:
                shutil.rmtree(self.path)

        if not os.path.exists(self.path):
            self.kapture = kt.Kapture()

    def write_to_dir(self):
        kapture_to_dir(self.path, self.kapture)

    def set_camera(self, id, name, cam: Camera):
        self.default_cam = ('%s' % id, name)
        
        if self.kapture.sensors is None:
            self.kapture.sensors = kt.Sensors()

        mx = cam.cam_mx
        sc = self.scale
        params = [cam.width*sc, cam.height*sc, mx[0, 0]*sc, mx[1, 1]*sc, mx[0, 2]*sc, mx[1, 2]*sc] + [0.]*8
        if cam.dist_coefs is not None:
            for i, c in enumerate(cam.dist_coefs):
                params[6 + i] = c

        self.kapture.sensors[self.default_cam[0]] = kt.Camera(CameraType.FULL_OPENCV, camera_params=params, name=name)

    def add_frames(self, frames: List[Frame], points3d: List[Keypoint]):
        k = self.kapture
        
        if k.records_camera is None:
            k.records_camera = kt.RecordsCamera()
        if k.trajectories is None:
            k.trajectories = kt.Trajectories()
        if k.keypoints is None:
            k.keypoints = {self.default_kp_type: kt.Keypoints(self.default_kp_type, np.float32, 2)}
        if k.points3d is None:
            k.points3d = kt.Points3d()
        if k.observations is None:
            k.observations = kt.Observations()

        def check_kp(kp):
            return kp.inlier_count > self.min_pt3d_obs and kp.inlier_count/kp.total_count > self.min_pt3d_ratio

        kp_ids, pts3d = zip(*[(kp.id, kp.pt3d) for kp in points3d if check_kp(kp)])
        I = np.argsort(kp_ids)
        pt3d_ids = dict(zip(np.array(kp_ids)[I], np.arange(len(I))))
        pt3d_arr = np.array(pts3d)[I, :]
        k.points3d = kt.Points3d(np.concatenate((pt3d_arr, np.ones_like(pt3d_arr)*128), axis=1))

        for f in frames:
            if not f.pose.post:
                continue

            id = f.frame_num
            img = f.orig_image
            img_file = os.path.join(self.default_cam[1], 'frame%06d.%s' % (id, self.img_format))
            img_fullpath = get_record_fullpath(self.path, img_file)
            os.makedirs(os.path.dirname(img_fullpath), exist_ok=True)

            if not np.isclose(self.scale, 1.0):
                img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            if self.img_format == self.IMG_FORMAT_PNG:
                cv2.imwrite(img_fullpath, img, (cv2.IMWRITE_PNG_COMPRESSION, 9))
            elif self.img_format == self.IMG_FORMAT_JPG:
                cv2.imwrite(img_fullpath, img, (cv2.IMWRITE_JPEG_QUALITY, self.jpg_qlt))
            else:
                assert False, 'Invalid image format: %s' % (self.img_format,)

            record_id = (id, self.default_cam[0])
            k.records_camera[record_id] = img_file

            pose = f.pose.post if 1 else (-f.pose.post)
            k.trajectories[record_id] = kt.PoseTransform(r=pose.quat.components, t=pose.loc)
            k.keypoints[self.default_kp_type].add(img_file)

            uvs = np.zeros((len(f.kps_uv), 2), np.float32)
            i = 0
            for kp_id, uv in f.kps_uv.items():
                if kp_id in pt3d_ids:
                    k.observations.add(int(pt3d_ids[kp_id]), self.default_kp_type, img_file, i)
                    uvs[i, :] = uv / f.img_sc * self.scale
                    i += 1

            image_keypoints_to_file(get_keypoints_fullpath(self.default_kp_type, self.path, img_file), uvs[:i, :])
