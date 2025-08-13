from pathlib import PurePath
from glob import glob
import json

import cv2
import numpy as np

import numpy.typing as npt
from typing import Optional as Opt

from conf.paths import CALIB_IMGS_PATH
from conf.paths import CAM_CONFIG_PATH
from conf.paths import PRINTED_MEASUREMENT_FNAME
from conf.imgs import TERM_CRITERIA


class Calibrator:
    def __init__(self) -> None:
        self.obj_points: list[npt.NDArray[np.float32]] = []
        self.img_points: list[npt.NDArray[np.floating]] = []
        self.mtx: Opt[npt.NDArray[np.floating]] = None
        self.dist: Opt[npt.NDArray[np.floating]] = None
        self.r_vecs: Opt[npt.NDArray[np.floating]] = None
        self.t_vecs: Opt[npt.NDArray[np.floating]] = None
        self.roi = None
        self.new_camera_mtx: Opt[npt.NDArray[np.floating]] = None
        self.p_dist: Opt[int] = None
        self.width: Opt[int] = None
        self.height: Opt[int] = None

    def read_printed_nbrs(self, fname: str = PRINTED_MEASUREMENT_FNAME) -> None:
        with open(fname, 'r') as file:
            calib_values = json.load(file)

        self.p_dist = calib_values.get('distance_between_points', None)
        self.width = calib_values.get('width', None)
        self.height = calib_values.get('height', None)

    def read_imgs_and_calib_cam(
        self,
        p_dist: int | None = None,
        size: Opt[tuple[int, int]] = None,
        imgs_path: str = CALIB_IMGS_PATH,
        config_path: str = CAM_CONFIG_PATH,
        save: bool = True
    ) -> bool:
        if p_dist is not None:
            self.p_dist = p_dist
        elif self.p_dist is None:
            raise ValueError('distance between points is missing')

        if size is not None:
            self.width, self.height = size
        elif self.width is None or self.height is None:
            raise ValueError('width and height must get a value')

        objp = np.zeros((self.height * self.width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.width, 0:self.height].T.reshape(-1, 2)
        objp *= self.p_dist

        img_fnames = glob(str(PurePath(imgs_path, '*.jpg')))
        gray = None
        for fname in img_fnames:
            gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

            # Find the chess board corners;
            # size must be exact or ret will be false
            ret, corners = cv2.findChessboardCorners(
                gray,
                (self.width, self.height),
                None
            )

            if ret:
                self.obj_points.append(objp)
                corners2 = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    TERM_CRITERIA
                )
                self.img_points.append(corners2)
            else:
                print(f'no corners found for {fname}')

        if gray is None:
            return False

        (
            ret,
            self.mtx,
            self.dist,
            self.r_vecs,
            self.t_vecs
        ) = cv2.calibrateCamera(
            self.obj_points,
            self.img_points,
            gray.shape[::-1],
            None,
            None
        )

        if not ret:
            return False

        if save:
            np.savetxt(str(PurePath(config_path, 'mtx.txt')), self.mtx)
            np.savetxt(str(PurePath(config_path, 'dist.txt')), self.dist)
        return True

    def prepare_undistortion(
        self,
        alpha: int = 1,
        config_path: str = CAM_CONFIG_PATH
    ) -> None:
        if self.mtx is None or self.dist is None:
            self.mtx = np.loadtxt(str(PurePath(config_path, 'mtx.txt')))
            self.dist = np.loadtxt(str(PurePath(config_path, 'dist.txt')))

        self.new_camera_mtx, self.roi = cv2.getOptimalNewCameraMatrix(
            self.mtx,
            self.dist,
            (self.width, self.height),
            alpha,
            (self.width, self.height))

    def undistort(
        self,
        img: npt.NDArray[np.uint8]
    ) -> Opt[npt.NDArray[np.uint8]]:
        if (
            self.mtx is not None
            and self.dist is not None
            and self.new_camera_mtx is not None
        ):
            return cv2.undistort(
                img,
                self.mtx,
                self.dist,
                None,
                self.new_camera_mtx
            )
        else:
            return None

    def error(self) -> Opt[float]:
        if (
            self.obj_points is not None and
            self.mtx is not None and
            self.dist is not None and
            self.r_vecs is not None and
            self.t_vecs is not None
        ):
            error_sum = 0
            for i in range(len(self.obj_points)):
                img_points2, _ = cv2.projectPoints(
                    self.obj_points[i],
                    self.r_vecs[i],
                    self.t_vecs[i],
                    self.mtx,
                    self.dist
                )
                error = (
                    cv2.norm(self.img_points[i], img_points2, cv2.NORM_L2)
                    / len(img_points2)
                )
                error_sum += error

            return error_sum / len(self.obj_points)
        else:
            return None
