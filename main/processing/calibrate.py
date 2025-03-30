from pathlib import PurePath
from glob import glob
import json

import cv2
import numpy as np

import typing
import numpy.typing as npt

from main.conf.paths import CALIB_IMGS_PATH
from main.conf.paths import CAM_CONFIG_PATH
from main.conf.paths import PRINTED_MEASUREMENT_FNAME
from main.conf.imgs import TERM_CRITERIA

class Calibrator:
    def __init__(self, p_dist: int = 1):
        self.obj_points = []
        self.img_points = []
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.roi = None
        self.newcameramtx = None
        self.p_dist = None
        self.width = None
        self.height = None

    def read_printed_nbrs(self, fname: str = PRINTED_MEASUREMENT_FNAME):
        with open(fname, 'r') as file:
            calib_values = json.load(file)

        self.p_dist = calib_values.get('distance_between_points', None)
        self.width = calib_values.get('width', None)
        self.height = calib_values.get('height', None)

    def img_corners_into_list(
        self,
        p_dist: int | None = None,
        size: tuple[int, int] | None = None,
        path: str = CALIB_IMGS_PATH,
        save: bool = True
    ) -> bool:
        if p_dist is not None:
            self.p_dist = p_dist

        if size is not None:
            self.width, self.height = size

        objp = np.zeros((self.height * self.width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.width, 0:self.height].T.reshape(-1, 2)
        objp *= self.p_dist

        img_fnames = glob(str(PurePath(path, '*.jpg')))
        gray = None
        for fname in img_fnames:
            gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

            # Find the chess board corners; size must be exact or ret will be false
            ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)

            if ret:
                self.obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), TERM_CRITERIA)
                self.img_points.append(corners2)
            else:
                print(f'no corners found for {fname}')

        if gray is None:
            return False

        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, gray.shape[::-1], None, None)
        if ret:
            if save:
                np.savetxt(str(PurePath(path, 'mtx.txt')), self.mtx)
                np.savetxt(str(PurePath(path, 'dist.txt')), self.dist)

            return True

        else:
            return False

    def prepare_undistortion(self, alpha: int = 1, path: str = CALIB_IMGS_PATH) -> None:
        if self.mtx is None or self.dist is None:
            self.mtx = np.loadtxt(str(Purepath(path, 'mtx.txt')))
            self.dist = np.loadtxt(str(Purepath(path, 'dist.txt')))

        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
            self.mtx,
            self.dist,
            (self.width, self.height),
            alpha,
            (self.width, self.height))

    def undistort(self, img: npt.NDArray[np.uint8]) -> typing.Optional[npt.NDArray[np.uint8]]:
        if self.mtx is not None and self.dist is not None and self.newcameramtx is not None:
            return cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
        else:
            return None

    def compare(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        undistorted = self.undistort(img)
        return cv2.addWeighted(img, 0.5, undistorted, 0.5, 0)

    def error(self):
        if (
            self.obj_points is not None and
            self.mtx is not None and
            self.dist is not None and
            self.rvecs is not None and
            self.tvecs is not None
        ):
            mean_error = 0
            for i in range(len(self.obj_points)):
                img_points2, _ = cv2.projectPoints(
                    self.obj_points[i],
                    self.rvecs[i],
                    self.tvecs[i],
                    self.mtx,
                    self.dist
                )
                error = cv2.norm(self.img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
                mean_error += error

            return mean_error / len(self.obj_points)
        else:
            return None

