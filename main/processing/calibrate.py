#!/bin/env python3

from pathlib import PurePath

import cv2

import typing

from main.conf.path_consts import calibration_images_basepath
from main.conf.path_consts import camera_config_basepath

class Calibrator:
    def __init__(self, p_dist: int = 1):
        self.objp = np.zeros((6 * 9, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        self.objp *= p_dist
        self.mtx = None
        self.dist = None
        self.roi = None
        self.newcameramtx = None

    def img_corners_into_list(self, path: typing.Optional[str] = None, save: bool = True) -> bool:
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        img_fnames = glob.glob(str(PurePath(path, '*.jpg')))
        gray = None
        for fname in img_fnames:
            gray = cv.imread(fname, cv2.IMREAD_GRAYSCALE)

            # Find the chess board corners; size must be exact or ret will be false
            ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

            if ret:
                objpoints.append(objp)

                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_criteria)
                imgpoints.append(corners2)
            else:
                print(f'no corners found for {fname}')

        if gray is None:
            return False

        ret, self.mtx, self.dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            if save:
                if path is None:
                    path = calibration_images_basepath
                np.savetxt(str(PurePath(path, 'mtx.txt')), self.mtx)
                np.savetxt(str(PurePath(path, 'dist.txt')), self.dist)

            return True

        else:
            return False

    def prepare_undistortion(self, alpha: int = 1, path: typing.Optional[str] = None) -> None:
        if self.mtx is None or self.dist is None:
            if path is None:
                path = calibration_images_basepath
            self.mtx = np.loadtxt(str(Purepath(path, 'mtx.txt')))
            self.dist = np.loadtxt(str(Purepath(path, 'dist.txt')))
            self.newcameramtx, self.roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))

