#!/bin/env python3

import glob

import cv2
import numpy as np

from main.conf.path_consts import calibration_images_basepath
from main.processing.calibrate import Calibrator

if __name__ == '__main__':
    calib = Calibrator()
    if not calib.img_corners_into_list():
        exit(1)
    calib.prepare_undistortion()
    img_fname = glob.glob(str(PurePath(path, '*.jpg')))[0]
    img = cv2.imread(img_fname)
    if img is None:
        exit(1)
    result = calib.undistort(img)
    combined = np.hstack((img, result))
    cv2.imshow('calibration', combined)
    cv2.waitKey(0)

    merged = cv.addWeighted(img, 0.5, result, 0.5, 0)
    cv.imshow('merged', merged)
    cv.waitKey(0)

    cv.destroyAllWindows()

    result = calib.error()
    if result is not None:
        print(f'total error: {result}')
    else:
        print('error could not be calculated')
