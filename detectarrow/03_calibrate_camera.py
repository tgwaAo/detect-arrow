#!/bin/env python3

from glob import glob
from pathlib import PurePath

import cv2
import numpy as np

from conf.paths import CALIB_IMGS_PATH
from processing.calibrate import Calibrator

if __name__ == '__main__':
    calib = Calibrator()
    calib.read_printed_nbrs()
    if not calib.read_imgs_and_calib_cam():
        exit(1)
    calib.prepare_undistortion()
    img_fname = glob(str(PurePath(CALIB_IMGS_PATH, '*.jpg')))[0]
    img = cv2.imread(img_fname)
    if img is None:
        exit(2)
    result = calib.undistort(img)
    combined = np.hstack((img, result))
    cv2.imshow('calibration', combined)

    merged = cv2.addWeighted(img, 0.5, result, 0.5, 0)
    cv2.imshow('merged', merged)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    result = calib.error()
    if result is not None:
        print(f'total error: {result}')
    else:
        print('error could not be calculated')
