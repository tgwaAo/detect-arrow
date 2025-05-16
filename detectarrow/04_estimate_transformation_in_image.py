#!/bin/env python3

from glob import glob
import pathlib as pl
import traceback

import numpy as np
import cv2

from conf.paths import EXAMPLES_PATH
from conf.paths import PRINTED_PATH
from conf.paths import CAM_CONFIG_PATH
from processing.model_handler import ModelHandler
from processing.utils import est_poses_in_img


if __name__ == '__main__':
    model_handler = ModelHandler()
    try:
        model_handler.load_model()
    except ValueError:
        print('could not load model')
        exit(1)
    except Exception as e:
        print(e)
        exit(2)

    model = model_handler.model
    model_handler.model = None
    model.trainable = False

    printed_fname = str(pl.PurePath(PRINTED_PATH, 'coords_of_arrow.txt'))
    points_printed = np.loadtxt(printed_fname, dtype=int)
    mtx_fname = str(pl.PurePath(CAM_CONFIG_PATH, 'mtx.txt'))
    mtx = np.loadtxt(mtx_fname)

    hsv_pixel = np.ones((1, 1, 3), np.uint8) * 240
    low_color_val = 0
    high_color_val = 180
    rnd_gentr = np.random.default_rng()

    np.set_printoptions(suppress=True)

    for img_fname in glob(str(pl.PurePath(EXAMPLES_PATH, '*'))):
        img = cv2.imread(img_fname)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.blur(gray_img, (3,3))
        text = ''
        try:
            result = est_poses_in_img(gray_img, model, points_printed, mtx, verbose=True)
            if result is None:
                print(f'could not get any information from {img_fname}')
                continue

        except Exception as e:
            print(traceback.format_exc())
            exit(1)

        y_location = img.shape[0]
        for res in result:
            if res is None:
                continue

            R, T, cnt, pred = res
            text = (
                f'R:{np.array2string(R, precision=3, floatmode='fixed')}; '
                f'T:{np.array2string(T, precision=3, floatmode='fixed')}; '
                f'pred:{pred:.4f}'
            )
            hsv_pixel[0, 0, 0] = rnd_gentr.integers(low_color_val, high_color_val, dtype=np.uint8)
            color = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0,0].astype(object)
            cv2.drawContours(img, [cnt], -1, color, 2)
            y_location -= 15
            cv2.putText(
                img,
                text,
                (10, y_location),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color
            )

        cv2.imshow(img_fname, img)

    _ = cv2.waitKey(0)
    cv2.destroyAllWindows()
