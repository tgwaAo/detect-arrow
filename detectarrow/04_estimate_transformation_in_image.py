#!/bin/env python3

from glob import glob
import pathlib as pl
import traceback

import numpy as np
import cv2

from conf.paths import EXAMPLES_PATH
from conf.paths import PRINTED_PATH
from conf.paths import PRINTED_BNAME
from conf.paths import CAM_CONFIG_PATH
from conf.imgs import BLUR_KERNEL
from processing.model_handler import ModelHandler
from processing.utils import est_poses_in_img
from processing.utils import sort_pt_biggest_dist_y


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

    printed_fname = str(pl.PurePath(PRINTED_PATH, PRINTED_BNAME))
    points_printed = np.loadtxt(printed_fname, dtype=int)
    points_printed = sort_pt_biggest_dist_y(points_printed, False, points_printed)

    mtx_fname = str(pl.PurePath(CAM_CONFIG_PATH, 'mtx.txt'))
    mtx = np.loadtxt(mtx_fname)

    hsv_pixel = np.ones((1, 1, 3), np.uint8) * 240
    low_color_val = 0
    high_color_val = 180
    rnd_gentr = np.random.default_rng()
    darker_part_pt = (-30, 0, 30)
    np.set_printoptions(suppress=True)

    for img_fname in pl.Path(EXAMPLES_PATH).glob('*.jpg'):
        img = cv2.imread(str(img_fname))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.blur(gray_img, BLUR_KERNEL)
        text = ''
        try:
            result = est_poses_in_img(gray_img, model, points_printed, mtx, verbose=True)
            if result is None:  # "20_deg_y.jpg" is to far away/small.
                print(f'could not get any information from {img_fname}')
                continue

        except Exception as e:
            print(traceback.format_exc())
            exit(1)

        y_location = img.shape[0]
        for res in result:
            if res is None:
                continue

            R, T, cnt, pred, hull_pts = res
            if R is not None:
                text = (
                    f'R:{np.array2string(R, precision=3, floatmode='fixed')}; '
                    f'T:{np.array2string(T, precision=3, floatmode='fixed')}; '
                    f'pred:{pred:.4f}'
                )

            hsv_pixel[0, 0, 0] = rnd_gentr.integers(low_color_val, high_color_val, dtype=np.uint8)
            color = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0,0].astype(object)

            cv2.drawContours(img, [cnt], -1, color, 1)

            for idx, pt in enumerate(hull_pts):
                pt = pt.astype(int)
                cv2.circle(img, pt, 2, color + darker_part_pt, -1)
                cv2.putText(img, str(idx), pt + (5, -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color + darker_part_pt)

            y_location -= 15
            cv2.putText(
                img,
                text,
                (10, y_location),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color
            )

        cv2.imshow(str(img_fname), img)

    _ = cv2.waitKey(0)
    cv2.destroyAllWindows()
