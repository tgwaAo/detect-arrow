#!/bin/env python3

import glob
from pathlib import PurePath
import traceback

import cv2
import numpy

from main.conf.path_consts import examples_basepath
from main.conf.path_consts import printed_basepath
from main.processing.model_handler import ModelHandler
from main.processing.utils import est_pose_in_img

if __name__ == '__main__':
    handler = ModelHandler()
    ret = handler.load_model()
    if not ret:
        exit(1)
    model = handler.get_model()
    handler.unref_model()
    model.trainable = False

    printed_fname = str(PurePath(printed_basepath, 'coords_of_arrow.txt'))
    points_printed = np.loadtxt(printed_fname, dtype=int)

    for img_fname in glob.glob(str(PurePath(examples_basepath, '*'))):
        img = cv2.imread(img_fname)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = ''
        try:
            result = est_pose_in_img(gray_img, model, points_printed)
            if result is None:
                continue
            R, T, cnt, pred = result
            text = (
                f'R:{np.array2string(R, precision=3, floatmode='fixed')};'
                f' T:{np.array2string(T, precision=3, floatmode='fixed')}'
            )
            cv2.drawContours(img, -1, [cnt], (255, 0, 0), 2)
        except Exception as e:
            print(traceback.format_exc())
            exit(1)

        cv2.putText(img, text, (10, y_location), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

