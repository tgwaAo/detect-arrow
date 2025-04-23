#!/bin/env python3

import pathlib as pl
from time import time
import json
import traceback

import numpy as np
import cv2

from main.conf.paths import PRINTED_PATH
from main.conf.paths import CAM_CONFIG_PATH
from main.conf.imgs import ABORT_VIDEO_KEYS
from main.processing.model_handler import ModelHandler
from main.processing.utils import est_pose_in_img
from main.io.video import VideoCapture

if __name__ == '__main__':
    cam_target = 0
    cam_height = None
    cam_width = None
    time_till_update = 1
    y_location_text = 10

    cam_conf_filepath = pl.Path(CAM_CONFIG_PATH, 'cam_conf.json')
    if cam_conf_filepath.is_file():
        try:
            with open(str(cam_conf_filepath), 'r') as file:
                cam_conf = json.load(file)
            cam_target = cam_conf.get('cam_target', cam_target)
            cam_height = cam_conf.get('cam_height', cam_height)
            cam_width = cam_conf.get('cam_width', cam_width)
            time_till_update = cam_conf.get('time_till_update', time_till_update)
            y_location_text = cam_conf.get('y_location_text', y_location_text)
        except IOError as e:
            print(f'could not read file, because {str(e)}')
            print('using camera 0')
        except Exception as e:
            print(f'unexpected exeption occured: {str(e)}')
            print('using camera 0')

    printed_fname = str(pl.PurePath(PRINTED_PATH, 'coords_of_arrow.txt'))
    points_printed = np.loadtxt(printed_fname, dtype=int)
    mtx_fname = str(pl.PurePath(CAM_CONFIG_PATH, 'mtx.txt'))
    mtx = np.loadtxt(mtx_fname)

    handler = ModelHandler()
    ret = handler.load_model()
    if not ret:
        exit(1)
    model = handler.model
    handler.model = None
    model.trainable = False

    print(f'using camera target {cam_target}')
    cap = VideoCapture(cam_target, cam_width, cam_height, drop_if_full=False)
    if not cap.isOpened():
        cap.release()
        exit(1)

    abort = False
    text = ''
    pos_cnt = ()
    neg_cnt = ()
    time_start = time()
    R = None
    T = None
    pred = None
    try:
        while not abort:
            ret, img = cap.read()
            if not ret:
                print('can not receive frame -> abort')
                break

            if img is None:
                continue

            time_now = time()
            if (time_now - time_start) > 1:
                time_start = time_now
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_img = cv2.blur(gray_img, (3,3))
                result = est_pose_in_img(gray_img, model, points_printed, mtx)
                if result is not None:
                    R, T, pos_cnt, pred = result
                else:
                    R = T = pos_cnt = pred = None


            if R is not None:
                text = (
                    f'R:{np.array2string(R, precision=3, floatmode='fixed')}; '
                    f'T:{np.array2string(T, precision=3, floatmode='fixed')}; '
                    f'pred:{pred:.4f}'
                )
            elif pred is not None:
                text = (
                    f'pred:{pred:.4f}'
                )
            else:
                text = (
                    'no arrow found'
                )
            cv2.drawContours(img, [pos_cnt], -1, (255, 0, 0), 3)
            cv2.putText(
                img,
                text,
                (10, y_location_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 0)
            )
            cv2.imshow('camera test', img)
            key = cv2.waitKey(20)
            if key in ABORT_VIDEO_KEYS:
                break

    except Exception as e:
        print(traceback.format_exc())

    cap.release()
    cv2.destroyAllWindows()
    print('done')
