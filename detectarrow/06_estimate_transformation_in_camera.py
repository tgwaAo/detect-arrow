#!/bin/env python3

import pathlib as pl
from time import time
import json
import traceback

import numpy as np
import cv2

import numpy.typing as npt
from typing import Optional as Opt
from typing import Union

from conf.paths import PRINTED_PATH
from conf.paths import PRINTED_BNAME
from conf.paths import CAM_CONFIG_PATH
from conf.paths import CAM_CONFIG_BNAME
from conf.imgs import ABORT_VIDEO_KEYS
from conf.imgs import BLUR_KERNEL
from processing.model_handler import ModelHandler
from processing.utils import est_pose_in_img
from processing.utils import sort_pt_biggest_dist_y
from inout.video import VideoCapture

if __name__ == '__main__':
    cam_target: Union[str, int] = 0
    cam_height = 480
    cam_width = 640
    time_till_update = 1
    y_location_text = 470

    cam_conf_filepath = pl.Path(CAM_CONFIG_PATH, CAM_CONFIG_BNAME)
    if cam_conf_filepath.is_file():
        try:
            with open(str(cam_conf_filepath), 'r') as file:
                cam_conf = json.load(file)
            cam_target = cam_conf.get('cam_target', cam_target)
            cam_height = cam_conf.get('cam_height', cam_height)
            cam_width = cam_conf.get('cam_width', cam_width)
            time_till_update = cam_conf.get(
                'time_till_update',
                time_till_update
            )
            y_location_text = cam_conf.get('y_location_text', y_location_text)
        except IOError as e:
            print(f'could not read file, because {str(e)}')
            print('using camera 0')
        except Exception as e:
            print(f'unexpected exeption occured: {str(e)}')
            print('using camera 0')

    printed_fname = str(pl.PurePath(PRINTED_PATH, PRINTED_BNAME))
    points_printed = np.loadtxt(printed_fname, dtype=int)
    points_printed = sort_pt_biggest_dist_y(
        points_printed,
        False,
        points_printed
    )

    mtx_fname = str(pl.PurePath(CAM_CONFIG_PATH, 'mtx.txt'))
    mtx = np.loadtxt(mtx_fname)

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

    print(f'using camera target {cam_target}')
    if isinstance(cam_target, str):
        drop_if_full = False
    else:
        drop_if_full = True

    cap = VideoCapture(
        cam_target,
        cam_width,
        cam_height,
        drop_if_full=drop_if_full
    )
    if not cap.is_opened():
        cap.release()
        exit(1)

    abort = False
    color = (255, 0, 0)
    text = ''
    cnt = ()
    neg_cnt = ()
    time_start = time()
    R = None
    T = None
    pred = None
    hull_pts = None
    try:
        while not abort:
            ret, img = cap.read()
            if not ret:
                print('can not receive frame -> abort')
                break

            if img is None:
                continue

            time_now = time()
            if (time_now - time_start) > time_till_update:
                time_start = time_now
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_img = cv2.blur(gray_img, BLUR_KERNEL)
                result = est_pose_in_img(gray_img, model, points_printed, mtx)
                if result is not None:
                    R, T, cnt, pred, hull_pts = result
                else:
                    R = T = cnt = pred = hull_pts = None  # type:ignore

            if R is not None:
                text = (
                    f'R:{np.array2string(R, precision=3, floatmode='fixed')}; '
                    f'T:{np.array2string(
                        T,  # type: ignore
                        precision=3,
                        floatmode='fixed'
                    )}; '
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
            cv2.putText(
                img,
                text,
                (10, y_location_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 0)
            )

            if cnt is not None:
                cv2.drawContours(img, [cnt], -1, (255, 0, 0), 1)

            if hull_pts is not None:
                for idx, pt in enumerate(hull_pts):
                    pt = pt.astype(int)
                    cv2.circle(img, pt, 2, color, -1)
                    cv2.putText(
                        img,
                        str(idx),
                        pt + (5, -5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color
                    )

            cv2.imshow('camera test', img)
            key = cv2.waitKey(20)
            if key in ABORT_VIDEO_KEYS:
                break

    except Exception:
        print(traceback.format_exc())

    cap.release()
    cv2.destroyAllWindows()
    print('done')
