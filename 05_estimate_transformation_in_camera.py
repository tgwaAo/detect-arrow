#!/bin/env python3

from pathlib import PurePath
from time import time
import json
import traceback

from main.conf.path_consts import calibration_images_basepath
from main.processing.utils import est_pose_in_img
from main.io.video import VideoCapture

if __name__ == '__main__':
    cam_nbr = 0
    cam_height = None
    cam_width = None
    time_till_update = 1
    y_location_text = 10

    cam_conf_filepath = PurePath(calibration_images_basepath, 'cam_nbr.json')
    if cam_conf_filepath.is_file():
        try:
            with open(cam_conf_filepath, 'r') as file:
                cam_conf = json.load(fid)
            cam_nbr = cam_conf.get('cam_nbr', cam_nbr)
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

    cap = VideoCapture(cam_nbr)

    if not cap.isOpened():
        cap.release()
        print(f'could not open camera at index {cam_nbr}')
        exit(1)

    abort = False
    text = ''
    pos_cnts = ()
    neg_cnts = ()
    time_start = time()
    while not abort:
        ret, img = cap.read()
        if not ret:
            print('can not receive frame -> abort')
            break

        time_now = time()
        if (time_now - time_start) > 1:
            time_start = time_now
            try:
                result = est_pos_in_img(img, model, points_printed)
                if result is not None:
                    R, T, tmp_pos_cnts, tmp_neg_cnts = result
                else:
                    R = T = tmp_pos_cnts = tmp_neg_cnts = None

            except Exception as e:
                print(traceback.format_exc())
                break

            if tmp_pos_cnts is not None:
                pos_cnts = tmp_pos_cnts
                neg_cnts = tmp_neg_cnts

            if R is not None:  # T is not None, too
                text = f'R:{np.array2string(R, precision=3, floatmode='fixed')}; T:{np.array2string(T, precision=3, floatmode='fixed')}'

            cv2.drawContours(img, pos_cnts, -1, (255, 0, 0), 2)
            cv2.drawContours(img, neg_cnts, -1, (0, 0, 255), 2)
            cv2.putText(img, text, (10, y_location), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 0))
            cv2.imshow('camera test', img)

    cap.release()
    cv2.destroyAllWindows()
    print('done')
