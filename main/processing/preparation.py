from pathlib import PurePath
import pathlib as pl
import random
from shutil import copy2
from glob import glob
import itertools

import numpy as np
import cv2
from keras import Sequential
from keras import layers
# noinspection PyUnresolvedReferences
from keras.models import load_model
# noinspection PyUnresolvedReferences
from keras.preprocessing import image_dataset_from_directory

import numpy.typing as npt

from main.conf.paths import RAW_VIDS_PATH
from main.conf.paths import RAW_IMGS_PATH
from main.conf.paths import ORIGINAL_POS_PATH
from main.conf.paths import ORIGINAL_POS_SUBPATH
from main.conf.paths import ARROWS_PATH
from main.conf.paths import ANYTHING_PATH
from main.conf.paths import RAW_NEG_IMGS_PATH
from main.conf.paths import UNUSED_NEG_PATH
from main.conf.imgs import COLORS
from main.conf.imgs import TARGET_SIZE
from main.conf.imgs import AREA_BORDER
from main.conf.imgs import BLUR_KERNEL
from main.conf.ai import FRAMES_TILL_IMAGE_EXTRACTION
from main.conf.ai import AUG_RANDOM_ROTATION
from main.conf.ai import AUG_RANDOM_TRANSLATION_H
from main.conf.ai import AUG_RANDOM_TRANSLATION_W
from main.conf.ai import AUG_FILL_VALUE
from main.conf.ai import ROUGHLY_NBR_OF_FILES_PER_CLASS
from .utils import get_current_time_string
from .utils import extract_cnts
from .utils import filter_cnts
from .utils import sort_cnts
from .utils import filter_and_extract_img_from_cnt
from .model_handler import ModelHandler


class Preparation:
    def __init__(self):
        self.model = None

    def check_contours_manually(
        self,
        cnts: npt.ArrayLike,
        gray_img: npt.ArrayLike,
        color: COLORS,
        min_area_cnt: int
    ) -> npt.NDArray | None:
        for num, cnt in enumerate(cnts):
            min_rect = cv2.minAreaRect(cnt)
            area = min_rect[1][0] * min_rect[1][1]
            if min_area_cnt and area < min_area_cnt:
                continue
            col_frame = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(col_frame, cnts, num, color, thickness=10)
            cv2.imshow('Frame', col_frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                return cnt
            elif key == ord('q') or key == 27:
                return None
            elif key == ord('r'):
                return self.check_contours_manually(cnts, gray_img, color, min_area_cnt)

        return None


    def extract_raw_pos_imgs_from_video(self, video_fname: str, nth_frame: int = FRAMES_TILL_IMAGE_EXTRACTION) -> None:
        cap = cv2.VideoCapture(video_fname)

        if not cap.isOpened():
            cap.release()
            print(f'could not open{videos_fname}')
            raise exit(1)

        counter = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret is False:
                    break

                if counter % nth_frame == 0:
                    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    timestring = get_current_time_string()
                    retval = cv2.imwrite(str(PurePath(RAW_IMGS_PATH, f'img_{timestring}.jpg')), gray_img)
                    if not retval:
                        raise ValueError(f'retval:{retval}')

                counter += 1

        except Exception as e:
            print(e)
        cap.release()

    def extract_raw_pos_imgs_from_videos(self, videos_path: str = RAW_VIDS_PATH, nth_frame: int = FRAMES_TILL_IMAGE_EXTRACTION) -> None:
        from datetime import datetime
        from datetime import timezone

        for fname in glob(str(PurePath(videos_path, '*'))):
            self.extract_raw_pos_imgs_from_video(fname, nth_frame)

    def load_model_for_classification(self, model_path: str = None, ignore_when_model_exists: bool = True):
        if self.model is not None and ignore_when_model_exists:
            return False

        m_handler = ModelHandler()
        if model_path is None:
            ret = m_handler.load_model()
        else:
            ret = m_handler.load_model(model_path)

        if ret:
            self.model = m_handler.model
            m_handler.model = None
        else:
            return False

        self.model.trainable = False
        return True

    def extract_pos_imgs_from_imgs(
        self,
        use_model: bool = False,
        min_area_cnt: int = AREA_BORDER-10,
        expected_pts: int = ARROW_CONTOUR_POINTS
    ):
        for img_filename in glob(str(PurePath(RAW_IMGS_PATH, '*.jpg'))):
            gray_img = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                print(f'WARNING: {img_filename} could not be loaded')
                continue

            blurred = cv2.blur(gray_img, BLUR_KERNEL)
            cnts = extract_cnts(blurred)

            if use_model and self.model is not None:
                filtered_list, cnts, hull_rot_pts = filter_cnts(cnts, gray_img, expected_pts)
                if not len(filtered_list):
                    print(f'no candidate for prediction found in {img_filename}')
                    continue
                prediction = self.model.predict(filtered_list).flatten()
                pos_cnts, neg_cnts, _, _ = sort_cnts(prediction, cnts)
            else:
                pos_cnts = []
                neg_cnts = cnts

            shown_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(shown_img, pos_cnts, -1, COLORS.BLUE.value, 2)
            cv2.drawContours(shown_img, neg_cnts, -1, COLORS.RED.value, 2)

            cv2.imshow('Frame', shown_img)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('n'):
                contour = self.check_contours_manually(cnts, gray_img, COLORS.PURPLE.value, min_area_cnt)
                if contour is not None:
                    small_img = filter_and_extract_img_from_cnt(gray_img, contour, area_filter=False, w_h_filter=False)
                    timestring = get_current_time_string()
                    retval = cv2.imwrite(
                        str(PurePath(ORIGINAL_POS_SUBPATH, f'img_{timestring}.jpg')),
                        small_img
                    )
                    if not retval:
                        raise ValueError(f'retval:{retval}')

            elif key == ord('y'):
                if len(pos_cnts) == 1:
                    small_img = filter_and_extract_img_from_cnt(
                        gray_img, pos_cnts[0],
                        area_filter=False,
                        w_h_filter=False
                    )
                    timestring = get_current_time_string()
                    retval = cv2.imwrite(f'../original-positives/original-positives/img_{timestring}.jpg',
                                         small_img)
                    if not retval:
                        raise ValueError(f'retval:{retval}')

                else:
                    print('too many values -> no save')

            elif key == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()

    def build_aug_model(
        self,
        rnd_rot: float = AUG_RANDOM_ROTATION,
        w_factor: float = AUG_RANDOM_TRANSLATION_H,
        h_factor: float = AUG_RANDOM_TRANSLATION_W,
        fill_value: int = AUG_FILL_VALUE
    ) -> Sequential:
        return Sequential([
            layers.RandomRotation(rnd_rot),
            layers.RandomTranslation(h_factor, w_factor, fill_value=fill_value),
        ])

    def aug_imgs_and_build_pos_dataset(
        self,
        source_dir: str = ORIGINAL_POS_PATH,
        roughly_created_size: int = ROUGHLY_NBR_OF_FILES_PER_CLASS,
        batch_size: int = 1_000
    ):
        aug_net = self.build_aug_model()

        org_ds = image_dataset_from_directory(
            source_dir,
            image_size=(TARGET_SIZE[1], TARGET_SIZE[0]),
            batch_size=batch_size
        )

        nbr_files = len(list(pl.Path(ORIGINAL_POS_SUBPATH).iterdir()))
        digits_in_fname = int(np.floor(np.log10(roughly_created_size)))
        nbr_augs_same_img = roughly_created_size / nbr_files
        if nbr_augs_same_img.is_integer():
            digits_in_fname += 1 # for equal dividable numbers that reach 10_000
        nbr_augs_same_img = int(np.floor(nbr_augs_same_img))

        num = 0
        for images_per_iter, label in org_ds:
            for _ in range(nbr_augs_same_img):
                aug_imgs = aug_net(images_per_iter).numpy()
                for img in aug_imgs:
                    filename = str(PurePath(ARROWS_PATH, f'img_{str(num).rjust(digits_in_fname, '0')}.jpg'))
                    retval = cv2.imwrite(filename, img)
                    if not retval:
                        raise ValueError(f'retval saving augmented image:{retval}')
                    num += 1

        print(f'number of created images: {num}')

    def extract_neg_candidates(self):
        path = pl.Path(RAW_NEG_IMGS_PATH)
        for num, image_name in enumerate(path.iterdir()):
            big_neg_image = cv2.imread(str(image_name))
            if big_neg_image is None:
                print(f'could not load image {image_name}')
                break

            gray_img = cv2.cvtColor(big_neg_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.blur(gray_img, BLUR_KERNEL)
            cnts = extract_cnts(blurred)

            for con in cnts:
                small_img = filter_and_extract_img_from_cnt(gray_img, con)
                if small_img is not None:
                    timestring = get_current_time_string()
                    retval = cv2.imwrite(str(PurePath(UNUSED_NEG_PATH, f'img_{timestring}.jpg')), small_img)
                    if not retval:
                        raise ValueError(f'retval saving negative image:{retval}')

            if num % 10 == 0:
                print(f'at least {num} raw images finished', end='\r')

    def move_part_of_unused_to_neg_dataset(self, nbr_imgs=ROUGHLY_NBR_OF_FILES_PER_CLASS):
        used_neg_filenames = itertools.islice(pl.Path(UNUSED_NEG_PATH).glob('*.jpg'), nbr_imgs)
        for file in used_neg_filenames:
            copy2(file, PurePath(ANYTHING_PATH, file.name))
