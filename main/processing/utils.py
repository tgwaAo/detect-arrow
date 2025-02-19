#!/bin/env python3

import numpy as np
import cv2

import numpy.typing as npt

from main.conf.img_consts import MIN_WIDTH_TO_HEIGHT
from main.conf.img_consts import MAX_WIDTH_TO_HEIGHT
from main.conf.img_consts import AREA_BORDER
from main.conf.img_consts import TARGET_SIZE
from main.conf.img_consts import COMPARED_SIZE
from main.conf.img_consts import BLUR_KERNEL
from datetime import datetime
from datetime import timezone


def get_current_time_string():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S-%f')


def filter_and_extract_img_from_cnt(
    gray_img,
    con,
    area_filter: bool = True,
    w_h_filter: bool = True
) -> npt.NDArray | None:
    min_rect = cv2.minAreaRect(con)
    center, size, angle = min_rect
    area = size[0] * size[1]

    if area_filter and area < AREA_BORDER:
        return None

    low_value = min(size[0], size[1])
    high_value = max(size[0], size[1])
    width_to_height = low_value / high_value

    if MIN_WIDTH_TO_HEIGHT < width_to_height < MAX_WIDTH_TO_HEIGHT or not w_h_filter:
        cropped_img = rotate_and_crop_min_rect(gray_img, min_rect)
        small_img = cv2.resize(cropped_img, TARGET_SIZE)
        return small_img

    return None


def filter_and_extract_norm_img_from_cnt(gray_img, con):
    small_img = filter_and_extract_img_from_cnt(gray_img, con)
    if small_img is not None:
        small_img = small_img / 255
        return small_img

    return None


def rotate_and_crop_min_rect(image, min_rect, factor: float = 1.3):
    box = cv2.boxPoints(min_rect)
    box = np.intp(box)

    width = round(min_rect[1][0])
    height = round(min_rect[1][1])

    size_of_transformed_image = max(min_rect[1])
    min_needed_height = int(np.sqrt(2 * np.power(size_of_transformed_image, 2)))
    width_to_height = min_rect[1][0] / min_rect[1][1]

    if width_to_height >= 1:
        angle = -1 * (90 - min_rect[2])
    else:
        angle = min_rect[2]

    size = (min_needed_height, min_needed_height)

    x_coordinates_of_box = box[:,0]
    y_coordinates_of_box = box[:,1]
    x_min = min(x_coordinates_of_box)
    x_max = max(x_coordinates_of_box)
    y_min = min(y_coordinates_of_box)
    y_max = max(y_coordinates_of_box)

    center = (int((x_min+x_max)/2), int((y_min+y_max)/2))
    cropped = cv2.getRectSubPix(image, size, center)
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    cropped = cv2.warpAffine(cropped, M, size)

    if width_to_height >= 1:
        cropped_rotated = cv2.getRectSubPix(cropped, (int(factor * height), int(factor * width)), (size[0]/2, size[1]/2))
    else:
        cropped_rotated = cv2.getRectSubPix(cropped, (int(factor * width), int(factor * height)), (size[0]/2, size[1]/2))

    return cropped_rotated


def extract_cnts(img, sigma=0.33):
    v = np.median(img)
    # ---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))  # ---- lower threshold
    upper = int(min(255, (1.0 + sigma) * v))  # ---- upper threshold
    thresh_img = cv2.Canny(img, lower, upper)
    cnts, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def extract_feature_pts(pos_cnt, factor: float = 0.04):
    retval = cv2.arcLength(pos_cnt, True)
    points = cv2.approxPolyDP(pos_cnt, factor * retval, True)
    return points


def merge_points(points, max_merge_dist=4):
    to_merge = []
    checked_points_idx = []
    last_to_merge = False
    for idx in range(len(points) - 1):
        first_point = points[idx, 0]
        if idx not in checked_points_idx:
            to_merge_bundle = [first_point]

            for idx2 in range(idx + 1, len(points)):
                second_point = points[idx2, 0]
                dist = np.abs(first_point - second_point)

                if dist[0] < max_merge_dist and dist[1] < max_merge_dist:
                    to_merge_bundle.append(second_point)
                    checked_points_idx.append(idx2)
                    if idx2 == len(points) - 1:
                        last_to_merge = True

            to_merge.append(to_merge_bundle)

    if not last_to_merge:
        to_merge.append([points[-1, 0]])

    filtered_points = []
    for to_merge_bundle in to_merge:
        if len(to_merge_bundle) == 1:
            filtered_points.append(to_merge_bundle[0])
        else:
            filtered_point = np.sum(to_merge_bundle, axis=0) / len(to_merge_bundle)
            filtered_points.append(filtered_point)

    return filtered_points


def rotate_and_crop(image, min_area_rect, factor=1.3, cnt=None):
    box = cv2.boxPoints(min_area_rect)
    box = np.intp(box)

    width = round(min_area_rect[1][0])
    height = round(min_area_rect[1][1])

    size_of_transformed_image = max(min_area_rect[1])
    min_needed_height = int(np.sqrt(2 * np.power(size_of_transformed_image, 2)))

    width_to_height = min_area_rect[1][0] / min_area_rect[1][1]

    if width_to_height >= 1:
        min_rect_angle_deg = -1 * (90 - min_area_rect[2])
    else:
        min_rect_angle_deg = min_area_rect[2]

    size = (min_needed_height, min_needed_height)

    x_coordinates_of_box = box[:, 0]
    y_coordinates_of_box = box[:, 1]
    x_min = min(x_coordinates_of_box)
    x_max = max(x_coordinates_of_box)
    y_min = min(y_coordinates_of_box)
    y_max = max(y_coordinates_of_box)

    center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
    cropped = cv2.getRectSubPix(image, size, center)
    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), min_rect_angle_deg, 1.0)
    cropped = cv2.warpAffine(cropped, M, size)

    if width_to_height >= 1:
        cropped_rotated = cv2.getRectSubPix(cropped, (int(factor * height), int(factor * width)),
                                            (size[0] / 2, size[1] / 2))
    else:
        cropped_rotated = cv2.getRectSubPix(cropped, (int(factor * width), int(factor * height)),
                                            (size[0] / 2, size[1] / 2))

    if cnt is not None:
        hull_pts = extract_feature_pts(cnt)
        hull_pts = np.array(merge_points(hull_pts)) - min_area_rect[0]
        rot_pts = np.zeros((len(hull_pts), 2))

        for idx, pt in enumerate(hull_pts):
            angle_rad = np.arctan2(pt[1], pt[0]) - np.deg2rad(min_rect_angle_deg)
            dist = np.hypot(pt[0], pt[1])
            pt_x = dist * np.cos(angle_rad)
            pt_y = dist * np.sin(angle_rad)
            rot_pts[idx] = pt_x, pt_y

    return cropped_rotated, rot_pts


def sort_cnts(prediction, pos_filtered_to_pos_source, cnts):
    mask = np.zeros(len(cnts), dtype=bool)
    mask[np.where(prediction >= 0.5)[0]] = True
    positive_contours = cnts[mask]
    negative_contours = cnts[~mask]
    return positive_contours, negative_contours


def filter_cnts(cnts, gray_img=None):
    small_imgs = []
    filtered_cnts = []
    center_list = []
    hull_rot_pts = []
    too_close = False
    for pos_source, con in enumerate(cnts):
        min_rect = cv2.minAreaRect(con)
        center, size, angle_deg = min_rect
        area = size[0] * size[1]

        if area < AREA_BORDER:
            continue

        low_value = min(size[0], size[1])
        high_value = max(size[0], size[1])
        if high_value == 0:
            continue

        width_to_height = low_value / high_value

        if MIN_WIDTH_TO_HEIGHT < width_to_height < MAX_WIDTH_TO_HEIGHT:
            for c_point in center_list:
                too_close = np.all(np.isclose(center, c_point, rtol=0, atol=20))
                if too_close:
                    break

            if too_close:
                continue

            center_list.append(center)
            filtered_cnts.append(con)

            if gray_img is not None:
                cropped_img, rot_pts = rotate_and_crop(gray_img, min_rect, cnt=con)
                hull_rot_pts.append(rot_pts)
                small_img = cv2.resize(cropped_img, COMPARED_SIZE)
                small_imgs.append(small_img)

    small_imgs = np.array(small_imgs)
    return small_imgs, filtered_cnts, hull_rot_pts
