import numpy as np
import cv2

import numpy.typing as npt

from main.conf.imgs import MIN_WIDTH_TO_HEIGHT
from main.conf.imgs import MAX_WIDTH_TO_HEIGHT
from main.conf.imgs import AREA_BORDER
from main.conf.imgs import TARGET_SIZE
from main.conf.imgs import COMPARED_SIZE
from main.conf.imgs import BLUR_KERNEL
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
        cropped_img, _ = rotate_and_crop(gray_img, min_rect)
        small_img = cv2.resize(cropped_img, TARGET_SIZE)
        return small_img

    return None


def filter_and_extract_norm_img_from_cnt(gray_img, con):
    small_img = filter_and_extract_img_from_cnt(gray_img, con)
    if small_img is not None:
        small_img = small_img / 255
        return small_img

    return None


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


def rotate_and_crop(
        image,
        min_area_rect,
        factor=1.3,
        cnt=None
) -> tuple[npt.NDArray[np.uint8], list[tuple[float,float]]]:
    box = cv2.boxPoints(min_area_rect)
    box = np.intp(box)

    width = round(min_area_rect[1][0])
    height = round(min_area_rect[1][1])

    size_of_transformed_image = max(min_area_rect[1])
    min_needed_height = int(np.sqrt(2 * np.power(size_of_transformed_image, 2)))

    width_to_height = min_area_rect[1][0] / min_area_rect[1][1]

    min_rect_angle_deg = get_rotation(min_area_rect[2], width_to_height)

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

    else:
        rot_pts = None

    return cropped_rotated, rot_pts


def get_rotation(min_area_angle, width_to_height):
    if width_to_height >= 1:
        min_rect_angle_deg = -1 * (90 - min_area_angle)
    else:
        min_rect_angle_deg = min_area_angle
    return min_rect_angle_deg


def sort_cnts(prediction, cnts, hull_rot_pts):
    mask = np.zeros(len(cnts), dtype=bool)
    mask[np.where(prediction >= 0.5)[0]] = True
    positive_contours = cnts[mask]
    negative_contours = cnts[~mask]
    pos_prediction = prediction[0][mask]
    neg_prediction = prediction[0][~mask]
    hull_rot_pts = hull_rot_pts[mask]
    return positive_contours, negative_contours, pos_prediction, neg_prediction, hull_rot_pts


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
    hull_rot_pts = np.array(hull_rot_pts)
    return small_imgs, filtered_cnts, hull_rot_pts


def more_pts_up(pts, center=None):
    if center is not None:
        y_max = center[1]
    else:
        y_max = 0

    pts_up = 0
    pts_down = 0

    for x, y in pts:
        if y < y_max:
            pts_up += 1
        else:
            pts_down += 1

    return pts_down < pts_up


def angle_x_axis(pt):
    angle_rad = np.arctan2(pt[0], pt[1])
    if angle_rad < 0:
        angle_rad += 2 * np.pi
    return angle_rad


def get_max_dist_reference(pts, ref_up):
    max_dist = 0
    min_idx = -1
    for idx, pt in enumerate(pts):
        dist = np.power(pt[0], 2) + np.power(pt[1], 2)
        if max_dist < dist:
            if ref_up and pt[1] < 0:  # up is lower
                min_idx = idx
                max_dist = dist
            elif not ref_up and 0 < pt[1]:
                min_idx = idx
                max_dist = dist

    if min_idx == -1:
        return None
    return pts[min_idx]


def rot_centered_pts(pts, ref_angle_rad):
    rot_pts = np.zeros((len(pts), 2))
    for idx, pt in enumerate(pts):
        angle_rad = np.arctan2(pt[1], pt[0]) + ref_angle_rad
        dist = np.hypot(pt[0], pt[1])
        pt_x = dist * np.cos(angle_rad)
        pt_y = dist * np.sin(angle_rad)
        rot_pts[idx] = pt_x, pt_y

    return rot_pts


def sort_pts_by_angles(rot_pts, org_pts):
    angles_rad = [angle_x_axis(pt) for pt in rot_pts]
    idx_sorted = np.argsort(angles_rad)
    return org_pts[idx_sorted]


def sort_pt_biggest_dist_center(pts, ref_up, org_pts):
    closest_y = get_max_dist_reference(pts, ref_up)
    ref_angle_rad = (np.pi / 2) - np.arctan2(closest_y[1], closest_y[0])
    rot_pts = rot_centered_pts(pts, ref_angle_rad)
    return sort_pts_by_angles(rot_pts, org_pts)


def calc_rot_and_trans(H, K):
    H = H.T
    h1 = H[0]
    h2 = H[1]
    h3 = H[2]
    K_inv = np.linalg.inv(K)
    L = 1 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = L * np.dot(K_inv, h1)
    r2 = L * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)

    T = L * np.dot(K_inv, h3)
    # print(T)

    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3))
    U, S, V = np.linalg.svd(R, full_matrices=True)
    U = np.matrix(U)
    V = np.matrix(V)
    R = U * V

    alpha = np.rad2deg(np.arctan2(R[2, 1], R[2, 2]))
    beta = np.rad2deg(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] * R[2, 1] + R[2, 2] * R[2, 2])))
    gamma = np.rad2deg((np.arctan2(R[1, 0], R[0, 0])))
    return np.array((alpha, beta, gamma)), T


def est_cnts_in_img(
    gray_img,
    model,
    verbose: bool = False
) -> tuple[npt.NDArray[int], npt.NDArray[int], npt.NDArray[float], npt.NDArray[float], npt.NDArray[int]]:
    cnts, gray_img = extract_cnts(gray_img)
    filtered_images, cnts, hull_rot_pts = filter_cnts(cnts, gray_img)

    if not len(filtered_images):
        if verbose:
            print('no candidate for prediction found')
        return None

    prediction = model(filtered_images, verbose=0)

    return sort_cnts(prediction, cnts, hull_rot_pts)


def est_pose_of_cnt(best_cnt, points_printed, best_hull_rot_pts, verbose):
    hull_points = extract_feature_pts(best_cnt)
    hull_points = merge_points(hull_points)
    if len(hull_points) != 5:
        if verbose:
            print(f'not enough hull points, got {len(hull_points)} need 5')
        return None
    hull_points = np.reshape(hull_points, (5, 2))
    ref_up = more_pts_up(best_hull_rot_pts)
    hull_points = sort_pt_biggest_dist_center(best_hull_rot_pts, ref_up, hull_points)
    H, _ = cv2.findHomography(points_printed, hull_points, cv2.RANSAC)
    R, T = calc_rot_and_trans(H, K)
    return R, T, best_cnt


def est_pose_in_img(gray_img, model, points_printed, verbose: bool = False):
    pos_cnts, neg_cnts, pos_pred, neg_pred, hull_rot_pts = est_cnts_in_img(gray_img, model, verbose=verbose)
    if not len(pos_cnts):
        if verbose:
            print('no positive contour found')
        return None

    idx = np.argmax(pos_pred[0])
    best_cnt = pos_conts[idx]
    best_hull_rot_pts = hull_rot_pts[idx]
    result = est_pose_of_cnt(best_cnt, points_printed, best_hull_rot_pts, verbose)
    if result is None:
        return result

    return *result, pos_pred[idx]


def est_poses_in_img(gray_img, model, points_printed, verbose: bool = False):
    pos_cnts, neg_cnts, pos_pred, neg_pred, hull_rot_pts = est_cnts_in_img(gray_img, model, verbose=verbose)
    if not len(pos_cnts):
        if verbose:
            print('no positive contour found')
        return None

    all_ret_vals = [None] * len(pos_cnts)
    for idx in range(len(pos_cnts)):
        result = est_pose_of_cnt(pos_cnts[idx], points_printed, hull_rot_ptsverbose[idx])
        if result is None:
            if verbose:
                print(f'could not estimate pose at {idx}')
            result = None, None, None

        all_ret_vals[idx] = *result, pos_pred[idx]

    return all_ret_vals



