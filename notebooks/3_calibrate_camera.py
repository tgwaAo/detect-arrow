#!/usr/bin/env python
# coding: utf-8

# # Calibrate camera

# Sources:  
# > https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html  
# > https://github.com/opencv/opencv/blob/4.x/doc/pattern.png  
# > https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html  
# > https://answers.opencv.org/question/99030/findchessboardcorners-returing-false-boolean-value/  

# In[2]:


import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt


# In[4]:


images = glob.glob('../../calibration-images/*.jpg')
path = '../../cam-config/'

term_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
corners = None

if not images:
    raise SystemExit(1)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners; size must be exact or ret will be false
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    extension = 'txt'
    np.savetxt(f'{path}mtx.{extension}', mtx)
    np.savetxt(f'{path}dist.{extension}', dist)
    print(f'mtx:\n{mtx}\n')
    print(f'dist:\n{dist}\n')

h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
mini_dst = dst[y:y + h, x:x + w]
combined = np.hstack((img, dst))
cv.imshow('calibration', combined)

merged = cv.addWeighted(img, 0.5, dst, 0.5, 0)
cv.imshow('merged', merged)
cv.waitKey(0)

cv.destroyAllWindows()

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))


# In[6]:


images = glob.glob('../../calibration-images/*.jpg')
path = '../../cam-config/'

term_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objp *= 23 # in mm

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
corners = None

if not images:
    raise SystemExit(1)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners; size must be exact or ret will be false
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    extension = 'txt'
    np.savetxt(f'{path}mtx.{extension}', mtx)
    np.savetxt(f'{path}dist.{extension}', dist)
    print(f'mtx:\n{mtx}\n')
    print(f'dist:\n{dist}\n')

h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
mini_dst = dst[y:y + h, x:x + w]
combined = np.hstack((img, dst))
cv.imshow('calibration', combined)

merged = cv.addWeighted(img, 0.5, dst, 0.5, 0)
cv.imshow('merged', merged)
cv.waitKey(0)

cv.destroyAllWindows()

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))


# In[ ]:




