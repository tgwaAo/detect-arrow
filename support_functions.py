#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 22:44:27 2019

Function to support extraction and sorting of features of arrows.

@author: me
"""
import cv2
import numpy as np
import skimage as sk

NUM_FEATURES = 15
NUM_POINTS = 5

def find_contours(pic):
    kernel = np.ones((5,5),np.uint8)
    pic = cv2.morphologyEx(pic, cv2.MORPH_GRADIENT, kernel)
    dilated = cv2.dilate(pic, None, iterations=1)
    filtered= cv2.erode(dilated, None, iterations=1)
    canny = cv2.Canny(filtered, 200, 400)
    dilated = cv2.dilate(canny, None, iterations=1)
    filtered= cv2.erode(dilated, None, iterations=1)
    
    im2, con, hierarchy = cv2.findContours(filtered,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    return con

def get_shape(c):
    M = cv2.moments(c)
    if (M["m00"] != 0):
        x_c = int(M["m10"] / M["m00"])
        y_c = int(M["m01"] / M["m00"])
    else:
        x_c = -1
        y_c = -1
    retval = cv2.arcLength(c, True)
    points = cv2.approxPolyDP(c, 0.04 * retval, True)
    
    return (points,x_c,y_c)
    
def make_digital_ready(pic,angle):
    im = sk.transform.rotate(pic,angle)
    toint = im * 255
    res = np.invert(toint.astype(np.uint8))
    con = find_contours(res)
    c = 0
    points,x_c,y_c = get_shape(con[c])
    
    return (res,con[c],points,x_c,y_c)

def distance_and_angle(dx,dy):
    r = np.sqrt(dy**2 + dx**2)
    alpha = np.angle((dx+dy*1j),deg = True)
    
    if (alpha < 0):
        alpha += 360
        
    return (r,alpha)
        
#        describe corners
def get_ranges_and_angles(points,x_c,y_c):
    col = 0
    r_sum = 0
    ranges = np.zeros(len(points))
    angles = np.zeros(len(points))
    
    for i in range(0,len(points)):
        dx = points[i,0,0] - x_c
        dy = points[i,0,1] - y_c
        
        r,alpha = distance_and_angle(dx,dy)
        r_sum += r

        ranges[col] = r
        angles[col] = alpha
        col += 1
        
    return (ranges,angles,r_sum)

def find_nearest(points):
    if (len(points) == 1):
        return (0,0)
    else:
        dx = points[1,0,0] - points[0,0,0]
        dy = points[1,0,1] - points[0,0,1]
        dist_min,angle = distance_and_angle(dx,dy)
        idx0 = 0
        idx1 = 1
        
        for i in range(len(points)-1):
            for j in range(i+1,len(points)):
                dx = points[j,0,0] - points[i,0,0]
                dy = points[j,0,1] - points[i,0,1]
                dist,angle = distance_and_angle(dx,dy)
                
                if (dist < dist_min):
                    dist_min = dist 
                    idx0 = i
                    idx1 = j
                    
        return (idx0,idx1)

def data_between_nearest(points,x_c,y_c):
    idx0,idx1 = find_nearest(points)
    x_middle = int((points[idx1,0,0] + points[idx0,0,0]) /2)
    y_middle = int((points[idx1,0,1] + points[idx0,0,1]) /2)
    dx = x_middle - x_c
    dy = y_middle - y_c
    dist,angle = distance_and_angle(dx,dy)
    
    return (dist,angle,x_middle,y_middle)
                
def ranges_in_percentage(ranges,r_sum):
    for col in range(0,len(ranges)):
        ranges[col] = ranges[col] / r_sum
    
    return ranges

def sort_angles(angles,ranges,start_angle):
    for i in range(len(angles)):
        if (angles[i] < start_angle):
            angles[i] += 360
        
        angles[i] -= start_angle
            
    idx = angles.argsort()
    angles = angles[idx]
    ranges = ranges[idx]
    
    return (angles,ranges)

def prepare_data(im,c,points,x_c,y_c,shape_ref):
    ranges,angles,r_sum = get_ranges_and_angles(points,x_c,y_c)
    ranges = ranges_in_percentage(ranges,r_sum)
    dist,alpha,x_m,y_m = data_between_nearest(points,x_c,y_c)
    angles, ranges = sort_angles(angles,ranges,alpha)
    hsv = get_color_in_hsv(im,c)
    similarity = np.float(cv2.matchShapes(c,shape_ref,cv2.CONTOURS_MATCH_I2,0))
    
    return np.concatenate((angles,ranges,[dist],[alpha], hsv[0,0,1:],[similarity]))


def get_color_in_hsv(im,c):
    mask = np.zeros(im.shape[:2],np.uint8)
    cv2.drawContours(mask,[c],0,255,-1)
    color = cv2.mean(im,mask)
    hsv = cv2.cvtColor(np.uint8([[color[:3]]]),cv2.COLOR_BGR2HSV)
    
    return hsv
   