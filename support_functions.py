#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Published under GNU General Public License v3.0

You should have recieved a copy of the license GNU GPLv3. 
If not, see 
http://www.gnu.org/licenses/

Function to support extraction and sorting of features of arrows.

@author: me
"""
import cv2
import numpy as np

NUM_FEATURES = 15
NUM_POINTS = 5

def find_contours(image):
    """ 
    Change image with some filters and return contours in changed image.
    
    The original image gets changed by morphological gradient, then
    dilated, eroded. After that canny and a additional closing are used.
    In the end findContours searches for contours.
    
    Parameters
    ----------
    image : Mat
        Image used to find contours.
        
    Returns
    -------
    list
        List of found Contours.
    
    """
    kernel = np.ones((5,5),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    dilated = cv2.dilate(image, None, iterations=1)
    filtered= cv2.erode(dilated, None, iterations=1)
    canny = cv2.Canny(filtered, 200, 400)
    dilated = cv2.dilate(canny, None, iterations=1)
    filtered= cv2.erode(dilated, None, iterations=1)
    
    im2, con, hierarchy = cv2.findContours(filtered,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    return con

def get_shape(c):
    """
    Return main points and center data of contour c.
    
    Parameters
    ----------
    c : int32 matrix
        Contour used to extract information about it.
    
    Returns
    -------
    tuple
        Returns a tuple of a matrix containing the 'important' points and
        the x- and y-coordinates of the center of the contour.
        
    """
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
    
def distance_and_angle(dx,dy):
    """
    Calculates and returns distance and angle of values.
    
    Values are changed x- and y-values. Angle is calculated in degree,
    distance is calculated in pixel length, assumed pixelwidth is the same as
    pixelheight.
    
    Parameters
    ----------
    dx : int64
        Change between the x-coordinates of two points.
    dy : int64
        Change between the y-coordinates of two points.
        
    Returns
    -------
    tuple
        Tuple containing distance and angle from x and y changes of two points.
        
    """
    r = np.sqrt(dy**2 + dx**2)
    alpha = np.angle((dx+dy*1j),deg = True)
    
    if (alpha < 0):
        alpha += 360
        
    return (r,alpha)
        
def get_ranges_and_angles(points,x_c,y_c):
    """
    Calculates and returns distances and angles of from given points to center
    and the sum of these distances.
    
    Parameters
    ----------
    points : int32
        Matrix of points of a contour.
    x_c : int
        X-coordinate of center of a contour.
    y_c : int
        Y-coordinate of center of a contour.
        
    Returns
    -------
    tuple
        Tuple containing distances and angles from points to the center of 
        contour. Each from them in float64.
        
    """
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
    """
    Find the two points in a given list, which are closest to each other.
    
    Returns index of values in this list. If length of list is equal or smaller
    than one, the two indices returned are 0.
    
    Parameters
    ----------
    points : int32
        Points of a contour.
   
    Returns
    -------
    tuple
        Tuple containing indices of the closest two points found, or two zeros
        for an error.
        
    """
    if (len(points) <= 1):
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
    """
    Calculates and returns distance and angle from center to the middle of the 
    two closest points and the coordinates of this middle.
    
    This function calculates the middle of the two closest significant points 
    in a contour and calculates distance and angle from center to the middle.
    Later, this middle can be used to sort the points.
    
    Parameters
    ----------
    points : int32
        Matrix of points of a contour.
    x_c : int
        X-coordinate of center of a contour.
    y_c : int
        Y-coordinate of center of a contour.
        
    Returns
    -------
    tuple
        Tuple containing distance and angle from the center of a contour to the
        middle of the two closest points in a contour. Also the coordinates of 
        the middle of the closets points.
        
    """
    idx0,idx1 = find_nearest(points)
    x_middle = int((points[idx1,0,0] + points[idx0,0,0]) /2)
    y_middle = int((points[idx1,0,1] + points[idx0,0,1]) /2)
    dx = x_middle - x_c
    dy = y_middle - y_c
    dist,angle = distance_and_angle(dx,dy)
    
    return (dist,angle,x_middle,y_middle)
                
def ranges_in_percentage(ranges,r_sum):
    """
    Calculates the percentage of each given range compared to their sum.
    
    A symbol with different distances does not have the same range from each
    point to the center, but the percentage should not change. Therefore it is
    calculated here.
    
    Parameters
    ----------
    ranges : float64
        Array of ranges from points of a contour to the center of this contour.
    r_sum : float64
        Sum of given ranges.
        
    Returns
    -------
    Array
        Array containing all ranges in percentage of their sum in float64.
        
    """
    for col in range(0,len(ranges)):
        ranges[col] = ranges[col] / r_sum
    
    return ranges

def sort_angles(angles,ranges,start_angle):
    """
    Sort angles and ranges starting from given angle.
    
    An angle and a distance describes a point. To find the same contour in 
    every rotation, an angle is used as reference. The same contour should have
    the same angles as long as the reference finds the rotation. The angles are
    all calculated positive and start from reference afterwards.
    
    Parameters
    ----------
    angles : float64
        Array of angles of points describing a contour.
    ranges : float64
        Array of ranges of points describing a contour.
    start_angle : float64
        Reference angle is used to sort ranges and angles in respect to it.
        
    Returns
    -------
    tuple
        Tuple containing distances and angles now sorted after reference.
        All values are float64.
        
    """
    for i in range(len(angles)):
        if (angles[i] < start_angle):
            angles[i] += 360
        
        angles[i] -= start_angle
            
    idx = angles.argsort()
    angles = angles[idx]
    ranges = ranges[idx]
    
    return (angles,ranges)

def get_color_in_hsv(im,c):
    """
    Get mean color inside a contour in hsv.
    
    Parameters
    ----------
    im : Mat
        Image containing contours.
    c : int32
        Contour inside image.
        
    Returns
    -------
    uint8
        Matrix containing the mean values of a color in hsv space inside a 
        contour.
        
    """
    mask = np.zeros(im.shape[:2],np.uint8)
    cv2.drawContours(mask,[c],0,255,-1)
    color = cv2.mean(im,mask)
    hsv = cv2.cvtColor(np.uint8([[color[:3]]]),cv2.COLOR_BGR2HSV)
    
    return hsv

def get_percentage_of_area(c):
    """
    Get percentage of contour area compared to the rectangle area.
    
    Parameters
    ----------
    c : int32
        Contour used to calculate areas.
        
    Returns
    -------
    float64
        Area of contour compared to its rectangular area.
        
    """
    area = cv2.contourArea(c)
    rect = cv2.minAreaRect(c)
    rect_size = rect[1][0] * rect[1][1]
    per_con_rect = area/rect_size

    return per_con_rect

def prepare_data(im,c,points,x_c,y_c,shape_ref):
    """
    Prepare data to have all information ready to check this contour.
    
    A collection of nearly all functions above. This is a 'faster' way to
    prepare data to check, whether this contour is the one an arrow would 
    have or not. Shape is compared to a reference shape only here.
    
    Parameters
    ----------
    im : Mat
        Image containing contours.
    c : int32
        Contour inside image.
    points : int32
        Matrix containing the most important points of this contour.
    x_c : int
        X-coordinate of the center of this contour.
    y_c : int
        Y-coordinate of the center of this contour.
    shape_ref : int32
        Matrix of points describing a reference shape.
        
    Returns
    -------
    Array
        Array containing the prepared data to check being the searched 
        symbol or not.
        
    """
    ranges,angles,r_sum = get_ranges_and_angles(points,x_c,y_c)
    ranges = ranges_in_percentage(ranges,r_sum)
    dist,alpha,x_m,y_m = data_between_nearest(points,x_c,y_c)
    angles, ranges = sort_angles(angles,ranges,alpha)
    hsv = get_color_in_hsv(im,c)
    similarity = np.float(cv2.matchShapes(c,shape_ref,cv2.CONTOURS_MATCH_I2,0))
    area = get_percentage_of_area(c)
    
    return np.concatenate((angles,ranges,[dist], hsv[0,0,1:],[similarity],[area]))
