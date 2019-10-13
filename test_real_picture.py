#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Published under GNU General Public License v3.0

You should have recieved a copy of the license GNU GPLv3. 
If not, see 
http://www.gnu.org/licenses/

Search and measure from a picture.

@author: me
"""

import cv2
import numpy as np
import _pickle
import support_functions as sf
import argparse

shape_ref = np.load("shape_reference.npy")
cam_data = np.loadtxt("cam_data.txt")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="input file for searching")

try:
    args = vars(ap.parse_args())
    filename = args["image"]
    im = cv2.imread(filename)
    
    with open("sign_ai.pkl","rb") as f:
        ai = _pickle.load(f)
    
    print(filename)
    con = sf.find_contours(im)
    data = np.zeros((1,sf.NUM_FEATURES))

    for i in range(len(con)):
        points,x_c,y_c = sf.get_shape(con[i])
        
        if (len(points) == sf.NUM_POINTS):
            ranges,angles,r_sum = sf.get_ranges_and_angles(points,x_c,y_c)
            ranges = sf.ranges_in_percentage(ranges,r_sum)
            dist,alpha,x_m,y_m = sf.data_between_nearest(points,x_c,y_c)
            angles, ranges = sf.sort_angles(angles,ranges,alpha)
            hsv = sf.get_color_in_hsv(im,con[i])
            similarity = np.float(cv2.matchShapes(con[i],shape_ref,cv2.CONTOURS_MATCH_I2,0))
            area = sf.get_percentage_of_area(con[i])
            
            data[0,:] = np.concatenate((angles,ranges,[dist], hsv[0,0,1:],[similarity],[area]))
    
            prob = ai.predict_proba(data)
            proba = prob[0][1]
        
            if (proba > 0.5):
                d = cam_data[0]*cam_data[1] / (dist*10)

                best_x_c = x_c
                best_y_c = y_c
                best_x_m = x_m
                best_y_m = y_m
                best_points = points
                best_d = d
                best_prob = proba
                best_i = i
                best_ranges = ranges
                best_angles = angles
                best_sim = similarity
                best_area = area
                best_hsv = hsv
                best_alpha = alpha

    print("Contour nbr: ",best_i)
    print()
    print("AI precision: ",best_prob)
    print("Angles: ",best_angles)
    print("Ranges: ",best_ranges)
    print("Color: ",best_hsv[0,0,1:])
    print("Difference to reference: ",best_sim)
    print("Contour area / rectangular area ",best_area )
    print("Distance to arrow: ",best_d)
    print("Angle of arrow: ", 180+best_alpha)
    dx_px = x_c - im.shape[1]/2
    dy_px = y_c - im.shape[0]/2
    print("The horizontal distance from center of arrow to center of image is ",dx_px," pixel and the vertical distance is ",dy_px, "pixel.")
                
    if (best_prob > 0):
        for j in range(len(best_points)):
            cv2.line(im,(best_points[j,0,0],best_points[j,0,1]),(best_points[j,0,0],best_points[j,0,1]),(0,0,255),10)
        
        cv2.drawContours(im,con,best_i,(255,0,0),2)
        cv2.line(im,(best_x_c,best_y_c),(best_x_c,best_y_c),(255,255,0),20)
        cv2.line(im,(best_x_m,best_y_m),(best_x_m,best_y_m),(0,255,0),15)
        cv2.putText(im, "{:.0f}".format(best_d)+" cm", (10,  im.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 0, 0), 5)    
        cv2.putText(im, "{:.2f}".format(best_prob)+" probability", (230,  im.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 0, 0), 5)    

    cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
    cv2.imshow(filename,im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("done")

except:
    print("Something went wrong.")
    
