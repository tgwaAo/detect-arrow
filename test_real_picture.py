#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:41:22 2019

Test searching of arrow pointing up.

@author: me
"""

import cv2
import numpy as np
import skimage as sk
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
    
    for i in range(len(con)):
        points,x_c,y_c = sf.get_shape(con[i])
        
        if (len(points) == sf.NUM_POINTS):
            ranges,angles,r_sum = sf.get_ranges_and_angles(points,x_c,y_c)
            ranges = sf.ranges_in_percentage(ranges,r_sum)
            dist,alpha,x_m,y_m = sf.data_between_nearest(points,x_c,y_c)
            
            if (dist > 25):
                angles, ranges = sf.sort_angles(angles,ranges,alpha)
                hsv = sf.get_color_in_hsv(im,con[i])
                similarity = np.float(cv2.matchShapes(con[i],shape_ref,cv2.CONTOURS_MATCH_I2,0))
                
                data = np.zeros((1,sf.NUM_FEATURES))
                data[0,:] = np.concatenate((angles,ranges,[dist],[alpha], hsv[0,0,1:],[similarity]))
        
                ans = ai.predict(data)
                
                if (ans == 1 ):
                    print("Contour nbr: ",i)
                    print()
                    print("Angles: ",angles)
                    print("Ranges: ",ranges)
                    print("End of arrow: ",dist,", ", alpha)
                    print("Color: ",hsv[0,0,1:])
                    print("Difference to reference: ",similarity)
                    print("AI precision: ",ai.predict_proba(data))

                    d = cam_data[0]*cam_data[1] / (dist*10)
                    dx_px = x_c - im.shape[1]/2
                    dy_px = y_c - im.shape[0]/2
                    print("Distance to arrow is: ",d," in cm")
                    print("The horizontal distance from center of arrow to center of image is ",dx_px," pixel and the vertical distance is ",dy_px, "pixel.")

                    for j in range(len(points)):
                        cv2.line( im,(points[j,0,0],points[j,0,1]),(points[j,0,0],points[j,0,1]),(0,0,255),10)
                    
                    cv2.drawContours( im,con,i,(255,0,0),2)
                    cv2.line( im,(x_c,y_c),(x_c,y_c),(255,255,0),20)
                    cv2.putText( im, "{:.0f}".format(d)+" mm", (10,  im.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 255, 0), 2)    
                    cv2.line( im,(x_m,y_m),(x_m,y_m),(0,255,0),15)       
                    break
                
                if (i == 33):
                    print("AI precision: ",ai.predict_proba(data))

    
    sk.io.imshow( im)
    sk.io.show()
    
except:
    print("Something went wrong.")
    
print("done")
