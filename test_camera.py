#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 00:05:52 2019

Test tracking of an arrow.

@author: me
"""

import cv2
import numpy as np
import _pickle
import support_functions as sf

WIDTH = 1280
HEIGHT = 960
shape_ref = np.load("shape_reference.npy")
cam_data = np.loadtxt("cam_data.txt")
data = np.zeros((1,sf.NUM_FEATURES))

best_x_c = -1
best_y_c = -1
best_x_m = -1
best_y_m = -1
best_points = np.zeros((4,1,2))
best_d = -1
best_prob = -1
best_i = -1

# Change window size
cv2.namedWindow("Camera",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera",640,480)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIDTH);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,HEIGHT);

if (not cap.isOpened()):
    print("Camera not found.")
else:
    with open("sign_ai.pkl","rb") as f:
        ai = _pickle.load(f)
    
    while (True):
        ret, frame = cap.read()
        
        con = sf.find_contours(frame)
        best_prob = -1
        
        for i in range(len(con)):
            points,x_c,y_c = sf.get_shape(con[i])

            if (len(points) == sf.NUM_POINTS):
                # extract infos of shape
                ranges,angles,r_sum = sf.get_ranges_and_angles(points,x_c,y_c)
                ranges = sf.ranges_in_percentage(ranges,r_sum)
                dist,alpha,x_m,y_m = sf.data_between_nearest(points,x_c,y_c)
                angles, ranges = sf.sort_angles(angles,ranges,alpha)
                hsv = sf.get_color_in_hsv(frame,con[i])
                similarity = np.float(cv2.matchShapes(con[i],shape_ref,cv2.CONTOURS_MATCH_I2,0))
                area = sf.get_percentage_of_area(con[i])
                
                data[0,:] = np.concatenate((angles,ranges,[dist], hsv[0,0,1:],[similarity],[area]))

                prob = ai.predict_proba(data)
                proba = prob[0][1]
        
                if (proba > 0.5):
                    
                    if (proba > best_prob):
                        d = cam_data[0]*cam_data[1] / (dist*10)

                        best_x_c = x_c
                        best_y_c = y_c
                        best_x_m = x_m
                        best_y_m = y_m
                        best_points = points
                        best_d = d
                        best_prob = proba
                        best_i = i
                        
        if (best_prob > 0):
            for j in range(len(best_points)):
                cv2.line(frame,(best_points[j,0,0],best_points[j,0,1]),(best_points[j,0,0],best_points[j,0,1]),(0,0,255),10)
                
            cv2.drawContours(frame,con,best_i,(255,0,0),2)
            cv2.line(frame,(best_x_c,best_y_c),(best_x_c,best_y_c),(255,255,0),20)
            cv2.line(frame,(best_x_m,best_y_m),(best_x_m,best_y_m),(0,255,0),15)
            cv2.putText(frame, "{:.0f}".format(best_d)+" cm ", (10,  frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 255, 0), 2)    
            cv2.putText(frame, "{:.2f}".format(best_prob)+" probability", (230,  frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,1.5,(255, 255, 0), 2)    
                    
        cv2.imshow("Camera",frame)
        
        pressed_key = cv2.waitKey(1) & 0xFF
        
        # press q or esc
        if (pressed_key == ord("q") or pressed_key == 27):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("done")
