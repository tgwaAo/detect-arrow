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
import extract_features as ef
import argparse

#########################################################
# Load data.
#########################################################
shape_ref = np.load("shape_reference.npy")
cam_data = np.loadtxt("cam_data.txt")

MIN_PROBABILITY = 0.6

#########################################################
# Add argparser to handle user argument.
#########################################################
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="input file for searching")

try:
    #######################################################
    # Load file and ai (gaussian process classifier).
    #######################################################
    args = vars(ap.parse_args())
    filename = args["image"]
    im = cv2.imread(filename)
    
    if im is None:
        raise FileNotFoundError("Image not found")
        
    with open("arrow_ai.pkl", "rb") as f:
        ai = _pickle.load(f)
    
    ######################################################################
    # Extract contours and search for the right contour.
    ######################################################################
    print(filename)
    con = ef.find_contours(im)
    data = np.zeros((1, ef.NUM_FEATURES))
    best_prob = -1
    
    for i in range(len(con)):
        points, x_c, y_c = ef.get_shape_points(con[i])
        
        if (len(points) == ef.NUM_POINTS):
            ranges, angles, r_sum = ef.get_ranges_and_angles(points, x_c, y_c)
            ranges = ef.ranges_in_percentage(ranges, r_sum)
            dist, alpha, x_m, y_m = ef.data_between_nearest(points, x_c, y_c)
            angles, ranges = ef.sort_angles(angles, ranges, alpha)
            hsv = ef.get_color_in_hsv(im, con[i])
            similarity = np.float(cv2.matchShapes(con[i], shape_ref, 
                                                  cv2.CONTOURS_MATCH_I2, 0))
            area = ef.get_percentage_of_area(con[i])
            
            data[0,:] = np.concatenate((angles, ranges, hsv[0,0,1:], [similarity], [area]))

            # Use this line, if you just want to create a usable dataset.
            # data[0,:] = ef.prepare_data(im, con[i], points, x_c, y_c, shape_ref)
            
            prob = ai.predict_proba(data)
            proba = prob[0][1]
        
            if (proba >= MIN_PROBABILITY):
                if (proba > best_prob):
                    d = cam_data[0] * cam_data[1] / (dist * 10)
    
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

    ################################################################
    # Print data of found contour and visualize result.
    ################################################################
    if (best_prob > 0):
        print("Contour nbr: ", best_i)
        print()
        print("AI precision: ", best_prob)
        print("Angles: ", best_angles)
        print("Ranges: ", best_ranges)
        print("Color: ", best_hsv[0,0,1:])
        print("Difference to reference: ", best_sim)
        print("Contour area / rectangular area ", best_area )
        print("Distance to arrow: ", best_d)
        print("Angle of arrow: ", 180 + best_alpha)
        dx_px = x_c - im.shape[1]/2
        dy_px = y_c - im.shape[0]/2
        print("The horizontal distance from center of arrow to center of image is ", dx_px,
              " pixel and the vertical distance is ", dy_px, " pixel.")
                
        for j in range(len(best_points)):
            cv2.line(im, (best_points[j, 0, 0], best_points[j, 0, 1]), 
                     (best_points[j, 0, 0], best_points[j, 0, 1]), (0, 0, 255), 
                     10)
        
        cv2.drawContours(im, con, best_i, (255, 0, 0), 2)
        cv2.line(im,(best_x_c, best_y_c), (best_x_c, best_y_c), (255, 255, 0), 20)
        cv2.line(im,(best_x_m, best_y_m), (best_x_m, best_y_m), (0, 255, 0), 15)
        cv2.putText(im, "{:.0f}".format(best_d) + " cm", 
                    (10,  im.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                    (255, 0, 0), 3)    
        cv2.putText(im, "{:.2f}".format(best_prob) + " probability", 
                    (230,  im.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 0), 3)    

    else:
        print("No arrow found.")
        
    cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(filename, 640, 480)
    cv2.imshow(filename,im)
    
    # Just close the window or press any key.
    while True:
        if cv2.getWindowProperty(filename, cv2.WND_PROP_VISIBLE):
            if cv2.waitKey(100) != -1: # Any key pressed.
                break
        else:
            break
    
    cv2.destroyAllWindows()
    print("done")

# Wrong input in argparse.
except FileNotFoundError as err:
    print(err.args)

# Option to abort.
except KeyboardInterrupt:
    print("Aborted")    
    
# Default.
except:
    print("Something went wrong.")
