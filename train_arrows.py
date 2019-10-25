#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Published under GNU General Public License v3.0.

You should have recieved a copy of the license GNU GPLv3.
If not, see
http://www.gnu.org/licenses/

Train ai with features.

@author: Uwe Simon
"""

import _pickle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from sklearn.model_selection import train_test_split

###########################################################
# Load training data and concatenate them.
##########################################################
features = np.load("arrow_features.npy")
target = np.load("arrow_target.npy")

#############################################################
# Get number of arrows and non arrows for comparison.
#############################################################
num_one = np.count_nonzero(target)
print("Number of arrows ", num_one)
print("Number of non arrows ", len(target) - num_one)

#############################################################
# Split data for training and testing, start training.
#############################################################
x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.2)

kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(x_train,
                                                                   y_train)

##############################################################
# Testing and saving.
##############################################################
score = gpc.score(x_train, y_train)
print("Accuracy of training data: ", score)
print('Accuracy on the testing subset:', format(gpc.score(x_test, y_test)))

with open("arrow_ai.pkl", "wb") as f:
    _pickle.dump(gpc, f)

print("done")
