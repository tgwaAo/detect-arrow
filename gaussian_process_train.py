#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:06:27 2019

Train ai with features an find bad training data.

@author: me
"""

import _pickle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from sklearn.model_selection import train_test_split

features = np.load("sign_features.npy")
target = np.load("sign_target.npy")

x_train, x_test, y_train, y_test = train_test_split(features,target,test_size=0.07)

kernel = 1.0 * RBF(1.0)
#gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(features, target)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(x_train, y_train)

score = gpc.score(features,target)
print("Accuracy of training data: ",score)
print('Accuracy on the testing subset:(:.3f)',format(gpc.score(x_test,y_test)))


if (gpc.score(features,target) < 1.0):
    for i in range(len(features[:,1])):
        predict = gpc.predict([features[i,:]])
        
        if (predict != target[i]):
            print("Feature nbr: ",i," probability ",gpc.predict_proba([features[i,:]])[0])

with open("sign_ai.pkl","wb") as f:
    _pickle.dump(gpc,f)

print("done")
