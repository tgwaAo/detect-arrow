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

features = np.load("sign_features.npy")
target = np.load("sign_target.npy")

kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(features, target)
score = gpc.score(features,target)
print("Accuracy of training data: ",score)

if (gpc.score(features,target) < 1.0):
    for i in range(len(features[:,1])):
        predict = gpc.predict([features[i,:]])
        
        if (predict != target[i]):
            print("Feature nbr: ",i," probability ",gpc.predict_proba([features[i,:]])[0])

with open("sign_ai.pkl","wb") as f:
    _pickle.dump(gpc,f)

print("done")
