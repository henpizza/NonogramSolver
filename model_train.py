#!/usr/bin/env python3
"""
Created on Mon Jul  8 11:17:09 2024
"""

import pandas as pd
from os import chdir,getcwd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler

chdir("nonogram")

from auxiliary_module import NonogramFrame,convert_to_nonogram_frame


data = pd.read_csv('training_data.csv')

target = data['target'].copy()
data = data.drop('target',axis='columns')

model = LogisticRegression()
model.fit(data,target)

data.hist(bins=20)
plt.tight_layout()
plt.show()


score = []
pred_test = []
y_test = []
for k in range(301,305+1):
    x = convert_to_nonogram_frame(f'{k}')
    y = x['target'].copy()
    x = x.drop('target',axis='columns')
    
    pred = model.predict(x)
    pred_proba = model.predict_proba(x)
    y = y.to_numpy()
    
    #for i in range(len(y)):
    #    print(pred[i],pred_proba[i],y[i])
        
    pred_proba = list(map(max,pred_proba))
    #print(pred_proba)
    
    for i in range(len(pred_proba)):
        if (pred_proba[i] >= 0.8):
            score.append((pred[i],y[i]))
            pred_test.append(pred[i])
            y_test.append(y[i])
    #print(score)
    
    print(precision_score(y_test, pred_test,zero_division=0))
