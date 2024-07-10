#!/usr/bin/env python3
"""
Created on Wed Jul 10 12:57:05 2024
"""

import pandas as pd
from os import chdir,getcwd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler


from auxiliary_module import NonogramFrame,convert_generated_data_to_nonogram_frame,nonogram_data_generate

nonogram_data_generate((5,5),num=1000)


data = convert_generated_data_to_nonogram_frame(range(1,300), (5,5), (1,1))
print(data)

target = data['target'].copy()
data = data.drop('target',axis='columns')

model = LogisticRegression()
model.fit(data,target)



score = []
pred_test = []
y_test = []

x = convert_generated_data_to_nonogram_frame(range(301,1000), (5,5), (1,1))
print(x)
y = x['target'].copy()
x = x.drop('target',axis='columns')

pred = model.predict(x)
pred_proba = model.predict_proba(x)
pred_proba = list(map(max,pred_proba))
y = y.to_numpy()

for k in range(len(y)):
    if pred_proba[k] > 0.95:
        y_test.append(y[k])
        pred_test.append(pred[k])

print(len(y_test))
print(precision_score(y_test, pred_test,zero_division=0))
