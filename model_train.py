#!/usr/bin/env python3


import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler


from auxiliary_module import convert_generated_data_to_data_frames,nonogram_data_generate


shape = (5,5)

nonogram_data_generate(shape,num=1000)
data,target = convert_generated_data_to_data_frames(range(1,100), shape)
target = target['target_3_3'].copy()

model = LogisticRegression()
model.fit(data,target)



score = []
pred_test = []
y_test = []
nono_index = []

x,y = convert_generated_data_to_data_frames(range(500,1000), shape)
y = y['target_3_3'].copy()


pred = model.predict(x)
pred_proba = model.predict_proba(x)
pred_proba = list(map(max,pred_proba))
y = y.to_numpy()

for k in range(len(y)):
    if pred_proba[k] > 0.95:
        nono_index.append(k)
        y_test.append(y[k])
        pred_test.append(pred[k])


print('Guessed nonograms')
print(500+np.array(nono_index))
print('Percentage of nonograms with certainty greater than 95 %')
print(len(y_test)/5)
print('Number of 0s and 1s guessed from the above nonograms')
print(np.unique(np.array(pred_test),return_counts=True))
print('Precision')
print(precision_score(y_test, pred_test,zero_division=0))