'''
This test was made for the version from commit no. 5.
It shows the performance of the models on their *first* guesses.
'''


import numpy as np

from sklearn import set_config
set_config(transform_output = "pandas")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler




from auxiliary_module import convert_generated_data_to_data_frames,generate_nonogram_data


shape = (15,15)
position = '1_1'
num = 1000
train_test_split = 500
decision_boundary = 0.99

#scaler = MinMaxScaler(feature_range=(-5,5))
scaler = StandardScaler()

generate_nonogram_data(shape,num=num)
data,target = convert_generated_data_to_data_frames(range(1,500), shape)
data = scaler.fit_transform(data)
target = target['target_' + position].copy()

model = LogisticRegression()
model.fit(data,target)



score = []
pred_test = []
y_test = []
nono_index = []

x,y = convert_generated_data_to_data_frames(range(train_test_split,num), shape)
x = scaler.transform(x)
y = y['target_' + position].copy()

pred = model.predict(x)
pred_proba = model.predict_proba(x)
pred_proba = list(map(max,pred_proba))
y = y.to_numpy()

for k in range(len(y)):
    if pred_proba[k] > decision_boundary:
        nono_index.append(k)
        y_test.append(y[k])
        pred_test.append(pred[k])


print('Nonograms, whose entries were guessed with certainty greater than 95 %')
print(train_test_split+np.array(nono_index))
print('Percentage of those nonograms over the whole testing set')
print(len(y_test)/(num-train_test_split)*100)
print('Number of 0s and 1s guessed from the above nonograms')
print(np.unique(np.array(pred_test),return_counts=True))
print('Accuracy')
if (len(pred_test) != 0):
    print(accuracy_score(y_test, pred_test))
else:
    print(0.0)
