'''
This test was made for the version from commit no. 6.
It shows the performance of the models on one entire nonogram.
It works decently with 5x5 nonograms, but performs otherwise relatively poorly.
'''


import numpy as np

from sklearn import set_config
set_config(transform_output = "pandas")

from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from auxiliary_module import convert_generated_data_to_data_frames,generate_nonogram_data,make_empty_nonogram
from testing import generate_testing_sample




shape = (5,5)
size = shape[0] * shape[1]
num = 300
train_test_split = 200
default_decision_boundary = 0.98
decision_boundary = default_decision_boundary
max_n_iter = 1000
n_to_guess = size

#scaler = MinMaxScaler(feature_range=(-5,5))
scaler = StandardScaler()

nonogram_data,answer = generate_testing_sample(shape)


nonogram = make_empty_nonogram(shape)
for _ in range(5):
    cur_guessed = 0
    generate_nonogram_data(shape,num=num,template=nonogram)

    data,targets = convert_generated_data_to_data_frames(range(1,train_test_split), shape)
    data = scaler.fit_transform(data)
    nonogram_data_transformed = scaler.transform(nonogram_data)

    model = LogisticRegression()

    for i,j in product(range(shape[0]),range(shape[1])):
        if nonogram[i,j] != -1: continue
        target = targets[f'target_{i+1}_{j+1}'].copy()
        model.fit(data,target)
        ''' # all targets might have the same label -> uncomment if such error appears
        try:
            model.fit(data,target)
        except ValueError:
            n_to_guess = 0
            break
        '''
        if max(model.predict_proba(nonogram_data_transformed)[0]) > decision_boundary:
            nonogram[i,j] = model.predict(nonogram_data_transformed)[0]
            print(i,j)
            n_to_guess -= 1
            cur_guessed += 1
        if n_to_guess == 0:
            break
    if n_to_guess == 0:
            break
    if cur_guessed == 0:
         decision_boundary -= 0.03
    else:
         decision_boundary = default_decision_boundary
    print(cur_guessed)

print('Answer')
print(nonogram)
print()
print('Right answer')
print(answer)
print()
print('Number of correctly guessed fields: ', np.sum(nonogram == answer), '/', size)
