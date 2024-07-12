'''
This test was made for the versions from commits no. 6 to no. 7.
It shows the performance of the models on one entire nonogram.
It works decently with 5x5 nonograms, but performs otherwise relatively poorly.
'''


import numpy as np

from sklearn import set_config
set_config(transform_output = "pandas")

from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from auxiliary_module import convert_generated_data_to_data_frames,generate_nonogram_data,make_empty_nonogram
from testing import generate_testing_sample


shape = (5,5)
size = shape[0] * shape[1]
num = 150
train_test_split = 100
default_decision_boundary = 0.98
decision_boundary = default_decision_boundary
min_decision_boundary = 0.9
decision_boundary_decrement = 0.01
default_accuracy_decision_boundary = 0.99
accuracy_decision_boundary = default_accuracy_decision_boundary
accuracy_decision_boundary_decrement = 0.005
cv_min_num_decided = 2
max_n_iter = 1000
n_to_guess = size

#scaler = MinMaxScaler(feature_range=(0,3))
scaler = StandardScaler()

nonogram_data,answer = generate_testing_sample(shape)


nonogram = make_empty_nonogram(shape)

try_again = True

model = LogisticRegression()

while (try_again):
    for _ in range(max_n_iter):
        cur_guessed = 0
        generate_nonogram_data(shape,num=num,template=nonogram)

        data,targets = convert_generated_data_to_data_frames(range(1,train_test_split), shape)
        cv_data,cv_targets = convert_generated_data_to_data_frames(range(train_test_split,num),shape)
        data = scaler.fit_transform(data)
        nonogram_data_transformed = scaler.transform(nonogram_data)
        cv_data = scaler.transform(cv_data)


        for i,j in product(range(shape[0]),range(shape[1])):
            if nonogram[i,j] != -1: continue  # If already filled

            target = targets[f'target_{i+1}_{j+1}'].copy()
            cv_target = cv_targets[f'target_{i+1}_{j+1}'].copy().to_numpy()
            
            try: # all targets might have the same label (although it is not probable)
                model.fit(data,target)
            except ValueError:
                continue

            # Cross validate
            cv_pred = model.predict(cv_data)
            cv_pred_proba = model.predict_proba(cv_data)
            cv_pred_proba = list(map(max,cv_pred_proba))

            cv_pred_total_num = len(cv_pred)
            cv_pred_decided = []
            cv_decided_true = []
            for k in range(cv_pred_total_num):
                if cv_pred_proba[k] > decision_boundary:
                    cv_pred_decided.append(cv_pred[k])
                    cv_decided_true.append(cv_target[k])
            
            if (len(cv_pred_decided) < cv_min_num_decided):
                continue
            accuracy = accuracy_score(cv_decided_true,cv_pred_decided)
            if (accuracy <= accuracy_decision_boundary):
                continue

            # Fill some fields
            if max(model.predict_proba(nonogram_data_transformed)[0]) > decision_boundary:
                nonogram[i,j] = model.predict(nonogram_data_transformed)[0]
                print(i+1,j+1)
                n_to_guess -= 1
                cur_guessed += 1
            if n_to_guess == 0:
                break

        print(cur_guessed)
        print()
        if n_to_guess == 0:
                break
        if cur_guessed == 0:
            decision_boundary -= decision_boundary_decrement
            if (decision_boundary < min_decision_boundary):
                break
            accuracy_decision_boundary -= accuracy_decision_boundary_decrement
        else:
            decision_boundary = default_decision_boundary
            accuracy_decision_boundary = default_accuracy_decision_boundary


    print('Answer')
    print(nonogram)
    print()
    print('Right answer')
    print(answer)
    print()
    print('Number of correctly guessed fields: ', np.sum(nonogram == answer), '/', size-n_to_guess, f' (from a total of {size})')

    print('Should we try again? (y/n)',end=' ')
    if (input() != 'y'):
        try_again = False
    else:
        nonogram = make_empty_nonogram(shape)
        n_to_guess = size
        decision_boundary = default_decision_boundary
        accuracy_decision_boundary = default_accuracy_decision_boundary