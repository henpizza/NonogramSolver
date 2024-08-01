# Adding parent directory to path
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_dir)
sys.path.append(parent)

#os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


from auxiliary_module import set_shape

import numpy as np
import keras

from sklearn.model_selection import train_test_split

# Common

shape = (13,8)
nonogram_frame_version = 2
if nonogram_frame_version == 1:
    n_dimensions = (shape[0]+shape[1])*2
elif nonogram_frame_version == 2:
    n_dimensions = int(np.ceil(shape[0]/2))*shape[1] + int(np.ceil(shape[1]/2))*shape[0]
else:
    raise(ValueError(f"Unexistent nonogram_frame_version: {nonogram_frame_version}"))
size = shape[0]*shape[1]
num = 10_000
max_sequences_in_row = np.ceil(shape[1]/2)
max_sequences_in_col = np.ceil(shape[0]/2)
max_n_iter = 1000


# Logistic regression

#train_test_split = 100
default_decision_boundary = 0.98
decision_boundary = default_decision_boundary
min_decision_boundary = 0.9
decision_boundary_decrement = 0.01
default_accuracy_decision_boundary = 0.99
accuracy_decision_boundary = default_accuracy_decision_boundary
accuracy_decision_boundary_decrement = 0.005
cv_min_num_decided = 2
n_to_guess = size

# Neural networks

validation_split = 0.1
n_epochs = 1_000
early_stop_accuracy = 0.98
early_stop_patience = 4
learning_rate = 0.01
activation = keras.activations.relu
early_stop_accuracy = 0.98
early_stop_patience = 5
batch_size = 32
min_delta = 1e-5
verbose = 1 # silent = 0, default = "auto"


seed = 0
keras.utils.set_random_seed(seed)
set_shape(shape)


# Neural networks model used across several files

model = keras.Sequential()
norm_layer = keras.layers.Normalization()
model.add(keras.layers.Input(shape=[n_dimensions]))
model.add(norm_layer)
#model.add(keras.layers.BatchNormalization()) # works worse than normalisation?
model.add(keras.layers.Dense(1_500,activation=activation))
#model.add(keras.layers.BatchNormalization()) # works worse than normalisation?
model.add(keras.layers.Dense(size,activation=keras.activations.sigmoid))
model.compile(
    optimizer=keras.optimizers.Adam(beta_2=0.99),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()])

def fit(data,target):
    data,data_val,target,target_val = train_test_split(data,target,test_size=0.1)
    norm_layer.adapt(data.to_numpy())
    hist = model.fit(data,target,epochs=n_epochs,
        validation_data=[data_val,target_val],
        callbacks=[
            keras.callbacks.EarlyStopping(
            'val_loss',min_delta=min_delta,patience=early_stop_patience,restore_best_weights=True
            )],
        verbose = verbose)
    return hist
