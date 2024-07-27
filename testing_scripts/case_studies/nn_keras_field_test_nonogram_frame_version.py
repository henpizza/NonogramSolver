'''
This test was made for the commit no. 10.
It is a copy of nn_keras_field_test_simple with an interesting configuration.
When nonogram_frame_version is v1, the learning is slower.
(Change nonogram_frame_version to 2 to see the difference.)
'''
import os
#os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Adding grandparent directory to path
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
grandparent = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent)



import keras
import numpy as np

from itertools import product

from auxiliary_module import convert_generated_data_to_data_frames,generate_nonogram_data,make_empty_nonogram,set_shape
from auxiliary_module.testing import generate_testing_sample,reset_weights



shape = (5,5)
nonogram_frame_version = 1
if nonogram_frame_version == 1:
    n_dimensions = (shape[0]+shape[1])*2
elif nonogram_frame_version == 2:
    n_dimensions = int(np.ceil(shape[0]/2))*shape[1] + int(np.ceil(shape[1]/2))*shape[0]
else:
    raise(ValueError(f"Unexistent nonogram_frame_version: {nonogram_frame_version}"))
size = shape[0]*shape[1]
num = 20_000
validation_split = 0.1
keras.utils.set_random_seed(0)
max_n_iter = 1000
n_epochs = 10_000
early_stop_accuracy = 0.98
early_stop_patience = 4
learning_rate = 0.01
activation = keras.activations.relu


set_shape(shape)

test_data,answer = generate_testing_sample(nonogram_frame_version=nonogram_frame_version)
nonogram = make_empty_nonogram()

model = keras.Sequential()
model.add(keras.layers.Input(shape=[n_dimensions]))
model.add(keras.layers.Normalization())
model.add(keras.layers.Dense(30,activation=activation))
model.add(keras.layers.Dense(30,activation=activation))
model.add(keras.layers.Dense(size,activation=keras.activations.sigmoid))
model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate,momentum=0.9),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()])


try_again = True

while (try_again):
    reset_weights(model)

    generate_nonogram_data(num=num)
    data,target = convert_generated_data_to_data_frames(range(1,num),nonogram_frame_version=nonogram_frame_version)
    hist = model.fit(data,target,epochs=n_epochs,validation_split=validation_split,
        #callbacks=[keras.callbacks.EarlyStopping('val_loss',min_delta=1e-4,patience=early_stop_patience)],
        verbose = 1) # silent = 0, default = "auto"
        
    predict_proba = model.predict(test_data,verbose=0)[0]
    for i,proba in enumerate(predict_proba):
        row = i // shape[1]
        col = i - row * shape[1]
        nonogram[row,col] = int(round(proba))

    print('Answer')
    print(nonogram)
    print()
    print('Right answer')
    print(answer)
    print()
    print('Number of correctly guessed fields: ', np.sum(nonogram == answer), '/', size)

    print('Should we try again? (y/n)',end=' ')
    if (input() != 'y'):
        try_again = False
    else:
        nonogram = make_empty_nonogram(shape)
