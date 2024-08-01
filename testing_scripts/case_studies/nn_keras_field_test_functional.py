'''
This test was made for the commit no. 11.

In this script, increasing the size of the nonogram causes overfitting.
'''
import os
#os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Adding parent directory to path
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_dir)
sys.path.append(parent)



import keras
import numpy as np

from itertools import product
from sys import exit

from auxiliary_module import convert_generated_data_to_data_frames,generate_nonogram_data,make_empty_nonogram,set_shape
from auxiliary_module.testing import generate_testing_sample,reset_weights



shape = (20,15)
nonogram_frame_version = 2
if nonogram_frame_version == 1:
    n_dimensions = (shape[0]+shape[1])*2
elif nonogram_frame_version == 2:
    n_dimensions = int(np.ceil(shape[0]/2))*shape[1] + int(np.ceil(shape[1]/2))*shape[0]
else:
    raise(ValueError(f"Unexistent nonogram_frame_version: {nonogram_frame_version}"))
size = shape[0]*shape[1]
num = 3_000
validation_split = 0.1
keras.utils.set_random_seed(0)
max_n_iter = 1000
n_epochs = 300
early_stop_accuracy = 0.98
early_stop_patience = 4
learning_rate = 0.01
activation = keras.activations.relu
early_stop_accuracy = 0.98
early_stop_patience = 4


set_shape(shape)

test_data,answer = generate_testing_sample(nonogram_frame_version=nonogram_frame_version)
nonogram = make_empty_nonogram()
nonogram_certainty = make_empty_nonogram().astype(float)
nonogram_accuracy = make_empty_nonogram().astype(float)

input_layer = keras.layers.Input(shape=[n_dimensions])
x = keras.layers.Normalization()(input_layer)
#x = keras.layers.Dense(n_dimensions,activation=activation)(x)
#x = keras.layers.Dense(n_dimensions,activation=activation)(x)
#x = keras.layers.Dense(n_dimensions,activation=activation)(x)

outputs = []
loss_dict = {}
metrics_dict = {}
for i in range(1,size+1):
    outputs.append(keras.layers.Dense(1,activation=keras.activations.sigmoid,name=f'{i}_out')(x))
    loss_dict.update({f'{i}_out' : 'binary_crossentropy'})
    metrics_dict.update({f'{i}_out' : 'binary_accuracy'})

model = keras.Model(inputs=input_layer,outputs=outputs)

model.compile(#optimizer=keras.optimizers.SGD(learning_rate=learning_rate,momentum=0.9,nesterov=True),
    optimizer=keras.optimizers.Adam(beta_2=0.9),
    loss=loss_dict,
    metrics=metrics_dict)


try_again = True

class LogCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        losses = []
        val_losses = []
        accuracy = []
        val_accuracy = []
        for i in range(1,size+1):
            losses.append(logs['loss'])
            val_losses.append(logs['val_loss'])
            accuracy.append(logs[f'{i}_out_binary_accuracy'])
            val_accuracy.append(logs[f'val_{i}_out_binary_accuracy'])
        losses = np.average(losses)
        val_losses = np.average(val_losses)
        accuracy = np.average(accuracy)
        val_accuracy = np.average(val_accuracy)
        print(f'Epoch: {epoch},   loss: {losses:.2f},   val_loss: {val_losses:.2f},   accuracy: {accuracy:.2f},   val_accuracy: {val_accuracy:.2f}')


while (try_again):
    reset_weights(model)

    while True:
        generate_nonogram_data(num=num)
        data,target = convert_generated_data_to_data_frames(range(1,num),nonogram_frame_version=nonogram_frame_version)
        target_out = []
        target = target.to_numpy()
        for i in range(size):
            target_out.append(target[:,i])
        hist = model.fit(data,target_out,epochs=n_epochs,validation_split=validation_split,
            #callbacks=[keras.callbacks.EarlyStopping(min_delta=0,patience=early_stop_patience)],
            callbacks=[LogCallback()],
            verbose = 0) # silent = 0, default = "auto"
        break
        if ((hist.history['val_binary_accuracy'][-1] > early_stop_accuracy)
            or (len(hist.history['loss']) < early_stop_patience+3)):
                break
    
        
    predict_proba = model.predict(test_data,verbose=0)
    for i,proba in enumerate(predict_proba):
        proba = proba[0][0]
        row = i // shape[1]
        col = i - row * shape[1]
        nonogram[row,col] = int(round(proba))
        nonogram_certainty[row,col] = (max(proba,1-proba)-1/2)*2
        nonogram_accuracy[row,col] = hist.history[f'val_{i+1}_out_binary_accuracy'][-1]

    print()
    print('Accuracy')
    with np.printoptions(precision=2,suppress=True):
        print(nonogram_accuracy)
    print()
    print('Certainty')
    with np.printoptions(precision=2,suppress=True):
        print(nonogram_certainty)
    print()
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
