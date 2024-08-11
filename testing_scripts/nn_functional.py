'''
This test was made for the commit no. 11.
It shows the performance of a neural network with guessing an entire nonogram.

It makes a model with Keras's functional API in order to be able to list accuracies for each field.
Otherwise, it should work similarly to the nn_..._simple.py

It is very slow with larger nonograms.
It slows down somewhere in the fit method and I do not know why.
'''
import settings as s
from auxiliary_module import generate_training_data,make_empty_nonogram,set_shape
from auxiliary_module.testing import generate_testing_sample,reset_weights

import keras
import numpy as np
import tensorflow as tf

from itertools import product
from sys import exit




test_data,answer = generate_testing_sample(nonogram_frame_version=s.nonogram_frame_version)
nonogram = make_empty_nonogram()
nonogram_certainty = make_empty_nonogram().astype(float)
nonogram_accuracy = make_empty_nonogram().astype(float)

input_layer = keras.layers.Input(shape=[s.n_dimensions])
norm_layer = keras.layers.Normalization()
x = norm_layer(input_layer)
x = keras.layers.Dense(1_500,activation=s.activation)(x)

outputs = []
loss_dict = {}
metrics_dict = {}
for i in range(1,s.size+1):
    outputs.append(keras.layers.Dense(1,activation=keras.activations.sigmoid,name=f'{i}_out')(x))
    loss_dict.update({f'{i}_out' : 'binary_crossentropy'})
    metrics_dict.update({f'{i}_out' : 'binary_accuracy'})

model = keras.Model(inputs=input_layer,outputs=outputs)

model.compile(
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
        for i in range(1,s.size+1):
            losses.append(logs['loss'])
            val_losses.append(logs['val_loss'])
            accuracy.append(logs[f'{i}_out_binary_accuracy'])
            val_accuracy.append(logs[f'val_{i}_out_binary_accuracy'])
        losses = sum(losses)/s.size
        val_losses = sum(val_losses)/s.size
        accuracy = sum(accuracy)/s.size
        val_accuracy = sum(val_accuracy)/s.size
        print(f'Epoch: {epoch},   loss: {losses:.2f},   val_loss: {val_losses:.2f},   accuracy: {accuracy:.2f},   val_accuracy: {val_accuracy:.2f}')



while True:
    data,target = generate_training_data(s.num)
    target = target.to_numpy()
    target_out = []
    #target_out = np.vstack()
    for i in range(s.size):
        target_out.append(target[:,i])
    norm_layer.adapt(data.to_numpy())
    
    #reshape = (int(np.ceil(shape[1]/2))*shape[0]+int(np.ceil(shape[0]/2))*shape[1])
    #target = target.reshape()
    #print(np.array(target_out))
    #exit(0)
    hist = model.fit(data,target_out,epochs=s.n_epochs,validation_split=s.validation_split,
        callbacks=[LogCallback(),
            keras.callbacks.EarlyStopping(
                'val_loss',min_delta=s.min_delta,patience=s.early_stop_patience,restore_best_weights=True
            )],
        verbose = 0) # silent = 0, default = "auto"
    break
    if ((hist.history['val_binary_accuracy'][-1] > early_stop_accuracy)
        or (len(hist.history['loss']) < early_stop_patience+3)):
            break

    
predict_proba = model.predict(test_data,verbose=0)
for i,proba in enumerate(predict_proba):
    proba = proba[0][0]
    row = i // s.shape[1]
    col = i - row * s.shape[1]
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
print('Number of correctly guessed fields: ', np.sum(nonogram == answer), '/', s.size)
