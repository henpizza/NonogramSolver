'''
This test was made for the commit no. 10.
It is the first attempt at using a neural network.
It splits the training phase and the second (predicting) phase,
while being as simple as possible.
This is the first (training) part.
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

from auxiliary_module import convert_generated_data_to_data_frames,generate_nonogram_data


shape = (5,5)
size = shape[0]*shape[1]
n_dimensions = (shape[0]+shape[1])*2
num = 500
validation_split = 0.3
keras.utils.set_random_seed(0)



model = keras.Sequential()
model.add(keras.layers.Input(shape=[n_dimensions]))
model.add(keras.layers.Dense(100))
model.add(keras.layers.Dense(50))
model.add(keras.layers.Dense(25,activation=keras.activations.sigmoid))
model.compile(optimizer="sgd",
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()])

generate_nonogram_data(shape,num=num)
data,target = convert_generated_data_to_data_frames(range(1,num),shape)

model.fit(data,target,epochs=400,validation_split=validation_split)


keras.saving.save_model(model,os.path.join(current_dir,"model.keras"))
