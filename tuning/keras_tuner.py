'''
A script made for commit no. 12.

It tries to find the optimal parameters depending on the nonogram's size.
Keras's RandomSearch is used for this purpose.
The models also use one Conv1D layer.

'''

# Adding parent directory to path
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current_dir)
sys.path.append(parent)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import keras_tuner as kt
import keras
import numpy as np
import pandas as pd
from shutil import rmtree

from auxiliary_module import generate_training_data,get_shape_info,set_shape

from sys import exit





    

class TuningHyperModelRandom(kt.HyperModel):
    def build(self,hp: kt.HyperParameters):
        n_neurons = hp.Int("n_neurons",min_value=1,max_value=15_000,sampling="log")
        has_filters = hp.Boolean("has_filters")
        if (has_filters):
            n_filters = hp.Int("n_filters",min_value=2,max_value=5_000,sampling="log")
        n_layers = hp.Int("n_layers",min_value=1,max_value=3)

        n_dimensions,size = get_shape_info()

        model = keras.Sequential()
        model.add(keras.layers.Input(shape=[1,n_dimensions]))
        if (has_filters):
            model.add(keras.layers.Conv1D(n_filters,shape[0],padding="same"))
        model.add(keras.layers.Flatten())
        for _ in range(n_layers):
            model.add(keras.layers.Dense(n_neurons))
        model.add(keras.layers.Dense(size,activation=keras.activations.sigmoid))
        model.compile(
            optimizer=keras.optimizers.Adam(beta_2=0.99),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy()])
        return model

    def fit(self,hp: kt.HyperParameters,model,val_data,val_target,**kwargs):
        data_size = hp.Int("data_size",min_value=500,max_value=10_000,sampling="log")
        min_delta = hp.Float("min_delta",min_value=1e-6,max_value=1e-1,sampling="log")

        data,target = generate_training_data(data_size)
        data = data.to_numpy().reshape((data.shape[0],1,data.shape[1]))

        return model.fit(data,target,epochs=10_000,
            validation_data=[val_data,val_target],
            callbacks=[
                keras.callbacks.EarlyStopping(
                'val_loss',min_delta=min_delta,patience=5,restore_best_weights=True
                )] + kwargs['callbacks'],
            verbose = 1)







max_trials = 150

for i in range(5,50+1):
    shape = (i,i)
    size = shape[0]*shape[1]
    n_dimensions = int(np.ceil(shape[0]/2))*shape[1] + int(np.ceil(shape[1]/2))*shape[0]
    set_shape(shape)
    print(shape)

    search_tuner = kt.RandomSearch(TuningHyperModelRandom(),objective="val_loss",max_trials=max_trials,
        directory="tuner_models",project_name=f"random_search_{shape[0]}_{shape[1]}",
        overwrite=False,
        seed=0)
    
    val_data,val_target = generate_training_data(3_000)
    val_data = val_data.to_numpy().reshape((val_data.shape[0],1,val_data.shape[1]))
    search_tuner.search(val_data,val_target)

    best_hp = search_tuner.oracle.get_best_trials(10_000)
    df = pd.DataFrame(columns=list(best_hp[0].hyperparameters.values.keys())+['score','accuracy_best','accuracy_last','epochs'])
    for trial in best_hp:
        hp = trial.hyperparameters.values
        score = trial.score
        accuracy_best = trial.metrics.get_best_value('val_binary_accuracy')
        accuracy_last = trial.metrics.get_last_value('val_binary_accuracy')
        epochs = trial.get_state()['best_step']
        hp.update({'score': score,'accuracy_best':accuracy_best, 'accuracy_last':accuracy_last,'epochs':epochs})
        df.loc[len(df)] = hp

    #print(df)
    df.to_csv(f'tuner_models/{shape[0]}_{shape[1]}_random.csv')
    rmtree(f'tuner_models/random_search_{shape[0]}_{shape[1]}')