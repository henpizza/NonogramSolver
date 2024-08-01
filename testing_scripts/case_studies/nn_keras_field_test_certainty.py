'''
This test was made for the commit no. 10.
It shows the performance of a neural network with guessing an entire nonogram.

Results:
- By running this particular script,
  it can be seen that the probabilities output by the sigmoid function are very unreliable.
  Some fields have high certainty, while being wrong.
  In fact, in this example, a field with the highest certainty is wrong (at least when run on my computer).
  Or, possibly, there could be other issues, but this case is something to keep in mind.

  Certainty was computed from the predicted probabilities as
  (PROBABILITY - 1/2) * 2

- Stopping early with val_loss takes much longer, while providing little additional precision.
  This observation needs to be verified under other circumstances, however.
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
n_dimensions = (shape[0]+shape[1])*2
size = shape[0]*shape[1]
num = 2000
validation_split = 0.1
keras.utils.set_random_seed(0)
max_n_iter = 1000
n_epochs = 100
early_stop_accuracy = 0.98
early_stop_patience = 4


set_shape(shape)

test_data,answer = generate_testing_sample()
nonogram = make_empty_nonogram()
nonogram_certainty = make_empty_nonogram().astype(float)

model = keras.Sequential()
model.add(keras.layers.Input(shape=[n_dimensions]))
model.add(keras.layers.Dense(100))
model.add(keras.layers.Dense(50))
model.add(keras.layers.Dense(25,activation=keras.activations.sigmoid))
model.compile(optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()])


try_again = True


while (try_again):
    reset_weights(model)

    while True:
        generate_nonogram_data(num=num)
        data,target = convert_generated_data_to_data_frames(range(1,num))
        hist = model.fit(data,target,epochs=n_epochs,validation_split=validation_split,
            callbacks=[keras.callbacks.EarlyStopping('val_loss',min_delta=0,patience=early_stop_patience)],
            verbose = 1) # silent = 0, default = "auto"
        if ((hist.history['val_binary_accuracy'][-1] > early_stop_accuracy)
            or (len(hist.history['loss']) < early_stop_patience+3)):
                break
        
    predict_proba = model.predict(test_data,verbose=0)[0]
    for i,proba in enumerate(predict_proba):
        row = i // shape[1]
        col = i - row * shape[1]
        nonogram[row,col] = int(round(proba))
        nonogram_certainty[row,col] = (max(proba,1-proba)-1/2)*2

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
