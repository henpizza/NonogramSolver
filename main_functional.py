'''
The main script that should be used by the end-user.
Run the file with python for instructions on how the program should be used.
'''

import keras
import numpy as np
from rich import print
from rich import text
import rich
import pathlib
from sklearn.model_selection import train_test_split
import sys

from auxiliary_module import get_shape,get_shape_info,set_shape_from_file
from auxiliary_module import convert_to_nonogram_frame2,generate_training_data,make_empty_nonogram
from auxiliary_module.testing import keras_nonogram_max_proba_fill

argv = sys.argv

HELP = \
'''Usage:
[green]main.py [FILE] [--settings or -s SETTINGS]
    [--help or -h [help_file]]
    [--out or -o output_file]
[/green]
[green]FILE[/green] - Input file.
[green]SETTINGS[/green] - Optional file with settings.

[green]help_file[/green] - When this option is set, an example of an input file is printed.
    Possible values: 'input' 'settings'
'''

INPUT_HELP = \
'''Example 'input.non' file:
\t---BEGIN---[green]
1,1,1
4
1,3
1,3
-
1
2,1
1
4
3
1,2
-
[/green]\t---END---

This file represents a nonogram with 4 rows and 6 columns.
The two sections (rows and columns) are delimited with a single '-'. The second '-' is also necessary.
A comma ',' needs to be put between the input numbers.
Insert an empty line for a row or a column with no 1s in it.
'''

SETTINGS_HELP = \
'''Example 'settings' file:
\t---BEGIN---[green]
n_neurons=1_000
has_filters=True
n_filters=10
n_layers=1
training_data_size=10_000
[/green]\t---END---

The program uses the following model:
- Input
- Conv1D (settings: n_filters, has_filters)
    - Kernel size is max(num_of_rows, num_of_columns)
    - Zero padding is used
- Dense layer (settings: n_neurons)
- (More dense layers if n_layers > 1)
- Output

Explanations:
[green]has_filters[/green] - Whether to use filters (Conv1D layer) or not (default is False)
[green]n_filters[/green] - Number of filters in the Conv1D layer
[green]n_neurons[/green] - Number of neurons per layer (by default only one layer is used)
[green]n_layers[/green] - Number of dense layers
[green]training_data_size[/green] - Size of the training data that will be generated before each guess of a field

No other settings are currently supported.
'''

# Custom callback
class LogCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        losses = []
        val_losses = []
        accuracy = []
        val_accuracy = []
        for i in range(1,size+1):
            accuracy.append(logs[f'{i}_out_binary_accuracy'])
            val_accuracy.append(logs[f'val_{i}_out_binary_accuracy'])
        losses.append(logs['loss'])
        val_losses.append(logs['val_loss'])
        losses = sum(losses)/size
        val_losses = sum(val_losses)/size
        accuracy = sum(accuracy)/size
        val_accuracy = sum(val_accuracy)/size
        print(f'Epoch: {epoch},   loss: {losses:.2f},   val_loss: {val_losses:.2f},   accuracy: {accuracy:.2f},   val_accuracy: {val_accuracy:.2f}')

# Print help
if len(argv) == 1:
    print(HELP)
    exit(0)
if '--help' in argv or '-h' in argv:
    if '--help' in argv:
        help_index = argv.index('--help')
    else:
        help_index = argv.index('-h')
    if help_index < len(argv)-1:
        help_file = argv[help_index+1]
        match help_file:
            case 'input':
                print(INPUT_HELP)
            case 'settings':
                print(SETTINGS_HELP)
            case _:
                print(HELP)
                print("[red]ERROR:[/red] Invalid help_file.")
    else:
        print(HELP)
    sys.exit(0)

has_filters = False
n_neurons = 500
n_filters = 200
n_layers = 2
training_data_size = 5_000

is_file_loaded = False
out_filename = None

# Process argv
argv_index = 1
while argv_index < len(argv):
    cur_argv = argv[argv_index]
    argv_index += 1
    match cur_argv:
        case x if x in ('-s','--settings'):
            next_argv = argv[argv_index]
            argv_index += 1
            with open(next_argv) as settings_file:
                for line in settings_file:
                    split = line.split('=')
                    if len(split) == 2:
                        setting,val = line.split('=')
                    elif len(line) == 1:
                        pass
                    else:
                        print(SETTINGS_HELP)
                        print("[red]ERROR:[/red] Invalid settings file.")
                    match setting:
                        case 'has_filters':
                            has_filters = False if val == 'False' else True
                        case 'n_neurons':
                            n_neurons = int(val)
                        case 'n_filters':
                            n_filters = int(val)
                        case 'n_layers':
                            n_layers = int(val)
                        case 'training_data_size':
                            training_data_size = int(val)
        case x if x in ('-o','--output'):
            next_argv = argv[argv_index]
            argv_index += 1
            out_filename = next_argv
        case _:
            try:
                if not is_file_loaded:
                    set_shape_from_file(cur_argv)
                    input_data = convert_to_nonogram_frame2([cur_argv])
                    print('File ' + cur_argv + ' was read successfully')
                    is_file_loaded = True
                else:
                    print("[red]ERROR:[/red] Only one input file is currently supported.")
                    exit(1)
            except FileNotFoundError:
                print(HELP)
                print(f"[red]ERROR:[/red] File '{cur_argv}' not found.")
                print("Current directory: " + str(pathlib.Path().resolve()))
                exit(2)
            except OSError as e:
                print(HELP)
                print("OSError:" + e)
                exit(1)
            except ValueError as e:
                print(INPUT_HELP)
                print(f"[red]ValueError:[/red] {e}")
                print("Did your input file have the right format?")
                exit(1)
            except EOFError:
                print(INPUT_HELP)
                print("[red]EOFError:[/red] End of file reached.")
                print("Did your input file have the right format?")
                print("Please make sure that a single '-' is between the row and column sections and below the second (column) section.")
                exit(1)

# Define constants
max_n_iter = 10_000
shape = get_shape()
n_dimensions,size = get_shape_info()

validation_split = 0.1
n_epochs = 1_000
#learning_rate = 0.01
activation = keras.activations.relu
early_stop_patience = 5
min_delta = 1e-5

# Make the model
input_layer = keras.layers.Input(shape=[1,n_dimensions])
x = input_layer
if (has_filters):
    x = keras.layers.Conv1D(n_filters,max(shape[0],shape[1]),padding="same")(x)
x = keras.layers.Flatten()(x)
for _ in range(n_layers):
    x = keras.layers.Dense(n_neurons,activation=activation)(x)
# The last layer has to have as many outputs as "layers" if we want accuracy not to be averaged over all outputs
outputs = []
loss_dict = {}
metrics_dict = {}
for i in range(1,size+1):
    outputs.append(keras.layers.Dense(1,activation=keras.activations.sigmoid,name=f'{i}_out')(x))
    loss_dict.update({f'{i}_out' : 'binary_crossentropy'})
    metrics_dict.update({f'{i}_out' : 'binary_accuracy'})
model = keras.Model(inputs=input_layer,outputs=outputs)
model.compile(
    optimizer=keras.optimizers.Adam(beta_2=0.99),
    loss=loss_dict,
    metrics=metrics_dict)

def fit(data,target):
    data,data_val,target,target_val = train_test_split(data,target,test_size=0.1)
    # 
    target = target.to_numpy()
    target_out = []
    #target_out = np.vstack()
    for i in range(size):
        target_out.append(target[:,i].view())
    target = target_out
    #
    target_val = target_val.to_numpy()
    target_out = []
    #target_out = np.vstack()
    for i in range(size):
        target_out.append(target_val[:,i].view())
    target_val = target_out
    #
    '''
    target = target.to_numpy()
    target_val = target_val.to_numpy()
    target = target.T.reshape((size,1,target.shape[0])).T
    target_val = target_val.T.reshape((size,1,target_val.shape[0])).T
    '''
    hist = model.fit(data,target,epochs=n_epochs,
        validation_data=[data_val,target_val],
        callbacks=[LogCallback(),
            keras.callbacks.EarlyStopping(
            'val_loss',min_delta=min_delta,patience=early_stop_patience,restore_best_weights=True
            )],
        verbose = 0)
    return hist

input_data = input_data.to_numpy().reshape((input_data.shape[0],1,input_data.shape[1]))
nonogram = make_empty_nonogram()
nonogram_accuracy = make_empty_nonogram().astype(float)
n_to_guess = size


# Fill the nonogram
for _ in range(max_n_iter):
    if (n_to_guess == 0):
        break
    data,target = generate_training_data(training_data_size,template=nonogram)
    data = data.to_numpy().reshape((data.shape[0],1,data.shape[1]))
    hist = fit(data,target)

    predict_proba = np.array(model.predict(input_data,verbose=0))
    predict_proba = predict_proba.flatten()
    for i in range(size):
        row = i // shape[1]
        col = i - row * shape[1]
        nonogram_accuracy[row,col] = hist.history[f'val_{i+1}_out_binary_accuracy'][-1]
    max_row,max_col = keras_nonogram_max_proba_fill(predict_proba,nonogram,nonogram_accuracy)
    n_to_guess -= 1

    print()
    for line in nonogram:
        for num in line:
            if num == 1:
                print('[red]1[/red]',end=' ')
            elif num == 0:
                print('[blue]0[/blue]', end=' ')
            else:
                print('[grey78]X[/grey78]',end=' ')
        print()
    print("Guessed " + str(nonogram[max_row,max_col]))
    print(max_row+1,max_col+1,f'({n_to_guess} more to fill)')
    print()

# Print the answer
print('[blue]===================[/blue]')
print('[blue]Answer[/blue]')
for line in nonogram:
    for num in line:
        if num == 1:
            print('[red]1[/red]',end=' ')
        else:
            print('[grey78]0[/grey78]',end=' ')
    print()

# Print output to file
if out_filename is not None:
    with open(out_filename,'w') as out_file:
        for line in nonogram:
            for num in line:
                if num == 1:
                    print('[red]1[/red]',end=' ',file=out_file)
                else:
                    print('[grey78]0[/grey78] ',end=' ',file=out_file)
            print(file=out_file)

