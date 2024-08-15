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
from auxiliary_module import UNKNOWN
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
forgetting_enabled=True
has_filters=True
interval_between_fillings=25
learning_rate_exp=3
print_tmp_results=True
print_tmp_frequency=10
n_neurons=500
n_filters=100
n_layers=1
n_models=2
to_forget=5
training_data_size=5000
verbose=False
[/green]\t---END---

The program uses the following model(s):
- Input
- Conv1D (settings: n_filters, has_filters)
    - Kernel size is max(num_of_rows, num_of_columns)
    - Zero padding is used
- Dense layer (settings: n_neurons)
- (More dense layers if n_layers > 1)
- Output

Explanations:
[green]forgetting_enabled[/green] - Whether to allow the models to forget some fields (also see to_forget)
[green]has_filters[/green] - Whether to use filters (Conv1D layer) or not (default is True)
[green]learning_rate_exp[/green] - Learning rate is set to 10**(-learning_rate_exp)
[green]n_filters[/green] - Number of filters in the Conv1D layer
[green]n_models[/green] - Number of models that will fill the nonogram (also see ensemble learning)
[green]n_neurons[/green] - Number of neurons per layer
[green]n_layers[/green] - Number of dense layers (by default only one layer is used)
[green]print_tmp_results[/green] - Whether to print intermediate results (also see print_tmp_frequency)
[green]print_tmp_frequency[/green] - How often to print intermediate results. The value corresponds to the number of filled fields.
    (set print_tmp_results to True for this option to take effect.)
[green]to_forget[/green] - How many fields to forget during each filling (needs forgetting_enabled=True)
[green]training_data_size[/green] - Size of the training data that will be generated before each guess of a field (default: 30_000)
[grren]verbose[/green] - How verbose the Keras's output should be. True is equivalent to 1 in Keras.

No other settings are currently supported.
'''

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

# Default settings
forgetting_enabled = True
has_filters = True
interval_between_fillings = 25#size // 5
learning_rate_exp = 3
n_neurons = 500
n_filters = 100
n_layers = 1
n_models = 2
print_tmp_results = True
print_tmp_frequency = 5
to_forget = 5
training_data_size = 5000
verbose = False

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
                            has_filters = False if val.casefold() == 'false' else True
                        case 'forgetting_enabled':
                            forgetting_enabled = False if val.casefold() == 'false' else True
                        case 'interval_between_fillings':
                            interval_between_fillings = int(val)
                        case 'learning_rate_exp':
                            learning_rate_exp = int(val)
                        case 'n_models':
                            n_models = int(val)
                        case 'n_neurons':
                            n_neurons = int(val)
                        case 'n_filters':
                            n_filters = int(val)
                        case 'n_layers':
                            n_layers = int(val)
                        case 'print_tmp_frequency':
                            print_tmp_frequency = int(val)
                        case 'print_tmp_results':
                            print_tmp_results = True if val.casefold() == 'true' else False
                        case 'to_forget':
                            to_forget = int(val)
                        case 'training_data_size':
                            training_data_size = int(val)
                        case 'verbose':
                            verbose = False if val.casefold() == 'false' else True
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
                    print()
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

# Print current settings
print("[yellow]Settings:[/yellow]")
print(f"{forgetting_enabled=}")
print(f"{has_filters=}")
print(f"{interval_between_fillings=}")
print(f"{learning_rate_exp=}")
print(f"{n_filters=}")
print(f"{n_models=}")
print(f"{n_layers=}")
print(f"{n_neurons=}")
print(f"{to_forget=}")
print(f"{training_data_size=}")
print()

# Define constants
max_n_iter = 10_000
shape = get_shape()
n_dimensions,size = get_shape_info()
rng = np.random.default_rng()

# Define more constants
validation_split = 0.1
n_epochs = 1_000
#learning_rate = 0.01
activation = keras.activations.relu
early_stop_patience = 5
min_delta = 1e-7
if verbose:
    verbose = 1
else:
    verbose = 0

# Make the models
models = [keras.Sequential() for _ in range(n_models)]
for i in range(n_models):
    model = models[i]
    model.add(keras.layers.Input(shape=[1,n_dimensions]))
    if (has_filters):
        model.add(keras.layers.Conv1D(n_filters,max(shape[0],shape[1]),padding="same"))
    for _ in range(n_layers):
        model.add(keras.layers.Dense(n_neurons,activation=activation))
    model.add(keras.layers.Dense(size,activation=keras.activations.sigmoid))
    model.add(keras.layers.Flatten())
    model.compile(
        optimizer=keras.optimizers.Adam(beta_2=0.99,learning_rate=10**(-learning_rate_exp)),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()])

# Convenience function
def fit(data,target,model_index):
    model = models[model_index]
    data,data_val,target,target_val = train_test_split(data,target,test_size=0.1)
    hist = model.fit(data,target,epochs=n_epochs,
        validation_data=[data_val,target_val],
        callbacks=[
            keras.callbacks.EarlyStopping(
            'val_loss',min_delta=min_delta,patience=early_stop_patience,restore_best_weights=True
            )],
        verbose = verbose)
    return hist


# Define variables (and some related constants)
input_data = input_data.to_numpy().reshape((input_data.shape[0],1,input_data.shape[1]))
nonograms = [make_empty_nonogram() for _ in range(n_models)]
nonogram_answer = make_empty_nonogram()
n_to_guess = size
n_to_guess_tmp = n_to_guess
iterations_left_until_filling = interval_between_fillings
no_filling_tolerance = 2 
no_filling_count = 0
print_tmp_counter = 0


# Fill the nonogram
for _ in range(max_n_iter):
    print_tmp_counter += 1 # Has meaning only if print_tmp_results is True
    iterations_left_until_filling -= 1
    if (n_to_guess == 0):
        break
    # Fill some fields for each model
    for model_index in range(n_models):
        nonogram = nonograms[model_index]
        model = models[model_index]
        data,target = generate_training_data(training_data_size,template=nonogram)
        data = data.to_numpy().reshape((data.shape[0],1,data.shape[1]))
        fit(data,target,model_index)

        predict_proba = model.predict(input_data,verbose=0)[0]
        keras_nonogram_max_proba_fill(predict_proba,nonogram)
    n_to_guess_tmp -= 1

    # Make (almost) final decisions about which fields should be filled
    if n_to_guess_tmp == 0 or iterations_left_until_filling == 0:
        no_filling_count += 1 # This is nullified if a field is filled during an iteration.
        print_tmp_counter = 0
        iterations_left_until_filling = interval_between_fillings
        colored_fields = []
        filled_fields = [] # for forgetting
        forgotten_fields = []
        print()
        # Fill if all models agree on a value
        for row in range(shape[0]):
            for col in range(shape[1]):
                if nonogram_answer[row,col] != UNKNOWN:
                    filled_fields.append((row,col))
                    continue
                val = nonograms[0][row,col]
                if n_models == 1:
                    if val != UNKNOWN:
                        nonogram_answer[row,col] = val
                        filled_fields.append((row,col))
                        colored_fields.append((row,col))
                        n_to_guess -= 1
                    no_filling_count = 0 # This variable is used meaningfully only if n_models > 1
                elif val in (0,1):
                    same_value = True
                    for nonogram in nonograms[1:]:
                        if nonogram[row,col] != val:
                            same_value = False
                            break
                    if same_value:
                        nonogram_answer[row,col] = val
                        print("Guessed " + str(val) + " at " + str(row+1) + " " + str(col+1))
                        filled_fields.append((row,col))
                        colored_fields.append((row,col))
                        no_filling_count = 0
                        n_to_guess -= 1

        # Nonogram targets are not unique. In some cases one field will be filled even if models do not agree on the answer.
        if no_filling_count == no_filling_tolerance:
            break_loops = False
            no_filling_count = 0
            for row in range(shape[0]):
                for col in range(shape[1]):
                    if nonogram_answer[row,col] != UNKNOWN:
                        continue
                    else:
                        for nonogram in nonograms:
                            if nonogram[row,col] != UNKNOWN:
                                nonogram_answer[row,col] = nonogram[row,col]
                                print("Guessed " + str(nonogram_answer[row,col]) + " at " + str(row+1) + " " + str(col+1))
                                colored_fields.append((row,col))
                                filled_fields.append((row,col))
                                n_to_guess -= 1
                                break_loops = True
                                break
                    if (break_loops):
                        break
                if (break_loops):
                    break
        
        # Reset the models' nonograms such that they are the same as nonogram_answer
        if n_models > 1:
            for model_index in range(n_models):
                nonograms[model_index] = nonogram_answer.copy()
            n_to_guess_tmp = n_to_guess
        
        # This enables the models to correct their mistakes
        if forgetting_enabled and n_to_guess > 0:
            for model_index,nonogram in enumerate(nonograms):
                for _ in range(to_forget):
                    row,col = rng.choice(filled_fields)
                    while nonogram[row,col] == UNKNOWN:
                        row,col = rng.choice(filled_fields)
                    nonogram[row,col] = UNKNOWN
                    if nonogram_answer[row,col] != UNKNOWN:
                        n_to_guess += 1
                        print(f"Forgotten {nonogram_answer[row,col]} at {row+1} {col+1} (model {model_index})")
                        nonogram_answer[row,col] = UNKNOWN
                        forgotten_fields.append((row,col))
            n_to_guess_tmp += to_forget
            n_to_guess_tmp = min(n_to_guess_tmp,size)

        print(f'({n_to_guess} more to fill)')

        # Print the current state of the nonogram
        for row,line in enumerate(nonogram_answer):
            for col,num in enumerate(line):
                if num == 1:
                    if (row,col) in colored_fields:
                        print('[green]1[/green]',end=' ')
                    else:
                        print('[red]1[/red]',end=' ')
                elif num == 0:
                    if (row,col) in colored_fields:
                        print('[green]0[/green]',end=' ')
                    else:
                        print('[blue]0[/blue]', end=' ')
                else:
                    if (row,col) in forgotten_fields:
                        print('[yellow]X[yellow]',end=' ')
                    else:
                        print('[grey78]X[/grey78]',end=' ')
            print()
        print()


    # Print temporary results (not confirmed and will possibly not appear in nonogram_answer)
    elif print_tmp_results and print_tmp_counter >= print_tmp_frequency and n_models > 1:
        print_tmp_counter = 0
        print('[yellow]==================[/yellow]')
        print("[yellow]TEMPORARY[/yellow]")
        print(f"Iterations until filling: {min(iterations_left_until_filling,n_to_guess_tmp)}")
        print(f"{n_to_guess_tmp} more to fill (temporary value)")
        print()
        for i,nonogram in enumerate(nonograms):
            print("MODEL " + str(i+1))
            for row,line in enumerate(nonogram):
                for col,num in enumerate(line):
                    if num == 1:
                        if num == nonogram_answer[row,col]:
                            print('[red]1[/red]',end=' ')
                        else:
                            print('[green]1[/green]',end=' ')
                    elif num == 0:
                        if num == nonogram_answer[row,col]:
                            print('[blue]0[/blue]', end=' ')
                        else:
                            print('[green]0[/green]',end=' ')
                    else:
                        print('[grey78]X[/grey78]',end=' ')
                print()
            print()
        print('[yellow]==================[/yellow]')

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