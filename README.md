# NonogramSolver

Alpha version (commit no. 16).

Please use the `main.py` file.
When it is run, the instructions for use will be shown.

### The algorithm
1. Train the model on generated data that are based on the current state of the nonogram.
2. Find the field the model is most certain about and fill it with a value.
3. Repeat

<pre>







</pre>

## Logistic regression and other relevant information
for `logistic_regression_test.py` [here](testing_scripts/logistic_regression_test.py)

### The algorithm
Based on the total number of filled fields and the number of spaces,
make a LogisticRegression model for each field and predict whether a field has a one or a zero.
Afterwards, repeat this procedure with the already guessed fields until the entire nonogram is filled.

When the model is highly uncertain about what to fill in next, `logistic_regression_test` terminates prematurely.


### Results
- Some nonograms (even 5x5 nonograms) are unsolvable with the current approach.
The hyperparameters that have achieved the best results so far for 5x5 nonograms are
  ````python
        num = 150
        train_test_split = 100
        default_decision_boundary = 0.98
        min_decision_boundary = 0.9
        decision_boundary_decrement = 0.01
        default_accuracy_decision_boundary = 0.99
        accuracy_decision_boundary_decrement = 0.005
        cv_min_num_decided = 2
  ````
  Also, StandardScaler is used instead of MinMaxScaler.

- Large nonograms (like 10x10) are never guessed entirely correctly.
The reason might be inappropriate hyperparameters. Or high dimensionality.

- Cross-validating the trained model does not seem to necessarily help.

- The number of training samples is crucial.
Make it too large and the models are slow and unsure.
Make it too small and the models are inaccurate.
Cf. double descent phenomenon.

- Interestingly, the chosen scaler (MinMaxScaler or StandardScaler)
and the feature_range parameter influence the results to a large degree.
For example, LogisticRegression refuses to converge with feature_range = (0,1).
Could this be a sign of more profound issues?
StandardScaler seems more successful so far.

- SVM is not appropriate for my purposes.
Since the whole nonogram cannot be guessed from the outset,
a predict_proba method is needed in order to fill only the fields a model can be most certain about.
Although SVM provides one such method, it is very slow and unreliable.
(Scikit-learn computes it through cross-validation as of 2024.)

- Random forests do not seem to perform well,
but more tweaking and testing is in order.




## Neural networks
Thanks to the complexity of neural networks and the simplicity of the Keras API,
I was allowed a bigger diversity of approaches.
(The Keras API is more convenient than scikit-learn API for multidimensional output.)

What I have tried so far:
- [Guessing one field per iteration](testing_scripts/nn_one_field.py):
  Always make the most auspicious guess and then retrain the model.
- [Guessing fields with high enough certainty](testing_scripts/nn_proba.py):
  Similar to the previous one, but faster (although possibly less accurate).
- [Simply training a model until accuracy is high enough](testing_scripts/nn_simple.py):
  Try to guess the entire nonogram in one go.

### Results:
More experimenting is needed.
The training process is more involved than I expected.
Any conclusions made here will probably be inaccurate.

- (Verification necessary) Number of layers is very important.
Just by adding one layer, the accuracy dropped by 6%
(measured during the 100th epoch of training.)
Could this be caused by vanishing gradients?
I thought that with 3 layers it would not be as prominent.

- Using NonogramFrame v2 with padding did work,
maybe even better than the first version,
which was surprising to me,
since the model has no conscious awareness of what the parameters mean
and padding should, I think,
be confusing to the first few parameters.
That is because the same number might appear in the first position
regardless of the amount of padding.

- (Now doubted) The probability output by the sigmoid function might not be reliable.

- Normalization()'s adapt has to have numpy array as data.
  Pandas will fail.

- `nn_one_field.py` is successful with small nonograms.

- `nn_simple.py` is so far unable to guess the entire nonogram correctly.

- Simply changing the data with PCA does not influence the results significantly.
However, no dimensionality reduction was involved, since the input dimensions would be changed.

- Convolutions are very important.

- Tried the functional API in the dev branch. It seems to be comparatively slow.

- Simply training the same model several times (as a form of ensemble learning) helps only marginally.
I might have to introduce some form of "forgetting" or the possibility of having different models.


_Addendum_:
  In commit no. 9 I wrote:

>  When a LogisticRegression model is unsure,
>  a more powerful model such as a neural network might be helpful.
>  But it would also be much slower.

and

> With a neural network, I could maybe use one model for the entire nonogram.

I must state that the neural networks are much faster than I expected them to be
(if they are not too complex, that is.)
I have to wait a similar amount of time on my computer as with the logistic regression.
  As for the second point,
that might be correct.


## Results not dependent on the model used:
- All the features I have so far are needed.
(See the next point about PCA and [this file](data_analysis/correlations.py).)

- PCA was not very successful.
The number of components that needs to be preserved for high variance is large.
That is, the function relating variance and the number of components increases slowly.
See [here](plots/pca_test_50x50.png) or [here](plots/pca_test_10x25.png).

- Normalisation damages the data.
Almost everywhere it is said that normalisation/standardisation helps the model converge.
But by removing normalisation I was able to achieve much better results.

- There is not a one-to-one correspondence between the task and the result.
Implication: my program might work even on bigger nonograms!



## Current objectives or ideas

**Common to all models**

- Using numba to speed up things. This might not be compatible with pandas.

- Try generating nonograms with a fixed number of spaces. Perhaps with another machine learning model.

- Test normalisation/standardisation.

- Test the algorithm with nonograms of various shapes.
A sane objective should be that the program works on shapes ~50x50.

- I might have to make default parameters and perhaps default 'degrees of carefulness' that set those parameters automatically.

- Try ensemble learning.
Boosting seems to be the most useful at the moment.

- Try some more dimensionality reduction.
(But fitting a manifold does not appear stable at first sight.)

- (!) Gather more information about the task at hand,
e.g. through visualization.

- Maybe I should try transforming other features as well?

- The NonogramFrame should be smaller if some fields were already guessed.
There is currently a massive overhead when part of the nonogram is already filled.
The same amount of data is generated as for an empty nonogram.



**Logistic regression**:
- I do not use the position as a feature,
since the relationship between the position and the target is not linear.
But what about a transformation of the feature? Like a quadratic one?

- The models might have to be cross-validated,
although I am not quite certain about this one.

- The number of training samples, maximal number of guesses per iteration and the decision boundary
(in this case the boundary, after which we insert a 0 or a 1) are hyperparameters.
Thus I should find their optimal values, probably depending on the nonogram's shape.


**Neural networks**:
- (!) Find the optimal parameters with the help of `keras_tuner.py`.

- (!) Try the program out on a real sample.

- Adding more layers and trying apply a correction for vanishing gradients.

- Using another optimizer.

- Convolutional NNs can help when dealing with large amount of data.
But this could be unnecessary on the small scale of the data I am currently working with.

- Using pretrained layers might speed up training.
I could perhaps make one large model for very large nonograms and use
the deep layers as starting point for every smaller shape.
But perhaps it would not be usable due to normalisation?

- Try models with "forgetting".




## Design decisions (for the LogisticRegression model)
Here I discuss some obstacles I encountered during my attempts at solving a nonogram.
This part's main purpose is to document why I made the program the way it is,
so it is unnecessary for you to read it.

It is quite possible that these issues can be bypassed in a better way.

### Why is there a model for every field?
It can be shown that the output for two fields can be different.
For example, with a 5x5 nonogram,
the model trained on the field (1,1) will be most confident with inserting 0s,
while a model trained on (3,3) will more likely insert 1s.
A model trained on every field would thus have parameters that befit an 'average' field.
Also, the parameters indicating the row and the column would obviously require,
due to the above said, a non-linear (quadratic?) relationship,
and I have no knowledge of the correct transformation that should be used.

### Why a general model cannot be shape-invariant.
Although the previous answer explains this at least in part,
there is another reason this approach was chosen.
A general model trained on a variety of nonograms differing in shape would need to know that a sum of fields in a row has different weight for each shape.
For example, a 5 for a 5x5 nonogram makes the answer obvious (XXXXX),
but for a 100x100 nonogram it is quite obviously not.

### Do I need so many features?
Unfortunately, it seems that I cannot make without a subset of the features completely.
When a RandomForestClassifier was trained on an empty nonogram,
then for every field all the features that do not describe the field's row or column
were almost equally important, but their influence was not negligible,
in my opinion.
The respective code is in the file `random_forest_feature_importance.py`
([here](data_analysis/random_forest_feature_importance.py)).
(See the scikit-learn documentation for the way in which the RandomForestClassifier computes feature importances.)
Also see [this plot](plots/feature_importance_10x7.png)
and [this plot](plots/feature_importance_5x5.png).
The RandomForestClassifier had the following parameters:
````python
    RandomForestClassifier(n_estimators=num//2, max_leaf_nodes=30, n_jobs=-1).
````
For the 5x5 nonogram I used 150 training samples, for the 10x7 nonogram 300.

The conclusion made here is also somewhat supported by the correlations between target and the column/row data (see [this file](data_analysis/correlations.py)).
Being farther does not necessarily mean being less important.
It varies between nonograms.




## Other notes and issues
It is advised (by Aurelien Geron) to always use the same training/testing set.
Should I do that?
Is his situation even applicable to this project?
His reasoning:

> Over time, you (or your machine learning algorithms) will get to see the whole dataset,
> which is what you want to avoid.

Why would I even want to avoid that?

I had an idea that instead of trying to make a model that guesses every field,
I could make a model that tries to find the most auspicious field.
It would only try to guess 1s (or 0s) and would be penalized only when the guessed field does not have a zero.
I think this could be implemented with a softmax function, but it would probably require some conditionals
and I have thus far been insuccessful in coding it in Keras (it might be impossible).