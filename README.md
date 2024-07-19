# NonogramSolver

Pre-alpha version.



## The algorithm (file `logistic_regression_test.py` ([here](testing_scripts/logistic_regression_test.py)))
Based on the total number of filled fields and the number of spaces, make a LogisticRegression model for each field and predict whether a field has a one or a zero.
Afterwards, repeat this procedure with the already guessed fields until the entire nonogram is filled.

When the model is highly uncertain about what to fill in next, `logistic_regression_test` terminates prematurely.

Currently only the testing phase is implemented.

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

- All the features I have so far are needed.

- Random forests do not seem to perform well,
but more tweaking and testing is in order.

- PCA was not very successful.
The number of components that needs to be preserved for high variance is large.
That is, the function relating variance and the number of components increases slowly.
See [here](plots/pca_test_50x50.png) or [here](plots/pca_test_10x25.png).







## Current objectives

- Try to think out more features.

- Try generating nonograms with a fixed number of spaces. Perhaps with another machine learning model.

- Test normalisation/standardisation.

- Test the algorithm with nonograms of various shapes.
A sane objective should be that the nonogram works on shapes ~50x50.

- The number of training samples, maximal number of guesses per iteration and the decision boundary (in this case the boundary, after which we insert a 0 or a 1) are hyperparameters.
Thus I should find their optimal values, probably depending on the nonogram's shape.

- Make a different model for later steps.

- (Possibly) wrap all models from step 1 into one model. Would be more memory-demanding, but perhaps easier to use.

- I might have to make default parameters and perhaps default 'degrees of carefulness' that set those parameters automatically.

- The models might have to be cross-validated,
although I am not quite certain about this one.

- (!) Try neural networks.

  Formerly it was planned to be used with logistic regression:
  When a LogisticRegression model is unsure,
  a more powerful model such as a neural network might be helpful.
  But it would also be much slower.

  Now it appears likely that using a neural network as the primary model might be preferable.

- Try ensemble learning.
Boosting seems to be the most useful at the moment.

- Try some more dimensionality reduction.

- (!) Gather more information about the task at hand,
e.g. through visualization.

- I do not use the position as a feature,
since the relationship between the position and the target is not linear.
But what about a transformation of the feature? Like a quadratic one?

- Maybe I should try transforming other features as well?


_Current main objective:_
Try a neural network that depends on the position.
I do not know momentarily how to make logistic regression work.
With a neural network, I could maybe use one model for the entire nonogram.





## Design decisions
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