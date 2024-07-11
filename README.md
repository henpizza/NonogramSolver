# NonogramSolver

The first somewhat working example.
Is able to find the correct label for one field in a fraction of cases (typically around 1/20-1/10) for a 5x5 nonogram.



## The algorithm
Step 1: Based on the total number of filled fields and the number of spaces, make a LogisticRegression model for each field and predict, whether a field has a one or a zero.

Currently only the testing phase is implemented.

### Results
For a 5x5 nonogram a model can for one field and in about 5% predict with almost 100% accuracy the correct value. Only values where the likelihood is greater than 95% were predicted. The models were trained on 100 samples.



## Current objectives

- Try to think out more features.

- Try generating a nonogram with a fixed number of spaces. Perhaps with another machine learning model.

- Try normalisation/standardisation.

- Test the algorithm with nonograms of various shapes.

- The number of training samples and the decision boundary (in this case the boundary, after which we insert a 0 or a 1) are hyperparameters. Thus I should find their optimal values, probably depending on the nonogram's shape.

- Make a model for step 2.

- (Possibly) wrap all models from step 1 into one model. Would be more memory-demanding, but perhaps easier to use.




## Design decisions
Here I discuss some obstacles I encountered during my attempts at solving a nonogram. This part's main purpose is to document why I made the program the way it is, so it is unnecessary for you to read it.

It is quite possible that these issues can be bypassed in a better way.

### Why is there a model for every field?
It can be shown that the output for two fields can be different. For example, with a 5x5 nonogram, the model trained on the field (1,1) will be most confident with inserting 0s, while a model trained on (3,3) will more likely insert 1s. A model trained on every field would thus have parameters that befit an 'average' field. Also, the parameters indicating the row and the column would obviously require, due to the above said, a non-linear (quadratic?) relationship, thus the logistic regression could not be used directly.

### Why a general model cannot be shape-invariant.
Although the previous answer explains this at least in part, there is another reason this approach was chosen. A general model trained on a variety of nonograms differing in shape would need to know that a sum of fields in a row has different weight for each shape. For example, a 5 for a 5x5 nonogram makes the answer obvious (XXXXX), but for a 100x100 nonogram it is quite obviously not.

## Other notes
Interestingly, the chosen scaler (MinMaxScaler or StandardScaler) and the feature_range parameter influence the results to a large degree. For example, LogisticRegression refuses to converge with feature_range = (0,1).
