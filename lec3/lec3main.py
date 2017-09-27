from sklearn import tree
import numpy as np
import pandas
from sklearn.metrics import f1_score
from time import time
from sklearn.model_selection import GridSearchCV
from operator import itemgetter
import graphviz

data = pandas.read_csv("d:\\Downloads\\bike\\day.csv")

for column in data:
    if column != "cnt":
        if data[column].dtype == np.int64 or data[column].dtype == np.float64:
            if np.corrcoef(data[column], data["cnt"]).min() > 0.9:
                del data[column]
        else:
            del data[column]

train = data.head(600)
valid = data.tail(data.size - 600)

clf = tree.DecisionTreeRegressor()
clf.fit(train.ix[:, train.columns != "cnt"], train["cnt"])

pred = clf.predict(valid.ix[:, valid.columns != "cnt"])

print(f1_score(valid["cnt"], pred, average='micro'))

# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.view()

# set of parameters to test
param_grid = {"criterion": ["mse", "friedman_mse", "mae"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 5, 10, 20],
              "min_samples_leaf": [1, 2, 5, 10, 25, 50, 75, 100],
              "max_leaf_nodes": [None, 20],
              }


def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters


def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    start = time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
            len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return top_params


top_par = run_gridsearch(train.ix[:, train.columns != "cnt"], train["cnt"], clf, param_grid, 10)

clf2 = tree.DecisionTreeRegressor(**top_par)
clf2.fit(train.ix[:, train.columns != "cnt"], train["cnt"])

pred2 = clf2.predict(valid.ix[:, valid.columns != "cnt"])

print(f1_score(valid["cnt"], pred2, average='micro'))