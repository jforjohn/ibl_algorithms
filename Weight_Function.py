from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier



def feature_select(X_train, y_train, X_test, method= "model-based"):
    """

    :param X_train: train set to transform
    :param X_test: test set to transform
    :param y_train: train labels
    :param method: methods allowed "univariate" with chi2 and 50% or "model-based" using random forest and median
    :return: filtered data set and filtered test set
    """

    # Select the correct model and create the "select" object filter
    if method == "univariate":
        select = SelectPercentile(chi2, percentile = 50)
    elif method == "model-based":
        select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=101), threshold='median')

    # Fit the model using train set
    select.fit(X_train, y_train)

    # Filter the X_train and X_test data set and save them
    X_train_selection = select.transform(X_train)
    X_test_selection = select.transform(X_test)


    return X_train_selection, X_test_selection



"""
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

"""