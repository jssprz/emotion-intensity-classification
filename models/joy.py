import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class JoyTweetClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, demo_param='demo'):
    self.demo_param = demo_param
    self.classifier = MultinomialNB()

  def fit(self, X, y):
    # Check that X and y have correct shape
    X, y = check_X_y(X, y, accept_sparse=True)
    # Store the classes seen during fit
    self.classes_ = unique_labels(y)

    self.X_ = X
    self.y_ = y

    self.classifier.fit(X, y)

    # Return the classifier
    return self

  def predict_proba(self, X):
    # Check is fit had been called
    check_is_fitted(self)
    # Input validation
    X = check_array(X, accept_sparse=True)

    # return np.random.rand(X.shape[0],3)
    return self.classifier.predict_proba(X)

  def predict(self, X):
    # Check is fit had been called
    check_is_fitted(self)

    # Input validation
    X = check_array(X)

    # closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
    # return self.y_[closest]
    return self.classifier.predict(X)