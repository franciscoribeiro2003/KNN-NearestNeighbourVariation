
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

# class BaseEstimatorModified:
#     y_required = True
#     fit_required = True

#     def _setup_input(self, X, y=None):
#         """Ensure inputs to an estimator are in the expected format.

#         Ensures X and y are stored as numpy ndarrays by converting from an
#         array-like object if necessary. Enables estimators to define whether
#         they require a set of y target values or not with y_required, e.g.
#         kmeans clustering requires no target labels and is fit against only X.

#         Parameters
#         ----------
#         X : array-like
#             Feature dataset.
#         y : array-like
#             Target values. By default is required, but if y_required = false
#             then may be omitted.
#         """
#         if not isinstance(X, np.ndarray):
#             X = np.array(X)

#         if X.size == 0:
#             raise ValueError("Got an empty matrix.")

#         if X.ndim == 1:
#             self.n_samples, self.n_features = 1, X.shape
#         else:
#             self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

#         lof = LocalOutlierFactor(n_neighbors=20, algorithm='kd_tree')
        
#         lof_scores_train = lof.negative_outlier_factor_
#         lof_scores_train = (lof_scores_train - min(lof_scores_train)) / (max(lof_scores_train) - min(lof_scores_train))
#         self.weigth = lof.fit_predict(X)
#         # print(self.weigth)

#         self.X = X

#         if self.y_required:
#             if y is None:
#                 raise ValueError("Missed required argument y")

#             if not isinstance(y, np.ndarray):
#                 y = np.array(y)

#             if y.size == 0:
#                 raise ValueError("The targets array must be no-empty.")

#         self.y = y

#     def fit(self, X, y=None):
#         self._setup_input(X, y)

#     def predict(self, X=None):
#         #  print("-----")
#         if not isinstance(X, np.ndarray):
#             X = np.array(X)

#         if self.X is not None or not self.fit_required:
#             return self._predict(X)
#         else:
#             raise ValueError("You must call `fit` before `predict`")

#     def _predict(self, X=None):
#         raise NotImplementedError()
    

class BaseEstimatorModified:
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):
        """Ensure inputs to an estimator are in the expected format.

        Ensures X and y are stored as numpy ndarrays by converting from an
        array-like object if necessary. Enables estimators to define whether
        they require a set of y target values or not with y_required, e.g.
        kmeans clustering requires no target labels and is fit against only X.

        Parameters
        ----------
        X : array-like
            Feature dataset.
        y : array-like
            Target values. By default is required, but if y_required = false
            then may be omitted.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError("Got an empty matrix.")

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        lof = LocalOutlierFactor(n_neighbors=20)
        lof.fit_predict(X) # -> [1,-1]
        lof_scores_train = lof.negative_outlier_factor_ # -> [qunado ]
        lof_scores_train = (lof_scores_train - min(lof_scores_train)) / (max(lof_scores_train) - min(lof_scores_train))
        self.weigth = lof_scores_train
        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError("Missed required argument y")

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError("The targets array must be no-empty.")

        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        #  print("-----")
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("You must call `fit` before `predict`")

    def _predict(self, X=None):
        raise NotImplementedError()
    