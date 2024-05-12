# coding:utf-8

from collections import Counter
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
from base.base_modified import BaseEstimatorModified
from collections import defaultdict
from sklearn.neighbors import LocalOutlierFactor

# class KNNBase(BaseEstimatorModified):
#     def __init__(self, k=5, distance_func = None):
#         """Base class for Nearest neighbors classifier and regressor.

#         Parameters
#         ----------
#         k : int, default 5
#             The number of neighbors to take into account. If 0, all the
#             training examples are used.
#         distance_func : function, default euclidean distance
#             A distance function taking two arguments. Any function from
#             scipy.spatial.distance will do.
#         """

#         self.k = None if k == 0 else k  # l[:None] returns the whole list
#         self.distance_func = distance_func
#         self.distance = [euclidean]

#     def aggregate(self, neighbors_targets):
#         raise NotImplementedError()

#     def _predict(self, X=None):
#         predictions = [self._predict_x(x) for x in X]
#         predictions = [Counter(prediction).most_common(1)[0][0] for prediction in predictions]
#         print(predictions)
#         return np.array(predictions)

#     def _predict_x(self, x):
#         """Predict the label of a single instance x."""
#         prediction = []
#         print("novo x")
#         if self.distance_func is None:
#             for k in self.distance:

#                 # compute distances between x and all examples in the training set.
#                 distances = (k(x, example) for example in self.X)

#                 # Sort all examples by their distance to x and keep their target value.
#                 neighbors = sorted(((dist, target,weigth) for (dist, target,weigth) in zip(distances, self.y, self.weigth)), key=lambda x: x[0])
#                 # print("Neighbors with distance:", neighbors[: self.k])
#                 # Get targets of the k-nn and aggregate them (most common one or
#                 # average).
#                 neighbors_targets = [(target,weigth) for (_, target, weigth) in neighbors[: self.k]]
#                 prediction.append(self.aggregate(neighbors_targets))
#         return prediction


# class KNNClassifier(KNNBase):
#     """Nearest neighbors classifier.

#     Note: if there is a tie for the most common label among the neighbors, then
#     the predicted label is arbitrary."""

#     def aggregate(self, neighbors_targets):
#         """Return the most common target label."""
#         print("Neighbors_target:", neighbors_targets)
        
#         # Inicialize um defaultdict com valor float
#         weighted_dict = defaultdict(float)
#         # weight_sum = 0
#         # Percorra os vizinhos e atualize o dicionário ponderado
#         for target, weight in neighbors_targets:
#             if weight != -1 :
#                 weighted_dict[target] += weight
#         #         weight_sum += 1
            
#         # for (target,_) in weighted_dict.items() :
#         #     weighted_dict[target] /= weight_sum
            
#         print(weighted_dict.items())
        
#         weighted_dict = dict(weighted_dict)
        
#         max_target = max(weighted_dict.items(), key=lambda item: item[1])

#         # Desempacote a chave e o valor máximo
#         max_target_key, max_target_value = max_target
#         # print("target:", max_target_key)
#         return max_target_key


# class KNNRegressor(KNNBase):
#     """Nearest neighbors regressor."""

#     def aggregate(self, neighbors_targets):
#         """Return the mean of all targets."""

#         return np.mean(neighbors_targets)


# coding:utf-8

class KNNBase(BaseEstimatorModified):
    def __init__(self, k=5, distance_func = None):
        """Base class for Nearest neighbors classifier and regressor.

        Parameters
        ----------
        k : int, default 5
            The number of neighbors to take into account. If 0, all the
            training examples are used.
        distance_func : function, default euclidean distance
            A distance function taking two arguments. Any function from
            scipy.spatial.distance will do.
        """

        self.k = None if k == 0 else k  # l[:None] returns the whole list
        self.distance_func = distance_func
        self.distance = [euclidean, cityblock, cosine]
        

    def aggregate(self, neighbors_targets):
        raise NotImplementedError()

    def _predict(self, X=None):
        lof = LocalOutlierFactor(n_neighbors=20, algorithm='kd_tree')
        train_weigth = lof.fit_predict(X)
        # print(train_weigth)
        
        predictions = [self._predict_x(x,weigth) for x, weigth in zip(X,train_weigth)]
        predictions = [Counter(prediction).most_common(1)[0][0] for prediction in predictions]
        # print(predictions)
        return np.array(predictions)

    def _predict_x(self, x, train_weigth):
        """Predict the label of a single instance x."""
        prediction = []
        if self.distance_func is None:
            for k in self.distance:

                # compute distances between x and all examples in the training set.
                distances = (k(x, example) for example in self.X)

                # Sort all examples by their distance to x and keep their target value.
                neighbors = sorted(((dist, target,weigth) for (dist, target,weigth) in zip(distances, self.y, self.weigth)), key=lambda x: x[0])
                # print("Neighbors with distance:", neighbors[: self.k])
                # Get targets of the k-nn and aggregate them (most common one or
                # average).
                neighbors_targets = [(target,weigth) for (_, target, weigth) in neighbors[: self.k]]
                prediction.append(self.aggregate(neighbors_targets,train_weigth))
        return prediction


class KNNClassifier(KNNBase):
    """Nearest neighbors classifier.

    Note: if there is a tie for the most common label among the neighbors, then
    the predicted label is arbitrary."""

    def aggregate(self, neighbors_targets,train_weigth):
        """Return the most common target label."""
        # print("Neighbors_target:", neighbors_targets)
        
        # Inicialize um defaultdict com valor float
        weighted_dict = defaultdict(float)
        # weight_sum = 0
        # Percorra os vizinhos e atualize o dicionário ponderado
        for target, weight in neighbors_targets:
            weighted_dict[target] += weight
            # weight_sum += weight
        
        # print(weighted_dict.items())
        # for (target,_) in weighted_dict.items() :
        #     weighted_dict[target] /= weight_sum
            
        weighted_dict = dict(weighted_dict)

        # 
# if train_weigth == -1:
        #     max_target = min(weighted_dict.items(), key=lambda item: item[1])
        # else:
        max_target = max(weighted_dict.items(), key=lambda item: item[1])
        # Desempacote a chave e o valor máximo
        max_target_key, max_target_value = max_target
        # print("target:", max_target_key)
        return max_target_key


class KNNRegressor(KNNBase):
    """Nearest neighbors regressor."""

    def aggregate(self, neighbors_targets):
        """Return the mean of all targets."""

        return np.mean(neighbors_targets)
