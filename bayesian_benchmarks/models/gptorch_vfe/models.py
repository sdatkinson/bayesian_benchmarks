"""
To add a model, create a new directory in bayesian_benchmarks.models (here) and make file models.py containing at
least one of the classes below.

Model usage is similar to sklearn. For regression:

model = RegressionModel(is_test=False)
model.fit(X, Y)
mean, var = model.predict(X_test)  # first two moments of the posterior
samples = model.sample(X_test, S)  # samples from the posterior

For classification:
model = ClassificationModel(K, is_test=False)  # K is the number of classes
model.fit(X, Y)
p = model.predict(X_test)          # predictive probabilities for each class (i.e. onehot)

It should be feasible to call fit and predict many times (e.g. avoid rebuilding a tensorflow graph on each call).

"""

import numpy as np
import torch

from gptorch.models.sparse_gpr import VFE
from gptorch import kernels, likelihoods

torch.set_default_dtype(torch.double)


class RegressionModel(object):
    def __init__(self, is_test=False, seed=0):
        """
        If is_test is True your model should train and predict in a few seconds (i.e. suitable for travis)
        """
        self._model = None

    def fit(self, X : np.ndarray, Y : np.ndarray):
        """
        Train the model (and probably create the model, too, since there is no shape information on the __init__)

        :param X: numpy array, of shape N, Dx
        :param Y: numpy array, of shape N, Dy
        :return:
        """
        kernel, likelihood = RegressionModel._init_kernel_and_likelihood(X, Y)
        self._model = VFE(X, Y, kernel, likelihood=likelihood)
        try:
            self._model.optimize(method="L-BFGS-B", max_iter=1000)
        except:
            self._model.optimize(max_iter=2000, learning_rate=0.01)

    def predict(self, Xs : np.ndarray):
        """
        The predictive mean and variance

        :param Xs: numpy array, of shape N, Dx
        :return: mean, var, both of shape N, Dy
        """
        with torch.no_grad():
            mu, s = self._model.predict_y(Xs)
            return mu.data.numpy(), s[:, None].expand_as(mu).data.numpy()

    def sample(self, Xs : np.ndarray, S : int):
        """
        Samples from the posterior
        :param Xs: numpy array, of shape N, Dx
        :param S: number of samples
        :return: numpy array, of shape (S, N, Dy)
        """
        with torch.no_grad():
            return self._model.predict_y_samples(Xs, n=S).dadta.numpy()

    def _init_kernel_and_likelihood(x, y):
        kernel = kernels.Rbf(x.shape[1], ARD=True)
        kernel.length_scales.data = torch.Tensor(np.log(x.max(axis=0) - x.min(axis=0)))
        kernel.variance.data = torch.Tensor([np.log(y.var())])
        likelihood = likelihoods.Gaussian(variance=0.001 * y.var())

        return kernel, likelihood


# class ClassificationModel:
#     def __init__(self, K, is_test=False, seed=0):
#         """
#         :param K: number of classes
#         :param is_test: whether to run quickly for testing purposes
#         """

#     def fit(self, X : np.ndarray, Y : np.ndarray):
#         """
#         Train the model (and probably create the model, too, since there is no shape information on the __init__)

#         Note Y is not onehot, but is an int array of labels in {0, 1, ..., K-1}

#         :param X: numpy array, of shape N, Dx
#         :param Y: numpy array, of shape N, 1
#         :return:
#         """
#         pass

#     def predict(self, Xs : np.ndarray):
#         """
#         The predictive probabilities

#         :param Xs: numpy array, of shape N, Dx
#         :return: p, of shape (N, K)
#         """
#         raise NotImplementedError
