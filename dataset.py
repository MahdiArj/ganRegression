import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def gen_data_linear(n_instance):
    a = np.random.normal(3, 3, n_instance)
    samples = int(n_instance / 2)
    X = np.hstack((np.random.normal(4, 3, samples), np.random.normal(4, 3, samples)))
    y = np.hstack((X[:samples] + a[:samples], X[samples:] + a[samples:]))
    X = X.reshape((n_instance, 1))
    y = y.reshape((n_instance, 1))

    return X, y


def gen_data_heteroscedastic(n_instance):
    X = np.random.normal(0, 1, n_instance)
    b = (0.001 + 0.5 * np.abs(X)) * np.random.normal(1, 1, n_instance)
    y = X + b
    X = X.reshape((n_instance, 1))
    y = y.reshape((n_instance, 1))

    return X, y


def gen_data_multimodal(n_instance):
    x = np.random.rand(int(n_instance / 2), 1)
    y1 = np.ones((int(n_instance / 2), 1))
    y2 = np.ones((int(n_instance / 2), 1))
    y1[x < 0.4] = 1.2 * x[x < 0.4] + 0.2 + 0.03 * np.random.randn(np.sum(x < 0.4))
    y2[x < 0.4] = x[x < 0.4] + 0.6 + 0.03 * np.random.randn(np.sum(x < 0.4))
    y1[np.logical_and(x >= 0.4, x < 0.6)] = 0.5 * x[np.logical_and(x >= 0.4, x < 0.6)] + 0.01 * np.random.randn(
        np.sum(np.logical_and(x >= 0.4, x < 0.6)))
    y2[np.logical_and(x >= 0.4, x < 0.6)] = 0.6 * x[np.logical_and(x >= 0.4, x < 0.6)] + 0.01 * np.random.randn(
        np.sum(np.logical_and(x >= 0.4, x < 0.6)))
    y1[x >= 0.6] = 0.5 + 0.02 * np.random.randn(np.sum(x >= 0.6))
    y2[x >= 0.6] = 0.5 + 0.02 * np.random.randn(np.sum(x >= 0.6))
    y = np.array(np.vstack([y1, y2])[:, 0]).reshape((n_instance, 1))
    x = np.tile(x, (2, 1)) + 0.02 * np.random.randn(n_instance, 1)
    x = np.array(x[:, 0]).reshape((n_instance, 1))

    return x, y


def gen_data_exp(n_instance):
    z = np.random.normal(0, 1, n_instance)
    X = np.random.normal(0, 1, n_instance)
    y = X + np.exp(z)
    X = X.reshape((n_instance, 1))
    y = y.reshape((n_instance, 1))

    return X, y


def get_dataset(n_instance=1000, scenario="linear", seed=1):
    """
    Create regression data: y = x(1 + f(z)) + g(z)
    """
        
    if scenario == "UAM":
        dataset = pd.read_excel("/content/gdrive/My Drive/Colab Notebooks/cGAN/data.xlsx")
        dataset = shuffle(dataset)
        X_train_full, y_train_full = np.array(dataset.iloc[:900 ,:6]), np.array(dataset.iloc[:900 ,6])
        X_test, y_test = np.array(dataset.iloc[900: ,:6]), np.array(dataset.iloc[900: ,6])
        
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=seed)

    else:
        raise NotImplementedError("Dataset does not exist")

    return X_train, y_train, X_test, y_test, X_valid, y_valid
