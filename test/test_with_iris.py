
import pytest

from src.model import run_k_means_clustering
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import homogeneity_score


@pytest.fixture()
def setup():
    dset = load_breast_cancer()
    X = dset.data
    y = dset.target
    param_grid = {'KMeans__n_clusters': range(2, 15, 1),
                  'KMeans__init': ['k-means++', 'random'],
                  'KMeans__n_init': [10],
                  'KMeans__max_iter': [300],
                  'KMeans__tol': [1e-4],
                  'KMeans__precompute_distances': ['auto'],
                  'KMeans__verbose': [0],
                  'KMeans__random_state': [42],
                  'KMeans__copy_x': [True],
                  'KMeans__n_jobs': [1],
                  'KMeans__algorithm': ['elkan'], }

    kmeans = run_k_means_clustering(X, param_grid)
    return y, kmeans, X


def test_that_the_result_for_breast_cancer_gives_two_clusters(setup):
    y, kmeans, X = setup
    assert kmeans.best_params_['KMeans__n_clusters'] == 2


def test_homogeneity_score_is_not_bad(setup):
    y, kmeans, X = setup
    assert homogeneity_score(y, kmeans.predict(X)) > 0.5
