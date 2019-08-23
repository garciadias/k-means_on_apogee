"""Run K-means in a sample of synthetic spectra."""
from os import getcwd

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import GridSearchCV
import pandas as pd


def cv_calinski_harabasz_scorer(estimator, X):
    estimator.fit(X)
    cluster_labels = estimator.steps[1][1].labels_
    return calinski_harabasz_score(X, cluster_labels)


def run_k_means_clustering(X, param_grid, n_iter=10):
    """Run K-Means clustering in a random search of model hyperparameters.

    Read more in the https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.

    Parameters
    ----------
    X : array-like, shape = [n_spectra, n_pixels]
        Training vector, where n_samples is the number of samples and n_features is the number of features.
        Here you will insert the data of the synthetic spectra.
    param_grid : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.
    Return
    ------
    grid_search: sklearn model
        Returns the RandomizedSearchCV model trained. With this you can see your clusters and apply the algorith to
        unseen data.
    """
    pipeline = Pipeline(steps=[('scaler', RobustScaler()), ('KMeans', KMeans())])
    grid_search = GridSearchCV(pipeline,
                               param_grid=param_grid,
                               refit=True,
                               n_jobs=-1,
                               cv=2,
                               iid=False,
                               scoring=cv_calinski_harabasz_scorer,
                               verbose=0)
    grid_search.fit(X)
    return grid_search


if __name__ == '__main__':
    PROJECT_PATH = getcwd()
    spectra = pd.read_csv('%s/data/all_spectra.csv' % getcwd(), index_col=0)
    param_grid = {'KMeans__n_clusters': range(2, 10, 1),
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
    k_means = run_k_means_clustering(spectra.values, param_grid, n_iter=10)
