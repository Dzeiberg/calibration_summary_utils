from tqdm import tqdm
from assay_calibration.fit_utils.fit import Fit
from typing import List, Tuple
from joblib import Parallel, delayed
import numpy as np


def get_priors(scoreset, fits: List[Fit], **kwargs):
    pathogenic_idx = kwargs.get("pathogenic_idx", 0)
    benign_idx = kwargs.get("benign_idx", 1)
    population_idx = kwargs.get("population_idx", 2)
    population_scores = scoreset.scores[
        scoreset.sample_assignments[:, population_idx] == 1
    ]

    priors = [
        fit.get_prior_estimate(
            population_scores, pathogenic_idx=pathogenic_idx, benign_idx=benign_idx
        )
        for fit in tqdm(fits, desc="Estimating priors")
    ]
    return priors


def get_thresholds(fits, medianPrior, inverted, point_values=[1, 2, 3, 4, 8], **kwargs):
    """
    Get the thresholds for each model

    Arguments
    ----------
    fits : list[Fit]
        List of Fit objects for the dataset
    medianPrior : float
       Prior estimates for the dataset

    Optional Arguments
    ----------
    point_values : list[float]
        List of point values to use for the thresholds (default [1,2,3,4,8])

    Returns
    ----------
    pathogenic_thresholds : np.ndarray
        Array of shape (n_models, n_point_values) containing the pathogenic score thresholds for each model
    benign_thresholds : np.ndarray
        Array of shape (n_models, n_point_values) containing the benign score thresholds for each model

    """
    n_models = len(fits)
    n_point_values = len(point_values)
    pathogenic_thresholds = np.zeros((n_models, n_point_values))
    benign_thresholds = np.zeros((n_models, n_point_values))

    def compute_thresholds(fit, prior, point_values, inverted):
        return fit.get_score_thresholds(prior, point_values, inverted)

    print("Computing thresholds")
    n_jobs = kwargs.get("n_jobs", -1)
    if n_jobs != 1:
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(compute_thresholds)(fit, medianPrior, point_values, inverted)
            for i, fit in enumerate(fits)
        )
    else:
        results = []
        for i, fit in enumerate(tqdm(fits, desc="Computing thresholds")):
            results.append(compute_thresholds(fit, medianPrior, point_values, inverted))

    for i, (pathogenic, benign) in enumerate(results):  # type: ignore
        pathogenic_thresholds[i], benign_thresholds[i] = pathogenic, benign
    return pathogenic_thresholds, benign_thresholds


def summarize_thresholds(
    pathogenic_thresholds: np.ndarray,
    benign_thresholds: np.ndarray,
    final_quantile: float,
    inverted: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the summary of the thresholds computed for each model

    Arguments
    ----------
    pathogenic_thresholds : np.ndarray
        Array of shape (n_models, n_point_values) containing the pathogenic score thresholds for each model
    benign_thresholds : np.ndarray
        Array of shape (n_models, n_point_values) containing the benign score thresholds for each model
    final_quantile : float
        Quantile to use for the final threshold [0,1]
    inverted : bool
        Whether the score set is 'flipped' from its canonical orientation

    Returns
    ----------
    final_pathogenic_thresholds : np.ndarray
        Array of shape (n_point_values,) containing the final pathogenic score thresholds
    final_benign_thresholds : np.ndarray
        Array of shape (n_point_values,) containing the final benign score thresholds
    """
    if inverted:
        QP = 1 - final_quantile
        QB = final_quantile
        pathogenic_thresholds[np.isinf(pathogenic_thresholds)] *= -1
        benign_thresholds[np.isinf(benign_thresholds)] *= -1
    else:
        QP = final_quantile
        QB = 1 - final_quantile
    P = np.quantile(pathogenic_thresholds, QP, axis=0)
    B = np.quantile(benign_thresholds, QB, axis=0)
    return P, B
