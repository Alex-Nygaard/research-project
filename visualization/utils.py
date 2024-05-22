from sklearn.metrics import mean_absolute_error, mean_squared_error
from fastdtw import fastdtw
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr


def calculate_mse(sequence1, sequence2):
    """
    Calculate the Mean Squared Error (MSE) between two sequences.

    Parameters:
    sequence1 (list or array): First sequence of data points.
    sequence2 (list or array): Second sequence of data points.

    Returns:
    float: MSE between the two sequences.
    """
    mse = mean_squared_error(sequence1, sequence2)
    return mse


def calculate_mae(sequence1, sequence2):
    """
    Calculate the Mean Absolute Error (MAE) between two sequences.

    Parameters:
    sequence1 (list or array): First sequence of data points.
    sequence2 (list or array): Second sequence of data points.

    Returns:
    float: MAE between the two sequences.
    """
    mae = mean_absolute_error(sequence1, sequence2)
    return mae


def calculate_dtw(sequence1, sequence2):
    """
    Calculate the Dynamic Time Warping (DTW) distance between two sequences.

    Parameters:
    sequence1 (list or array): First sequence of data points.
    sequence2 (list or array): Second sequence of data points.

    Returns:
    float: DTW distance between the two sequences.
    """
    distance, path = fastdtw(
        np.array(sequence1, dtype="float"),
        np.array(sequence2, dtype="float"),
        dist=2,
    )
    return distance


def calculate_pearson_correlation(sequence1, sequence2):
    """
    Calculate the Pearson Correlation Coefficient between two sequences.

    Parameters:
    sequence1 (list or array): First sequence of data points.
    sequence2 (list or array): Second sequence of data points.

    Returns:
    float: Pearson Correlation Coefficient between the two sequences.
    """
    correlation, _ = pearsonr(sequence1, sequence2)
    return correlation
