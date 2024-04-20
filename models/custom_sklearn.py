"""Adam's substitute for sklearn"""
import numpy as np


def train_test_split(*arrays, test_size=0.25, random_state=None):
    """
    Splits the data into train and test sets.

    Parameters:
        *arrays (list of numpy.ndarray): Datasets to be split (e.g., X and y).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for the random number generator for reproducibility.
    
    Returns:
        List: Split datasets. For each input array, two split arrays are returned (train and test).
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Check that all arrays have the same length
    if len(set(len(array) for array in arrays)) != 1:
        raise ValueError("All input arrays must have the same number of elements.")

    # Calculate the number of training instances
    n_samples = len(arrays[0])
    n_train = int((1 - test_size) * n_samples)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Split each array and store the result
    result = []
    for array in arrays:
        train = array[indices[:n_train]]
        test = array[indices[n_train:]]
        result.extend([train, test])

    return result


class StandardScaler:
    """
    Standardize features by optionally removing the mean and scaling to unit variance.

    Parameters:
        with_mean (bool): If True, center the data before scaling. Default is True.
    """
    def __init__(self, with_mean=True):
        self.with_mean = with_mean
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Calculate mean (if with_mean is True) and standard deviation to be used for later scaling.

        Parameters:
            X (numpy.ndarray): The data used to compute the mean and standard deviation.
        """
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])  # Use zero array to neutralize the mean subtraction in transform
        self.std_ = np.std(X, axis=0)

    def transform(self, X):
        """
        Perform standardization by optionally centering and always scaling.

        Parameters:
            X (numpy.ndarray): The data to be transformed.

        Returns:
            X_transformed (numpy.ndarray): The transformed data.
        """
        if self.with_mean:
            X_transformed = (X - self.mean_) / self.std_
        else:
            X_transformed = X / self.std_  # No centering if with_mean is False
        return X_transformed

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Parameters:
            X (numpy.ndarray): The data to fit and transform.

        Returns:
            X_transformed (numpy.ndarray): The transformed data.
        """
        self.fit(X)
        return self.transform(X)
