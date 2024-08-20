from sklearn.preprocessing import LabelEncoder
from collections import namedtuple
# import tensorflow as tf
import numpy as np
def encode_labels(data, columns):
    for column in columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    return data


def train_test_split(data, test_size, y_name):
    train_test = namedtuple('train_test', ['x_train', 'x_test', 'y_train', 'y_test'])
    split_row = len(data) - int(test_size * len(data))
    train_data = data.iloc[:split_row]
    test_data = data.iloc[split_row:]
    return train_test(x_train=train_data.drop(y_name, axis=1).to_numpy(), x_test=test_data.drop(y_name, axis=1).to_numpy(), y_train=train_data[y_name].to_numpy(), y_test=test_data[y_name].to_numpy())

def train_test_split_numpy (data, test_size, y_index=-1):
    train_test = namedtuple('train_test', ['x_train', 'x_test', 'y_train', 'y_test'])
    split_row = int(len(data) * (1 - test_size))
    x_train = np.array(data[:split_row, :y_index].tolist())
    x_test = np.array(data[split_row:, :y_index].tolist())
    y_train = np.array(data[:split_row, y_index].tolist())
    y_test = np.array(data[split_row:, y_index].tolist())
    return train_test(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def transform_to_returns(array: np.ndarray) -> np.ndarray:
    """Transforms an array of prices to returns.

    Args:
    - array (np.ndarray): Array of prices.

    Returns:
    - Array of returns.
    """
    return (array[1:] - array[:-1]) / array[:-1]

def drop_nulls(arr):
    return arr[~np.isnan(arr)]

def count_nulls(arr):
    return np.sum(np.isnan(arr))
