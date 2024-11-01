import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for binary classification.

    Parameters:
    - y_true: array-like, true binary labels (0 or 1).
    - y_pred: array-like, predicted binary labels (0 or 1).

    Returns:
    - accuracy: float, the proportion of true results among the total number of cases examined.
    - precision: float, the ratio of true positive predictions to the total predicted positives.
    - recall: float, the ratio of true positive predictions to the total actual positives.
    - f1: float, the harmonic mean of precision and recall.
    """
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))  # True positives
    tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))  # True negatives
    fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))  # False positives
    fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))  # False negatives

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1


def remove_percentage(X, Y, keep_percentage):
    """
    Remove a certain percentage of the minority class while keeping the majority class intact.

    Parameters:
    - X: array-like, feature set (input data).
    - Y: array-like, target labels (binary, 0 or 1).
    - keep_percentage: float, the percentage of the minority class (0 to 1) to keep.

    Returns:
    - train_x_small: array-like, reduced feature set for training.
    - train_y_small: array-like, reduced target labels for training.
    """
    # Separate the dataset into two classes
    zero_rows = X[Y == 0]
    one_rows = X[Y == 1]
    
    # Determine how many rows of the minority class to keep
    num_keep = int(len(zero_rows) * keep_percentage)
    
    # Randomly select rows to keep from the minority class
    idx_to_keep = np.random.choice(zero_rows.shape[0], num_keep, replace=False)
    zero_rows_keep = zero_rows[idx_to_keep]
    
    # Combine the kept minority class rows with all of the majority class rows
    train_x_small = np.concatenate((zero_rows_keep, one_rows), axis=0)
    train_y_small = np.concatenate((np.zeros(num_keep), np.ones(one_rows.shape[0])))

    return train_x_small, train_y_small


def split_balanced(y, x, ratio, seed=1):
    """
    Split the dataset into training and validation sets in a balanced way.

    Parameters:
    - y: array-like, target labels (binary, 0 or 1).
    - x: array-like, feature set (input data).
    - ratio: float, the ratio of positive class samples to take for training.
    - seed: int, random seed for reproducibility.

    Returns:
    - y_train: array-like, target labels for training.
    - x_train: array-like, feature set for training.
    - y_val: array-like, target labels for validation.
    - x_val: array-like, feature set for validation.
    - indeces_train: array, indices of the training samples.
    """
    np.random.seed(seed)  # Set random seed for reproducibility
    
    # Separate the dataset into two classes
    y1 = y[y == 1]
    y0 = y[y == 0]
    x1 = x[y == 1]
    x0 = x[y == 0]
    
    # Shuffle the classes
    np.random.shuffle(y1)
    np.random.shuffle(y0)
    
    # Determine the number of samples to take from each class
    q1 = int(ratio * y1.shape[0])
    q0 = int(ratio * y0.shape[0])
    
    # Create training indices
    indeces_train = np.concatenate((np.where(y == 1)[0][:q1], np.where(y == 0)[0][:q0]))
    
    # Prepare training and validation sets
    x_train = np.concatenate((x1[:q1], x0[:q0]))
    y_train = np.concatenate((y1[:q1], y0[:q0]))
    x_val = np.concatenate((x1[q1:], x0[q0:]))
    y_val = np.concatenate((y1[q1:], y0[q0:]))
    
    return y_train, x_train, y_val, x_val, indeces_train


def balanced_k_folds(y, n_folds):
    """
    Create balanced k-fold indices for cross-validation.

    Parameters:
    - y: array-like, target labels (binary, 0 or 1).
    - n_folds: int, number of folds to create.

    Returns:
    - fold_indeces: list of arrays, each containing indices for the corresponding fold.
    """
    fold_indeces = []  # Initialize a list to hold the indices of each fold
    
    # Get indices for the positive and negative classes
    c1_indeces = np.where(y == 1)[0]
    np.random.shuffle(c1_indeces)  # Shuffle positive class indices
    c0_indeces = np.where(y == 0)[0]
    np.random.shuffle(c0_indeces)  # Shuffle negative class indices
    
    # Determine the number of samples per fold for the positive class
    q1 = c1_indeces.shape[0] // n_folds
    r1 = c1_indeces.shape[0] % n_folds  # Remainder for distributing extra samples
    
    # Create folds for the positive class
    for i in range(n_folds):
        fold_indeces.append(c1_indeces[i * q1 : (i + 1) * q1])
    for i in range(r1):
        fold_indeces[i] = np.append(fold_indeces[i], c1_indeces[n_folds * q1 + i])
    
    # Determine the number of samples per fold for the negative class
    q0 = c0_indeces.shape[0] // n_folds
    r0 = c0_indeces.shape[0] % n_folds  # Remainder for distributing extra samples
    
    # Create folds for the negative class
    for i in range(n_folds):
        fold_indeces[i] = np.concatenate((fold_indeces[i], c0_indeces[i * q0 : (i + 1) * q0]))
    for i in range(r0):
        fold_indeces[i] = np.append(fold_indeces[i], c0_indeces[n_folds * q0 + i])
    
    # Sanity checks
    assert y.shape[0] == sum(fold.shape[0] for fold in fold_indeces)  # Ensure all samples are included
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            # Ensure no overlapping samples between folds
            assert np.intersect1d(fold_indeces[i], fold_indeces[j]).shape[0] == 0
    
    return fold_indeces  # Return the list of fold indices


def accuracy_f1(y, tx, w, threshold):
    pred = sigmoid(tx @ w)
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0
    accuracy = np.mean(y == pred)
    precision = np.sum((y == 1) & (pred == 1)) / np.sum(pred)
    recall = np.sum((y == 1) & (pred == 1)) / np.sum(y)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, f1



def sigmoid(t):
    """
    Apply the sigmoid function on t.
    """
    t = np.clip(t, -20, 20)
    return 1 / (1 + np.exp(-t))
    # return np.e**t / (1 + np.e**t)