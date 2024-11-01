import numpy as np



def standardize(x):
    """
    Standardize the data set x.
    """
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    stds[stds == 0] = 1
    std_data = (x - means) / stds
    return std_data


def compute_mse_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: labels
        tx: features
        w: vector of model parameters

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # e = y - tx.dot(w)
    e = y - np.dot(tx, w)
    loss = np.sum(e**2) / (2 * len(y))

    return loss


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # e = y - tx.dot(w)
    # gradient = -tx.T.dot(e) / len(y)

    e = y - np.dot(tx, w)
    gradient = -np.dot(tx.T, e) / len(y)

    return gradient



def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def sigmoid(t):
    """
    Apply the sigmoid function on t.
    """
    t = np.clip(t, -20, 20)
    return 1 / (1 + np.exp(-t))
    # return np.e**t / (1 + np.e**t)



def loss_logistic_regression(y, tx, w):
    """
    Compute the loss for logistic regression ( Cross-Entropy Loss ).
    """
    loss = (
        -np.sum(
            y * np.log(sigmoid(np.dot(tx, w)))
            + (1 - y) * np.log(1 - sigmoid(np.dot(tx, w)))
        )
        / y.shape[0]
    )
    return loss


def calculate_gradient_logistic_regression(y, tx, w):
    """
    Compute the gradient for logistic regression.
    """
    return np.dot(tx.T, sigmoid(np.dot(tx, w)) - y) / y.shape[0]



def calculate_hessian(y, tx, w):
    """
    Calculate the Hessian matrix for the given linear regression model.
    """

    S = np.diagflat(sigmoid(np.dot(tx, w)) * (1 - sigmoid(np.dot(tx, w))))
    return np.dot(tx.T, np.dot(S, tx)) / y.shape[0]



def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent

    Args:
        y : labels
        tx : features
        initial_w : initial weights
        max_iters : maximum number of iterations
        gamma : learning rate

    Returns:
        weights, loss corresponding to the last iteration
    """

    w = initial_w
    for n_iter in range(max_iters):
        # compute and print loss
        loss = compute_mse_loss(y, tx, w)
        # print(f"At iteration {n_iter}, loss = {loss}")
        # compute the gradient
        gradient = compute_gradient(y, tx, w)
        # update w
        w = w - gamma * gradient

    # compute the loss
    loss = compute_mse_loss(y, tx, w)

    return w, loss


def mean_squared_error_gd_loss_tracking(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent

    Args:
        y : labels
        tx : features
        initial_w : initial weights
        max_iters : maximum number of iterations
        gamma : learning rate

    Returns:
        weights, loss corresponding to the last iteration
    """

    losses = np.zeros(max_iters)
    w = initial_w
    for n_iter in range(max_iters):
        # compute and print loss
        loss = compute_mse_loss(y, tx, w)
        # print(f"At iteration {n_iter}, loss = {loss}")
        # compute the gradient
        losses[n_iter] = loss
        gradient = compute_gradient(y, tx, w)
        # update w
        w = w - gamma * gradient

    # compute the loss
    loss = compute_mse_loss(y, tx, w)

    return w, losses


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent with batch size of 1

    :param y: labels
    :param tx: features
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: learning rate

    :return: weights, loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            # compute and print loss
            loss = compute_mse_loss(y, tx, w)
            print(f"At iteration {n_iter}, loss = {loss}")
            # compute the gradient
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            # update w
            w = w - gamma * gradient

    # compute the loss
    loss = compute_mse_loss(y, tx, w)

    return w, loss
    
    
def mean_squared_error_sgd_batch(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent with batch size of 1

    :param y: labels
    :param tx: features
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: learning rate

    :return: weights, loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1000):
            # compute and print loss
            loss = compute_mse_loss(y, tx, w)
            # print(f"At iteration {n_iter}, loss = {loss}")
            # compute the gradient
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            # update w
            w = w - gamma * gradient

    # compute the loss
    loss = compute_mse_loss(y, tx, w)

    return w, loss



def mean_squared_error_sgd_loss_tracking(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent with batch size of 1

    :param y: labels
    :param tx: features
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: learning rate

    :return: weights, loss
    """

    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            # compute and print loss
            loss = compute_mse_loss(y, tx, w)
            # print(f"At iteration {n_iter}, loss = {loss}")
            # compute the gradient
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            # update w
            losses.append(loss)
            w = w - gamma * gradient

    # compute the loss
    return w, losses
    
    
def mean_squared_error_sgd_loss_tracking_batch(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent with batch size of 1

    :param y: labels
    :param tx: features
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: learning rate

    :return: weights, loss
    """

    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1000):
            # compute and print loss
            loss = compute_mse_loss(y, tx, w)
            # print(f"At iteration {n_iter}, loss = {loss}")
            # compute the gradient
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            # update w
            losses.append(loss)
            w = w - gamma * gradient

    # compute the loss
    return w, losses


def least_squares(y, tx):
    """
    Least squares regression using normal equations

    :param y: labels
    :param tx: features

    :return: weights, loss
    """

    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_mse_loss(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations

    :param y: labels
    :param tx: features
    :param lambda_: regularization parameter

    :return: weights, loss
    """

    w = np.linalg.solve(
        np.dot(tx.T, tx) + 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1]),
        np.dot(tx.T, y),
    )
    loss = compute_mse_loss(y, tx, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent

    :param y: labels (0 or 1)
    :param tx: features
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: learning rate

    :return: weights, loss
    """
    w = initial_w
    if max_iters == 0:
        return w, loss_logistic_regression(y, tx, w)
    for n_iter in range(max_iters):
        gradient = calculate_gradient_logistic_regression(y, tx, w)
        w = w - gamma * gradient
        loss = loss_logistic_regression(y, tx, w)

    return w, loss


def logistic_regression_gd_loss_tracking(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent

    :param y: labels (0 or 1)
    :param tx: features
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: learning rate

    :return: weights, loss
    """
    losses = np.zeros(max_iters)
    w = initial_w
    for i in range(max_iters):
        loss = loss_logistic_regression(y, tx, w)
        gradient = calculate_gradient_logistic_regression(y, tx, w)
        # print(f"At iteration {i}, loss = {loss}")
        losses[i] = loss
        w = w - gamma * gradient

    return w, losses


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent

    :param y: labels (0 or 1)
    :param tx: features
    :param lambda_: regularization parameter
    :param initial_w: initial weights
    :param max_iters: maximum number of iterations
    :param gamma: learning rate

    :return: weights, loss
    """

    w = initial_w
    if max_iters == 0:
        return w, loss_logistic_regression(y, tx, w)
    for i in range(max_iters):
        
        gradient = calculate_gradient_logistic_regression(y, tx, w) + 2 * lambda_ * w
        # print(f"At iteration {i}, loss = {loss}")
        w = w - gamma * gradient
        loss = loss_logistic_regression(y, tx, w)

    return w, loss
