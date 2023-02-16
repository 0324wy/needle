import struct
import gzip
import numpy as np

import sys

sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    ### BEGIN YOUR SOLUTION
    with gzip.open(label_filename, 'rb') as f:
        magic, nums = struct.unpack('>2I', f.read(8))
        label = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(image_filename, 'rb') as f:
        magic, nums, rows, cols = struct.unpack('>4I', f.read(16))
        image = np.frombuffer(f.read(), dtype=np.uint8).reshape(nums, 28 * 28)
        image = image.astype(np.float32) / 255.0

    return image, label
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    z_exp = ndl.exp(Z)
    z_exp_sum = ndl.summation(z_exp, axes=1)
    z_log = ndl.log(z_exp_sum)
    z_sum = ndl.summation(Z * y_one_hot, axes=1)

    return ndl.summation(z_log - z_sum) / y_one_hot.shape[0]


def one_hot(y_batch, nums_classes):
    return np.eye(nums_classes)[y_batch]


def get_batch_nums(batch, X):
    b = np.size(X, axis=0) // batch
    if np.size(X, axis=0) % batch == 0:
        batch_nums = b
    else:
        batch_nums = b + 1
    return batch_nums


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    # BEGIN YOUR CODE
    for i in range(get_batch_nums(batch, X)):
        # print('batch_nums', get_batch_nums(batch, X))
        # print('i', i)
        x_batch = X[i * batch: (i + 1) * batch]
        y_batch = y[i * batch: (i + 1) * batch]
        y_batch = one_hot(y_batch, np.size(W2, 1))

        x_batch = ndl.Tensor(x_batch)
        y_batch = ndl.Tensor(y_batch)
        # print('x_batch.shape--', x_batch.shape)
        # print('y_batch.shape--', y_batch.shape)

        z = ndl.matmul(ndl.relu(ndl.matmul(x_batch, W1)), W2)
        # print('z.shape--', z.shape)

        loss = softmax_loss(z, y_batch)
        # print('loss.shape--', loss.shape)
        loss.backward()

        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())

    return W1, W2


# CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h, y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
