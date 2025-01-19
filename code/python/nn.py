import numpy as np
from util import *

# do not include any more libraries here!
# do not put any code outside of functions!


############################## Q 2.1.2 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b.
# X be [Examples, Dimensions]
def initialize_weights(in_size, out_size, params, name=""):
    W, b = None, None
    # print("Initializing W{} with shape {}".format(name, (in_size, out_size)))
    # print("Initializing b{} with shape {}".format(name, (out_size,)))

    # get variance
    limit = np.sqrt(6) / np.sqrt(in_size + out_size)

    # initialize W with numbers with mean 0 and variance of variance from uniform distribution
    W = np.random.uniform(-limit, limit, (in_size, out_size))
    
    # initialize b with zeros
    b = np.zeros(out_size)

    params["W" + name] = W
    params["b" + name] = b


############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    # compute the sigmoid of x
    res = 1 / (1 + np.exp(-x))

    return res


############################## Q 2.2.1 ##############################
def forward(X, params, name="", activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params["W" + name]
    b = params["b" + name]

    # compute the pre-activation values
    pre_act = X @ W + b

    # compute the post-activation values
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params["cache_" + name] = (X, pre_act, post_act)

    return post_act


############################## Q 2.2.2  ##############################

def softmax(x):
    # x is [examples,classes]
    # softmax should be done for each row
    res = None

    # numerically stable softmax
    max_x = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_x)
    res = e_x / np.sum(e_x, axis=1, keepdims=True)  # [examples,classes]

    return res


############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    # compute cross entropy loss
    loss = -np.sum(y * np.log(probs))

    # compute accuracy
    acc = np.mean(np.argmax(y, axis=1) == np.argmax(probs, axis=1))

    return loss, acc


############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act * (1.0 - post_act)
    return res


def backwards(delta, params, name="", activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None

    # print("Backwards for layer", name)

    # everything you may need for this layer
    W = params["W" + name]
    b = params["b" + name]
    X, pre_act, post_act = params["cache_" + name]

    # do the derivative through activation first
    # (don't forget activation_deriv is a function of post_act)
    # then compute the derivative W, b, and X

    # convert to numpy arrays
    delta = np.array(delta)
    W = np.array(W)
    X = np.array(X)
    pre_act = np.array(pre_act)
    post_act = np.array(post_act)

    loss = delta * activation_deriv(post_act) # chain rule, back propagate the error through the activation function
    grad_W = X.T @ loss # gradient of the loss with respect to weights W
    grad_b = np.sum(loss, axis=0) # gradient of the loss with respect to bias b
    grad_X = loss @ W.T # gradient of the loss with respect to input X

    # store the gradients
    params["grad_W" + name] = grad_W
    params["grad_b" + name] = grad_b
    return grad_X


############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x, y, batch_size):
    batches = []
    
    # get the number of examples
    num_examples = x.shape[0]

    # shuffle the data
    indices = np.arange(num_examples)
    shuffled_indices = np.random.permutation(indices)

    # split the data into batches
    for i in range(0, num_examples, batch_size):
        batch_indices = shuffled_indices[i:i + batch_size]
        batch_x = x[batch_indices] # get the batch examples
        batch_y = y[batch_indices] # get the batch labels
        batches.append((batch_x, batch_y)) # append the batch to the list of batches
    
    return batches
