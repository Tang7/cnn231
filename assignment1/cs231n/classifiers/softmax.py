import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[1]
  num_class = W.shape[0]

  for i in xrange(num_train):
    mapping = W.dot(X[:,i])
    p = np.zeros(mapping.shape)
    mapping -= np.max(mapping)
    p = np.exp(mapping) / np.sum(np.exp(mapping))
    loss += -np.log(p[y[i]])

    # loss += 0.5 * reg * np.sum(W * W)
    for j in xrange(num_class):
      if j == y[i]:
        dW[y[i], :] += (p[y[i]]-1) * X[:, i].T

      if j != y[i]:
	      dW[j, :] += p[j] * X[:, i].T

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  mapping = W.dot(X)
  C = np.amax(mapping, axis = 0) # maximum in each column
  mapping -= C                   # normalize the mapping value

  # replace the loop with python embedded function !!!
  '''
  for i in range(len(y)):
	  correct_score[i] = mapping[y[i], i]
	  y_index[y[i],i] = 1.0
	'''

  y_range = np.arange(len(y))

  # sum_score = np.sum(np.exp(mapping), axis = 0))
  sum_score = np.sum(np.exp(mapping), axis = 0)
  p = np.exp(mapping) / sum_score

  loss = -np.sum(np.log(p[y, y_range]))   # loss only contain the correct index value

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  correct_index = np.zeros(mapping.shape)
  correct_index[y, y_range] = 1
  dW = (p - correct_index).dot(X.T)
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
