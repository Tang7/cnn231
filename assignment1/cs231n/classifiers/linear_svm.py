import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape)  # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  num_sum = 0.0
  delta = 1.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    # num_sum = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + delta # note delta = 1
      if margin > 0:
        loss += margin
        dW[j, :] += X[:, i].T
        num_sum += 1.0

    dW[y[i], :] -= num_sum * X[:, i].T
    num_sum = 0.0

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW /= num_train


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_train = X.shape[1]
  diff = np.zeros((dW.shape[0], num_train))
  # important the float number! In python the default value is integer !!!
  delta = 1.0
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  diff = W.dot(X)
  correct_score = np.zeros(y.shape)
  y_index = np.zeros(diff.shape)

	# how to implement them without loop?  see softmax.py, although the speed does not change to much
  for i in range(len(y)):
	  correct_score[i] = diff[y[i], i]
	  y_index[y[i],i] = 1.0

  correct_score -= delta
  diff -= correct_score.T

  diff[diff <= 0.0] = 0.0  # max(0, diff) function

  diff -= y_index * delta      # exclude the correct y_i value in WX, which equal to delta

  loss = np.sum(diff)
  loss /= float(num_train)
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # diff = np.clip(diff, 0.0, 1.0)  # make the diff to be float
  # should be set the >0 value to 1 !
  diff[diff>0] = 1

  num_sum = np.sum(diff, axis=0)  # so that num_sum would be float as well !

  y_index *= num_sum

  diff -= y_index
  # two accumulate sum can be converted to multiple between two matrix
  dW = diff.dot(X.T)

  dW /= num_train # again, float number in python !!!!


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
