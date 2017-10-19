import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #print num_train
  #print num_classes
  #print W.shape
  #print X.shape  
  #print dW[:,1].shape
  #print X[1].shape
  #print('Sharat')
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        count = count + 1
        loss += margin
        dW[:,j] += X[i]
    dW[:,y[i]] -= count * X[i]
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W
    
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
   
  delta = 1.0
  num_train = X.shape[0]
  scores = X.dot(W)
    
  #print "scores shape", scores.shape
  #print "y shape", y.shape
  #print "np.matrix(y).T.shape", np.matrix(y).T.shape
  #print test.shape

  test = scores[np.arange(num_train), y]

  margins = np.maximum(0, (scores - np.matrix(test).T) + delta)
  loss = np.sum(margins)
  loss -= num_train * delta
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)  
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  X_zeros = np.zeros(margins.shape)
  X_zeros[margins > 0] = 1
  incorrect_counts = np.sum(X_zeros, axis = 1)
  X_zeros[np.arange(num_train), y] = 0
  X_zeros[np.arange(num_train), y] -= incorrect_counts

  #print margins.shape
  #print X_zeros.shape
  #print X_zeros[1]
  #print incorrect_counts.shape
  #print "X.shape", X.shape
  #print "X_zeros.shape", X_zeros.shape
  #print "scores.shape", scores.shape
  #print "margins shape", margins.shape
  
  dW = X.T.dot(X_zeros)
  dW /= num_train
  dW += reg * W


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
