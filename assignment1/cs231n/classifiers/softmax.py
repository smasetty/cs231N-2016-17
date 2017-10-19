import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    loss += -scores[y[i]] + np.log(np.sum(np.exp(scores)))

    for j in xrange(num_classes):
      softmax_output = np.exp(scores[j])/np.sum(np.exp(scores))
                               
      if j == y[i]:
         dW[:,j] += (softmax_output - 1) * X[i] 
      else:
         dW[:,j] +=  softmax_output * X[i]
  
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
    
  #print (X.shape)
  #print (W.shape)
  #print (y.shape)
  #print (num_train)
  #print (num_classes)


  scores = X.dot(W)
  scores -= np.max(scores, axis = 1, keepdims=True)
  denm = np.sum(np.exp(scores), axis = 1, keepdims=True)  
    
  #print ("scores shape", scores.shape)
  #print ("scores shape", scores.shape)
  #print ("den shape", denm.shape)

  probs = np.exp(scores)/denm
  correct_probs = -np.log(probs[np.arange(num_train), y])
  loss = np.sum(correct_probs)
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  dscores = probs
  dscores[np.arange(num_train), y] -= 1
  
  dW = np.dot(X.T, dscores)
  dW /= num_train
  dW += reg *W
    
  return loss, dW