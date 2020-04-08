"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    
    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    m = y.shape[0]
    
    #Calculating for each data point
    for i in range(X.shape[0]):
        score = X[i].dot(W)
    
        nominator = np.exp(score)
        denominator = np.sum(nominator)
    
        p = nominator / denominator 
        log_p = np.log(p)
    
    

        log_likelihood = -np.log(p[y[i]])
        loss+=log_likelihood
        
        
        #C classes
        
        for j in range(W.shape[1]):
            if(j == y[i]): #We have the correct index: 
                new_p = p[y[i]] - 1
            else:
                new_p = p[j]
            
            #dW[:, j] += (p[y[i]]-(j == y[i])) * X[i, :]  
            #dW[:,j] += new_p * X[i,:]
            #assert(j <= 10)
            #assert(new_p.shape == (10,))
            #ctr = 0
            #Parsing through each neuron in the input vector
            for d in range(X.shape[1]): 
                dW[d,j] += new_p * X[i,d]
                #ctr+=1
            
    loss = (loss / m) + (.5 * reg * np.sum(W * W))
    dW /= m
    
    dW += reg * W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################
    
    
   
    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    #We get a N X C matrix
    scores = X.dot(W)
    prob = np.exp(scores) / np.sum(np.exp(scores),axis=1, keepdims=True)
    #prob[:,y_dev]
    log_likelihood = -np.log(prob[np.arange(prob.shape[0]),y[:]])
    loss += np.sum(log_likelihood) / y.shape[0]
    
    prob[np.arange(prob.shape[0]),y[:]] -= 1
    
    prob /= y.shape[0]
    dW = X.T.dot(prob)
    dW += reg * W
    loss += (.5 * reg * np.sum(W * W))
    print(prob)
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 2.5e-7, 5e-7]
    regularization_strengths = [1e4, 2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    
    reg = []
  
    best_learning_rate = None
    best_reg = None
    best_train_acc = -1
    
    for r in regularization_strengths: 
        for l in learning_rates:
            candidate_softmax = SoftmaxClassifier()
            candidate_softmax.train(X_train, y_train, learning_rate=l, reg=r ,num_iters=1500, verbose=True)
            y_train_pred = candidate_softmax.predict(X_train)
            y_val_pred = candidate_softmax.predict(X_val)
            y_train_acc = np.mean(y_train == y_train_pred)
            y_val_acc = np.mean(y_val == y_val_pred)
            
            if(y_val_acc > best_val):
                best_val = y_val_acc
                best_learning_rate = l
                best_reg = r
                best_train_acc = y_train_acc
                best_softmax = candidate_softmax
    
            results[(l,r)] = (y_train_acc,y_val_acc)       
            all_classifiers.append(candidate_softmax) 
    #highest_accuracy = np.max(i[0] for i in reg)
    

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
