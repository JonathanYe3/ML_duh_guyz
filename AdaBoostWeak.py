# thank you https://github.com/AlvaroCorrales/AdaBoost/blob/main/AdaBoost.py
# Imports
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Helper functions
def compute_error(y, y_pred, w_i, type2penalty):
    '''
    Calculate the error rate of a weak classifier m. Arguments:
    y: actual target value
    y_pred: predicted value by weak classifier
    w_i: individual weights for each observation

    
    Note that all arrays should be the same length. Convert sparse array to regular array
    '''
    if type2penalty:
        error = (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)
    else:
        type2 = type2err(y, y_pred)
        error = np.exp(type2)*(sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

    return error

def compute_alpha(error):
    '''
    Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
    alpha in chapter 10.1 of The Elements of Statistical Learning. Arguments:
    error: error rate from weak classifier m
    '''
    return np.log((1 - error) / error)

def update_weights(w_i, alpha, y, y_pred):
    ''' 
    Update individual weights w_i after a boosting iteration. Arguments:
    w_i: individual weights for each observation
    y: actual target value
    y_pred: predicted value by weak classifier  
    alpha: weight of weak classifier used to estimate y_pred
    '''  
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))

def type2err(y, y_pred):
        """
        Calculate the proportion of type 2 errors - when the true label is 1 - spam, and the predicted label is 0 - ham

        Args:
        y: true labels
        y_pred: predicted labels
        """
        n = y.shape[0]
        errors = (y == 1) & (y_pred == 0)
        return np.sum(errors)/n

# Define AdaBoost class
class AdaBoostWeak:
    
    def __init__(self, rounds, type2penalty = False, maxDTdepth = 1):
        # self.w_i = None
        self.alphas = []
        self.stumps = []
        self.rounds = rounds
        self.training_errors = []
        self.prediction_errors = []
        self.type2penalty = type2penalty
        self.maxDTdepth = maxDTdepth

    def fit(self, X, y):
        '''
        Fit model. Arguments:
        X: independent variables
        y: target variable
        rounds: number of boosting rounds. Default is 100
        '''
        
        # Clear before calling
        self.alphas = [] 
        self.training_errors = []
        y[y == 0] = -1

        # Iterate over M weak classifiers
        for m in range(0, self.rounds):
            
            # Set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N
            else:
                w_i = update_weights(w_i, alpha_m, y, y_pred)
            # print(w_i)
            
            # (a) Fit weak classifier and predict labels
            stump = DecisionTreeClassifier(max_depth = self.maxDTdepth)     # Stump: Two terminal-node classification tree
            stump.fit(X, y, sample_weight = w_i)
            y_pred = stump.predict(X)
            
            self.stumps.append(stump) # Save to list of weak classifiers

            # (b) Compute error
            error_m = compute_error(y, y_pred, w_i, self.type2penalty)
            self.training_errors.append(error_m)
            # print(error_m)

            # (c) Compute alpha
            alpha_m = compute_alpha(error_m)
            # if self.type2penalty:
            #     penalty = type2err(y = y, y_pred = y_pred)
            #     alpha_m *= penalty
            self.alphas.append(alpha_m)
            # print(alpha_m)

        assert len(self.stumps) == len(self.alphas)


    def predict(self, X):
        '''
        Predict using fitted model. Arguments:
        X: independent variables
        '''

        # Initialize an array to store the final predictions
        final_predictions = np.zeros(X.shape[0])

        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(self.rounds):
            y_pred_m = self.stumps[m].predict(X) * self.alphas[m]
            final_predictions += y_pred_m

        # Estimate final predictions
        y_pred = np.sign(final_predictions).astype(int) # need to change -1 to 0  
        y_pred[y_pred == -1] = 0
        return y_pred
      
    def error_rates(self, X, y):
        '''
        Get the error rates of each weak classifier. Arguments:
        X: independent variables
        y: target variables associated to X
        '''
        
        self.prediction_errors = [] # Clear before calling
        
        # Predict class label for each weak classifier
        for m in range(self.rounds):
            y_pred_m = self.stumps[m].predict(X)          
            error_m = compute_error(y = y, y_pred = y_pred_m, w_i = np.ones(len(y)))
            self.prediction_errors.append(error_m)

            
            
            
            
    def get_params(self, deep=True):
            return {'rounds': self.rounds,
                    'type2penalty': self.type2penalty,
                    'maxDTdepth': self.maxDTdepth}

    def set_params(self, **params):
        for key, value in params.items():
            
            
            if key == 'maxDTdepth':
                self.maxDTdepth = value
            else:
                setattr(self, key, value)
        return self
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)