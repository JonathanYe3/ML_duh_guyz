import numpy as np
from sklearn.tree import DecisionTreeClassifier

# We should weight each alpha by 1-p, where p is the type 2 errors/all samples 

# Class
class AdaBoostClassifier:
    """
    Implementation of adaboost - with custom loss function
    """
    def __init__(self, n_estimators, lr, type2penalty = False):
        """
        Args:
            n_estimators = number of stumps the final tree will be built from
            lr = learning rate
        """
        self.n_estimators = n_estimators
        self.lr = lr
        self.stumps = []
        self.weights = None
        self.penalty = type2penalty

    def exp_loss(self, error):
        # -1 is ham, 1 is spam
        #loss = np.exp(-1 * y_true * y_pred)
        loss = np.exp(error)
        return loss
    
    def fit(self, X, y):
        """
        Train the AdaBoost classifier.

        Args:
            X: Training data (2D array).
            y: Training labels (1D array).
        """
        n_samples, n_features = X.shape
        self.weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # Fit a stump/weak learner
            weak_learner = DecisionTreeClassifier()
            weak_learner.fit(X, y)

            # Predict using the weak learner
            y_pred = weak_learner.predict(X)
            #print(f'predicted values \n {y_pred} \n true values \n {y}')

            # Calculate correct predictions
            correct = (y == y_pred)
            #print(np.sum(correct))

            # Update weights based on correctness
            self.weights[correct] *= np.exp(-self.lr)  # Weight down correctly classified
            self.weights[~correct] *= np.exp(self.lr)  # Weight up incorrectly classified

            self.weights /= np.sum(self.weights)

            # Store the weak learner and its weight (alpha)
            alpha = self.lr * np.log(np.sum(correct) / (np.sum(~correct) + 1e-10))
            self.stumps.append((weak_learner, alpha))

    def predict(self, X):
        """
        Predict class labels for new data.

        Args:
        X: New data (2D array).

        Returns:
        Predicted class labels (1D array).
        """
        predictions = np.zeros((X.shape[0], len(self.stumps)))
        for i, (stump, alpha) in enumerate(self.stumps):
            predictions[:, i] = stump.predict(X)

        return np.sign(np.sum(alpha * predictions, axis=1))

    