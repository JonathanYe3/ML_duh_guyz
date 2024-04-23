import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# We should weight each alpha by 1-p, where p is the type 2 errors/all samples 

# Class
class AdaBoostClassifier:
    """
    Implementation of adaboost - with custom loss function
    """
    def __init__(self, n_estimators, lr, type2penalty = False, max_DT_depth = None):
        """
        Args:
            n_estimators = number of stumps the final tree will be built from
            lr = learning rate
        """
        self.n_estimators = n_estimators
        self.lr = lr
        self.stumps = []
        self.weights = None
        self.type2penalty = type2penalty
        self.DT_depth = max_DT_depth

    def exp_loss(self, error):
        # -1 is ham, 1 is spam
        #loss = np.exp(-1 * y_true * y_pred)
        loss = np.exp(error)
        return loss
    
    def fit(self, X, y):
        """
        Train the AdaBoost classifier.

        Args:
            X: Training data (2D numpy array).
            y: Training labels (1D numpy array).
            I think numpy is the best here
        """
        n_samples, n_features = X.shape
        self.weights = np.ones(n_samples) / n_samples # initial weights are 1/N

        for _ in range(self.n_estimators):
            # Fit a stump/weak learner
            #random_state = np.random.randint(1000)
            #weak_learner = DecisionTreeClassifier(max_depth = 1, random_state=random_state) # we should experiment with this
            weak_learner = BaggingClassifier(n_estimators=1)
            weak_learner.fit(X, y)

            # Predict using the weak learner
            y_pred = weak_learner.predict(X)

            # Calculate correct predictions
            correct = (y == y_pred)
            #print("correct:", np.sum(correct))

            # Update weights based on correctness
            self.weights[correct] *= np.exp(-self.lr)  # Weight down correctly classified
            self.weights[~correct] *= np.exp(self.lr)  # Weight up incorrectly classified
            #print("mean weights", np.mean(self.weights))

            self.weights /= np.sum(self.weights)

            # Store the weak learner and its weight (alpha)
            penalty = 1
            if self.type2penalty:
                penalty = np.exp(1 * self.type2err(y, y_pred))
            #print(penalty)
            alpha = self.lr * penalty * np.log(np.sum(correct) / (np.sum(~correct) + 1e-10))
            # if the proportion of type 2 error (incorrectly FTR null hypothesis) is high, we increase this weak learner's weight alpha to draw more attention to it
            self.stumps.append((weak_learner, alpha))

    def type2err(self, y, y_pred):
        """
        Calculate the proportion of type 2 errors - when the true label is 1 - spam, and the predicted label is 0 - ham

        Args:
        y: true labels
        y_pred: predicted labels
        """
        n = y.shape[0]
        errors = (y == 1) & (y_pred == 0)
        return np.sum(errors)/n


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

    