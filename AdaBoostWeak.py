import numpy as np
import math

class DecisionStump:
    def __init__(self):
        #initialize the stumps
        self.polarity = 1
        self.axis = None
        self.threshold = None

        self.alpha = None



class AdaBoostWeak:
    def __init__(self, X, y, n_learners, type2penalty = False):
        self.X = X
        self.y = y
        self.n_learners = n_learners
        self.N = X.shape[0]
        self.type2penalty = type2penalty

        #intiialize weights
        self.weights = np.ones(self.N) / self.N

        #classifiers
        self.clfs = []

    def find_next_weak_learner(self):
        clf = DecisionStump()
        min_error = math.inf
        feature_size = self.X.shape[1]

        for i in range(feature_size):
            feature_vals = np.expand_dims(self.X[:, i], axis=1)
            unique_vals = np.unique(feature_vals)

            for thresh in unique_vals:
                prediction = np.ones(shape=self.y.shape[0])
                prediction[self.X[:, i] < thresh] = -1

                error = sum(self.w[prediction != self.y])

                if error > 0.5:
                    error = 0.5 - error
                    polarity = -1
                else:
                    polarity = 1

                if error < min_error:
                    clf.polarity = polarity
                    clf.threshold = thresh
                    clf.axis = i
                    min_error = error

        clf.alpha = (0.5) * math.log((1.0 - min_error) / (min_error + 1e-8)) # small epsilon moment

        #Compute prediction using current clf - final vote moment
        pred = np.ones(np.shape(self.y))
        neg_idx = (clf.polarity * self.X[:, clf.axis] < clf.polarity * clf.threshold) # CHANGE THIS LATER FOR THE PENALTY
        pred[neg_idx] = -1 #WE MAY NEED TO REFACTOR OUR DATA - -1 AND 1

        #update sample weights
        self.weights *= np.exp(-clf.alpha * self.y * pred)
        self.weights /= np.sum(self.weights)

        # Append the weak classifier to the classifier list
        self.clfs.append(clf)

    def fit(self):
        for _ in range(self.n_learners):
            self.find_next_weak_learner()

    def predict(self, X):
        clf_preds = [clf.alpha * self.stump_predict(clf, X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred

    def stump_predict(self, clf, X):
        m = X.shape[0]
        pred = np.ones(m)
        neg_idx = (clf.polarity * X[:, clf.axis] < clf.polarity * clf.threshold)
        pred[neg_idx] = -1

        return pred

    

