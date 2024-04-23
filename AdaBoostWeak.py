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
    def __init__(self, n_learners, type2penalty = False, lr = 0.01):
        self.n_learners = n_learners
        self.type2penalty = type2penalty
        self.lr = lr

        #classifiers
        self.clfs = []

    # def find_next_weak_learner(self):
    #     clf = DecisionStump()
    #     min_error = math.inf
    #     feature_size = self.X.shape[1]

    #     for i in range(feature_size):
    #         #feature_vals = np.expand_dims(self.X[:, i], axis=1)
    #         feature_vals = np.expand_dims(self.X[:, i].A, axis=1)

    #         unique_vals = np.unique(feature_vals)

    #         for thresh in unique_vals:
    #             prediction = np.ones(shape=self.y.shape[0])
    #             #print((self.X[:, i].A < thresh).shape)
    #             ham_idx = (self.X[:, i].A < thresh).flatten()
    #             prediction[ham_idx] = -1

    #             error = sum(self.weights[prediction != self.y])

    #             if error > 0.5:
    #                 error = 0.5 - error
    #                 polarity = -1
    #             else:
    #                 polarity = 1

    #             if error < min_error:
    #                 clf.polarity = polarity
    #                 clf.threshold = thresh
    #                 clf.axis = i
    #                 min_error = error

    #     clf.alpha = (0.5) * math.log((1.0 - min_error + 1e-8) / (min_error + 1e-8)) # small epsilon moment

    #     #Compute prediction using current clf - final vote moment
    #     pred = np.ones(np.shape(self.y))
    #     neg_idx = (clf.polarity * self.X[:, clf.axis] < clf.polarity * clf.threshold) # CHANGE THIS LATER FOR THE PENALTY
    #     pred[neg_idx] = -1 #WE MAY NEED TO REFACTOR OUR DATA - -1 AND 1

    #     #update sample weights
    #     self.weights *= np.exp(-clf.alpha * self.y * pred) # perhaps learning rate here
    #     self.weights /= np.sum(self.weights)

    #     # Append the weak classifier to the classifier list
    #     self.clfs.append(clf)

    def find_next_weak_learner(self):
        clf = DecisionStump()
        min_error = math.inf
        feature_size = self.X.shape[1]

        for i in range(feature_size):
            feature_vals = self.X[:, i].A
            unique_vals = np.unique(feature_vals)

            # Calculate predictions for all thresholds at once - this doesn't do what I want it to
            predictions = np.ones((2, self.y.shape[0]))
            print("predictions shape ", predictions.shape)
            ham_idx = (feature_vals < unique_vals[:, None]).T
            print("ham_idx shape ", ham_idx.shape)
            predictions[ham_idx] = -1

            # Calculate errors for all thresholds at once
            errors = np.sum(self.weights * (predictions != self.y), axis=1)

            # Find the threshold with the minimum error
            min_error_idx = np.argmin(errors)
            min_error = errors[min_error_idx]
            best_thresh = unique_vals[min_error_idx]

            # Determine the polarity
            if min_error > 0.5:
                min_error = 0.5 - min_error
                polarity = -1
            else:
                polarity = 1

            # Update the classifier
            clf.polarity = polarity
            clf.threshold = best_thresh
            clf.axis = i

            clf.alpha = (0.5) * np.log((1.0 - min_error + 1e-8) / (min_error + 1e-8)) # small epsilon moment

            # Compute prediction using current clf - final vote moment
            pred = np.ones(np.shape(self.y))
            neg_idx = (clf.polarity * feature_vals < clf.polarity * clf.threshold).flatten() # CHANGE THIS LATER FOR THE PENALTY
            pred[neg_idx] = -1 #WE MAY NEED TO REFACTOR OUR DATA - -1 AND 1

            # Update sample weights
            self.weights *= np.exp(-clf.alpha * self.y * pred) # perhaps learning rate here
            self.weights /= np.sum(self.weights)

            # Append the weak classifier to the classifier list
            self.clfs.append(clf)



    def fit(self, X, y):
        self.X = X
        self.y = y # need to change to -1, 1
        self.y[self.y == 0] = -1
        self.N = X.shape[0]

        #intiialize weights
        self.weights = np.ones(self.N) / self.N

        print(f'starting weights: \n {self.weights}')

        for i in range(self.n_learners):
            self.find_next_weak_learner()
            if (i+1)%10 == 0:
                print(f'learner {i+1}')

    def predict(self, X):
        clf_preds = [clf.alpha * self.stump_predict(clf, X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        # change to 0 1 labels
        y_pred[y_pred == -1] = 0

        return y_pred

    def stump_predict(self, clf, X):
        m = X.shape[0]
        pred = np.ones(m)
        neg_idx = (clf.polarity * X[:, clf.axis] < clf.polarity * clf.threshold)
        pred[neg_idx] = -1

        return pred

    

