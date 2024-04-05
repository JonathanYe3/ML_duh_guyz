import numpy as np

# Class
class AdaBoostClassifer:
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
            weak_learner.fit(X, y, sample_weight=self.weights)

            # Predict using the weak learner
            y_pred = weak_learner.predict(X)

            # Calculate the total error/loss of the weak learner - using 0/1 loss - can change this guy later
            error = np.sum(self.weights * (y != y_pred))

            # Check for termination (all weights are 0)
            if error == 0.0:
                break

            # Update weights based on error
            self.weights[y != y_pred] *= np.exp(self.learning_rate * error)
            self.weights /= np.sum(self.weights)

            # Store the weak learner and its weight (alpha)
            alpha = self.learning_rate * np.log(1.0 / (error + 1e-10))
            if self.type2penalty:
                break
            self.weak_learners.append((weak_learner, alpha))

    def predict(self, X):
        """
        Predict class labels for new data.

        Args:
        X: New data (2D array).

        Returns:
        Predicted class labels (1D array).
        """
        predictions = np.zeros((X.shape[0], len(self.weak_learners)))
        for i, (weak_learner, alpha) in enumerate(self.weak_learners):
            predictions[:, i] = weak_learner.predict(X)

        return np.sign(np.sum(alpha * predictions, axis=1))

    

### Decision tree stump guy

class DecisionTreeNode:
  """
  Decision tree node class.
  """
  def __init__(self, feature_index=None, threshold=None, left_child=None, right_child=None, class_label=None):
    self.feature_index = feature_index  # Index of the feature used for splitting
    self.threshold = threshold  # Threshold value for splitting
    self.left_child = left_child  # Left child node
    self.right_child = right_child  # Right child node
    self.class_label = class_label  # Class label (for leaf nodes)


class DecisionTreeClassifier:
  """
  Simple Decision Tree Classifier.
  """
  def __init__(self, max_depth=2):
    self.max_depth = max_depth
    self.root = None

  def _entropy(self, y):
    """
    Calculate entropy of a label distribution.
    """
    unique, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p + 1e-10))

  def _information_gain(self, parent_entropy, left_entropy, right_entropy, weights):
    """
    Calculate information gain for a split.
    """
    if weights is None:
      weights = np.ones(len(parent_entropy))
    return parent_entropy - (np.sum(weights[left_entropy.index] * left_entropy) +
                             np.sum(weights[right_entropy.index] * right_entropy))

  def _find_best_split(self, X, y, sample_weight=None):
    """
    Find the best feature and threshold for splitting.
    """
    best_feature = None
    best_threshold = None
    best_gain = 0
    n_features = X.shape[1]
    parent_entropy = self._entropy(y)

    for feature_index in range(n_features):
      unique_values = np.unique(X[:, feature_index])
      for threshold in (unique_values[:-1] + unique_values[1:]) / 2:
        left_idx = X[:, feature_index] <= threshold
        right_idx = ~left_idx
        if sample_weight is None:
          left_y, right_y = y[left_idx], y[right_idx]
        else:
          left_y, right_y = y[left_idx], y[right_idx]
          left_weight, right_weight = sample_weight[left_idx], sample_weight[right_idx]
        if len(left_y) == 0 or len(right_y) == 0:
          continue

        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)
        gain = self._information_gain(parent_entropy, left_entropy, right_entropy, left_weight if sample_weight is not None else None)
        if gain > best_gain:
          best_feature = feature_index
          best_threshold = threshold
          best_gain = gain

    return best_feature, best_threshold

  def _build_tree(self, X, y, sample_weight=None, depth=0):
    """
    Recursively build the decision tree.
    """
    if depth >= self.max_depth or len(np.unique(y)) == 1:
        return DecisionTreeNode(class_label=np.argmax(np.average(y, weights=sample_weight)))

    best_feature, best_threshold = self._find_best_split(X, y, sample_weight)
    left_idx = X[:, best_feature] <= best_threshold
    right_idx = ~left_idx
    left_child = self._build_tree(X[left_idx], y[left_idx], sample_weight[left_idx] if sample_weight is not None else None, depth + 1)
    right_child = self._build_tree(X[right_idx], y[right_idx], sample_weight[right_idx] if sample_weight is not None else None, depth + 1)

    # Create and return the current node (root or internal node)
    return DecisionTreeNode(feature_index=best_feature, 
                            threshold=best_threshold,
                             left_child=left_child, 
                             right_child=right_child) 
