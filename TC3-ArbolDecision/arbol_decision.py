import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Step 1: Node Class Creation
class Node:
    def __init__(self, feature=None, threshold=None, impurity=None, sample_count=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.sample_count = sample_count
        self.value = value
        self.left = left
        self.right = right


# Function to find the most common class in a list of labels
def most_common_class(y):
    class_counts = Counter(y)
    most_common = class_counts.most_common(1)[0][0]
    return most_common


# Function to select the best feature and threshold for splitting
def find_best_split(X, y, criterion='gini'):
    best_impurity = float('inf')
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        unique_thresholds = np.unique(X[:, feature])
        for threshold in unique_thresholds:
            left_indices = X[:, feature] <= threshold
            right_indices = X[:, feature] > threshold
            impurity = calculate_impurity(
                y[left_indices], y[right_indices], criterion)

            if impurity < best_impurity:
                best_impurity = impurity
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


def calculate_impurity(y_left, y_right, criterion='gini'):
    if criterion == 'gini':
        impurity_left = gini_impurity(y_left)
        impurity_right = gini_impurity(y_right)
        total_samples = len(y_left) + len(y_right)
        weighted_impurity = (len(y_left) / total_samples) * impurity_left + \
            (len(y_right) / total_samples) * impurity_right
        return weighted_impurity
    elif criterion == 'entropy':
        entropy_left = entropy_impurity(y_left)
        entropy_right = entropy_impurity(y_right)
        total_samples = len(y_left) + len(y_right)
        weighted_entropy = (len(y_left) / total_samples) * entropy_left + \
            (len(y_right) / total_samples) * entropy_right
        return weighted_entropy
    else:
        raise ValueError(
            "Invalid criterion. Supported criteria are 'gini' and 'entropy'.")


# Function to calculate the entropy of a set of labels
def entropy_impurity(labels):
    num_samples = len(labels)
    if num_samples == 0:
        return 0.0

    class_counts = Counter(labels)
    impurity = 0.0
    for class_count in class_counts.values():
        class_probability = class_count / num_samples
        impurity -= class_probability * np.log2(class_probability)

    return impurity


# Function to calculate the Gini index of a set of labels
def gini_impurity(labels):
    num_samples = len(labels)
    if num_samples == 0:
        return 0.0
    class_counts = Counter(labels)
    impurity = 1.0
    for class_count in class_counts.values():
        class_probability = class_count / num_samples
        impurity -= class_probability ** 2

    return impurity


# Function to split the dataset into left and right subsets
def split_dataset(X, y, feature, threshold):
    left_indices = X[:, feature] <= threshold
    right_indices = X[:, feature] > threshold

    X_left = X[left_indices]
    y_left = y[left_indices]

    X_right = X[right_indices]
    y_right = y[right_indices]

    return X_left, y_left, X_right, y_right


# Step 2: Decision Tree Class Creation
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def train(self, X, y, depth=0):
        # Check stopping criteria
        if depth == self.max_depth or len(X) < self.min_samples_split:
            # Create a leaf node with the majority class or the average value
            # depending on the problem (classification or regression)
            # Example for classification:
            value = most_common_class(y)
            return Node(value=value)

        # Choose the best feature and threshold to split the dataset
        feature, threshold = find_best_split(X, y, self.criterion)

        # Split the dataset into left and right subsets
        X_left, y_left, X_right, y_right = split_dataset(
            X, y, feature, threshold)

        # Recursively build the sub-trees
        left = self.train(X_left, y_left, depth=depth + 1)
        right = self.train(X_right, y_right, depth=depth + 1)

        # Create and return a decision node
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def predict(self, X):
        # Initialize an array to store the predictions
        predictions = []

        # Traverse the decision tree for each sample in X
        for sample in X:
            node = self.root
            while node.left:
                if sample[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            # Append the predicted value for this sample
            predictions.append(node.value)

        return predictions


# Step 3: Data Splitting (Example using Numpy)
def manual_train_test_split(X, y, train_proportion=0.8, random_state=None):
    # Calculate the number of training samples
    n_train_samples = int(train_proportion * len(X))

    if random_state is not None:
        # Set the seed for pseudo-random number generation
        np.random.seed(random_state)

    # Split the data into training and test sets
    X_train, y_train = X[:n_train_samples], y[:n_train_samples]
    X_test, y_test = X[n_train_samples:], y[n_train_samples:]

    return X_train, y_train, X_test, y_test


# Step 4: Cross-Validation Implementation
def cross_validation(X, y, k=5, max_depth=None, min_samples_split=2, criterion='gini'):
    # Split the training set into k subsets
    subsets_X = np.array_split(X, k)
    subsets_y = np.array_split(y, k)

    # Lists to store metrics for each model
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for i in range(k):
        # Select the current validation set
        X_valid = subsets_X[i]
        y_valid = subsets_y[i]

        # Create the training set excluding the validation set
        X_train = np.concatenate([subsets_X[j] for j in range(k) if j != i])
        y_train = np.concatenate([subsets_y[j] for j in range(k) if j != i])

        # Train a decision tree model
        tree = DecisionTree(
            max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)
        tree.root = tree.train(X_train, y_train)

        # Make predictions on the validation set
        predictions = tree.predict(X_valid)

        # Calculate metrics and record them
        accuracy = accuracy_score(y_valid, predictions)
        precision = precision_score(y_valid, predictions)
        recall = recall_score(y_valid, predictions)
        f1 = f1_score(y_valid, predictions)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Calculate the mean and standard deviation of the metrics
    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    std_precision = np.std(precision_scores)
    mean_recall = np.mean(recall_scores)
    std_recall = np.std(recall_scores)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    return {
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_precision": mean_precision,
        "std_precision": std_precision,
        "mean_recall": mean_recall,
        "std_recall": std_recall,
        "mean_f1": mean_f1,
        "std_f1": std_f1
    }
