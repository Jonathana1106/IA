from collections import Counter
import numpy as np

# Step 1: Node Class Creation


class Node:
    def __init__(self, feature=None, threshold=None, gini=None, sample_count=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.gini = gini
        self.sample_count = sample_count
        self.value = value
        self.left = left
        self.right = right

# Function to find the most common class in a list of labels


def most_common_value(y):
    class_counts = Counter(y)
    most_common_class = class_counts.most_common(1)[0][0]
    return most_common_class

# Function to select the best feature and threshold for splitting


def select_best_split(X, y, criterion='gini'):
    best_gini = float('inf')
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        unique_thresholds = np.unique(X[:, feature])
        for threshold in unique_thresholds:
            left_indices = X[:, feature] <= threshold
            right_indices = X[:, feature] > threshold
            gini = calculate_gini(y[left_indices], y[right_indices], criterion)

            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


def calculate_gini(y_left, y_right, criterion='gini'):
    if criterion == 'gini':
        gini_left = gini_index(y_left)
        gini_right = gini_index(y_right)
        total_samples = len(y_left) + len(y_right)
        weighted_gini = (len(y_left) / total_samples) * \
            gini_left + (len(y_right) / total_samples) * gini_right
        return weighted_gini
    else:
        # Implementar cálculo de otro criterio si es necesario
        pass

# Función para calcular el índice Gini de un conjunto de etiquetas


def gini_index(labels):
    num_samples = len(labels)
    if num_samples == 0:
        return 0.0
    class_counts = Counter(labels)
    gini = 1.0
    for class_count in class_counts.values():
        class_probability = class_count / num_samples
        gini -= class_probability ** 2

    return gini


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
            value = most_common_value(y)
            return Node(value=value)

        # Choose the best feature and threshold to split the dataset
        feature, threshold = select_best_split(X, y, self.criterion)

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
