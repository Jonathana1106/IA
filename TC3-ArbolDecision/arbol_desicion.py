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
        X_left, y_left, X_right, y_right = split_dataset(X, y, feature, threshold)

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


