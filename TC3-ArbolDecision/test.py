'''
The following test is designed to test creating a simple decision tree and 
making predictions on a fictional binary classification data set.
Since the test data set contains two classes (0 and 1) and the tree has been 
trained with a depth limit of 3, the predictions should reflect how the tree 
divides the samples into the leaves of the tree.
The predictions will be printed to the console and should be a list of 
predicted values for the new samples in new_samples.
'''

# Import necessary classes and functions from arbol_decision.py
from arbol_decision import DecisionTree
import numpy as np

# Create a test dataset
X = np.array([[2, 3],
              [3, 2],
              [4, 6],
              [5, 7],
              [6, 5],
              [7, 8]])

y = np.array([0, 0, 1, 1, 1, 0])

# Create an instance of DecisionTree
tree = DecisionTree(max_depth=3, min_samples_split=2, criterion='gini')

# Train the decision tree
tree.root = tree.train(X, y)


'''
If the tree has learned to separate classes 0 and 1 based on the features 
provided in new_samples, the predictions should reflect this. For example, 
you could get a list like [1, 0, 1], where each value corresponds to the 
class predicted for a sample in new_samples.
'''

# Make predictions for new samples
new_samples = np.array([[4, 5], [2, 2], [6, 7]])
predictions = tree.predict(new_samples)
print(predictions)
