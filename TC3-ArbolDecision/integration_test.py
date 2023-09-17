import numpy as np
from arbol_decision import DecisionTree, manual_train_test_split, cross_validation


# Define different combinations of hyperparameters to test
param_combinations = [
    {"max_depth": 3, "min_samples_split": 2, "criterion": "gini"},
    {"max_depth": 5, "min_samples_split": 2, "criterion": "gini"},
    {"max_depth": 3, "min_samples_split": 4, "criterion": "gini"},
]


# Create an instance of DecisionTree and train it on your dataset
def test_decision_tree():
    X = np.array([[2, 3],
                  [3, 2],
                  [4, 6],
                  [5, 7],
                  [6, 5],
                  [7, 8]])

    y = np.array([0, 0, 1, 1, 1, 0])

    tree = DecisionTree(max_depth=3, min_samples_split=2, criterion='gini')
    tree.root = tree.train(X, y)

    # Make predictions on new examples
    new_samples = np.array([[4, 5], [2, 2], [6, 7]])
    predictions = tree.predict(new_samples)

    # Check that predictions are consistent with expectations
    expected_predictions = [1, 0, 1]
    assert np.array_equal(predictions, expected_predictions)


# Test data splitting into training and test sets
def test_train_test_split():
    X = np.array([[2, 3],
                  [3, 2],
                  [4, 6],
                  [5, 7],
                  [6, 5],
                  [7, 8]])

    y = np.array([0, 0, 1, 1, 1, 0])

    X_train, y_train, X_test, y_test = manual_train_test_split(
        X, y, train_proportion=0.8, random_state=42)

    # Check that the sizes of the sets are as expected
    assert len(X_train) == 4
    assert len(X_test) == 2


# Test cross-validation
def test_cross_validation():
    X = np.array([[2, 3],
                  [3, 2],
                  [4, 6],
                  [5, 7],
                  [6, 5],
                  [7, 8]])

    y = np.array([0, 0, 1, 1, 1, 0])

    for params in param_combinations:
        # Evaluate the tree's performance using cross-validation for each parameter combination
        results = cross_validation(
            X, y, k=3, max_depth=params["max_depth"], min_samples_split=params["min_samples_split"], criterion=params["criterion"])

        # Check that the reported metrics are within an expected range
        assert 0.0 <= results["mean_accuracy"] <= 1.0
        assert 0.0 <= results["mean_precision"] <= 1.0
        assert 0.0 <= results["mean_recall"] <= 1.0
        assert 0.0 <= results["mean_f1"] <= 1.0


if __name__ == "__main__":
    test_decision_tree()
    test_train_test_split()
    test_cross_validation()
    print("All tests passed!")
