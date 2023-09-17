import numpy as np
from arbol_decision import *
import warnings

# Ignorar warnings
warnings.filterwarnings("ignore")

# Combinaciones
param_combinations = [
    {'max_depth': None, 'min_samples_split': 2, 'criterion': 'gini'},
    {'max_depth': None, 'min_samples_split': 4, 'criterion': 'entropy'},
    {'max_depth': 5, 'min_samples_split': 2, 'criterion': 'gini'},
    {'max_depth': 5, 'min_samples_split': 4, 'criterion': 'entropy'},
    {'max_depth': 10, 'min_samples_split': 2, 'criterion': 'gini'},
    {'max_depth': 10, 'min_samples_split': 4, 'criterion': 'entropy'},
    {'max_depth': 15, 'min_samples_split': 2, 'criterion': 'gini'},
    {'max_depth': 15, 'min_samples_split': 4, 'criterion': 'entropy'},
    {'max_depth': 20, 'min_samples_split': 2, 'criterion': 'gini'},
    {'max_depth': 20, 'min_samples_split': 4, 'criterion': 'entropy'}
]

# Lista para almacenar los resultados de las metricas
results = []

# Iterar sobre las combinaciones de parametros
X = np.array([[2, 1],
              [3, 2],
              [4, 3],
              [6, 4],
              [7, 5],
              [8, 6]])

y = np.array([1, 0, 1, 1, 1])

X_train, y_train, X_test, y_test = manual_train_test_split(X, y, train_proportion=0.8, random_state=42)

for params in param_combinations:
    # Entrenar y evaluar el modelo utilizando validacion cruzada
    metrics = cross_validation(X_train, y_train, k=5, **params)

    # Agrega los resultados a la lista
    results.append({
        'params': params,
        'metrics': metrics
    })

# Resultados 10 combinaciones
for idx, result in enumerate(results):
    print(f"Combinacion {idx + 1}:")
    print(f"Parametros: {result['params']}")
    print("Metricas:")
    print(f"\tPrecision Media: {result['metrics']['mean_accuracy']}")
    print(f"\tPrecision Media: {result['metrics']['std_accuracy']}")
    print(f"\tSensibilidad Media: {result['metrics']['mean_precision']}")
    print(f"\tPuntuacion F1 Media: {result['metrics']['std_precision']}")
    print(f"\tPrecision Media: {result['metrics']['mean_recall']}")
    print(f"\tPrecision Media: {result['metrics']['std_recall']}")
    print(f"\tSensibilidad Media: {result['metrics']['mean_f1']}")
    print(f"\tPuntuacion F1 Media: {result['metrics']['std_f1']}")
    print("\n")