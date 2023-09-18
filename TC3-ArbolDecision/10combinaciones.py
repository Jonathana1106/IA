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

y = np.array([1, 0, 1, 1, 1, 0])

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
    print(f"\t Punteria Media: {result['metrics']['mean_accuracy']}")
    print(f"\t Punteria Desviacion Estandar: {result['metrics']['std_accuracy']}")
    print(f"\t Precision Media: {result['metrics']['mean_precision']}")
    print(f"\t Precision Desviacion Estandar: {result['metrics']['std_precision']}")
    print(f"\t Recall Media: {result['metrics']['mean_recall']}")
    print(f"\t Recall Desviacion Estandar: {result['metrics']['std_recall']}")
    print(f"\t F1 Media: {result['metrics']['mean_f1']}")
    print(f"\t Desviacion Estandar F1: {result['metrics']['std_f1']}")
    print("\n")

"""
Para los diferentes parámetros y los 2 criterios seleccionados respectivamente. Los resultados de media no cambian
si se altera la profundidad y la cantidad de splits dentro de la muestra bajo los criterios seleccionadors. Por lo
tanto, lo único que queda es verificar cuál de los modelos nos da mejores resultados. En este caso, el mejor modelo
es el gini y ese es el que escogemos.
"""

# Analisis del modelo
best_model_idx = np.argmax([result['metrics']['mean_accuracy'] for result in results])
best_model_params = results[best_model_idx]['params']
best_model_metrics = results[best_model_idx]['metrics']

print("Mejor modelo:")
print(f"Parametros: {best_model_params}")
print("Metricas en Cross-Validation:")
print(f"\t Punteria Media: {best_model_metrics['mean_accuracy']}")
print(f"\t Punteria Desviacion Estandar: {best_model_metrics['std_accuracy']}")
print(f"\t Precision Media: {best_model_metrics['mean_precision']}")
print(f"\t Precision Desviacion Estandar: {best_model_metrics['std_precision']}")
print(f"\t Recall Media: {best_model_metrics['mean_recall']}")
print(f"\t Recall Desviacion Estandar: {best_model_metrics['std_recall']}")
print(f"\t F1 Media: {best_model_metrics['mean_f1']}")
print(f"\t Desviacion Estandar F1: {best_model_metrics['std_f1']}")

# Step 7: Prueba en el set de prueba
selected_model = DecisionTree(**best_model_params)
selected_model.root = selected_model.train(X_train, y_train)

test_predictions = selected_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions)
test_recall = recall_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions)

print("\n Metricas en el set de prueba:")
print(f"Punteria: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1}")

"""
Conclusiones:
"""