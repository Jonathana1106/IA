import random

# Parámetros del algoritmo genético
POPULATION_SIZE = 100
NUM_GENES = 5
NUM_MAX = 50
NUM_LENGTH = 5
MUTATION_RATE = 0.1
NUM_GENERATIONS = 50  # Definir el número máximo de generaciones

def create_individual():
    return [random.randint(1, NUM_MAX) for _ in range(NUM_GENES)]

def fitness(individual, target_list):
    diff_sum = sum(abs(a - b) for a, b in zip(individual, target_list))
    fitness_value = 1.0 / (1 + diff_sum)
    return fitness_value

def crossover(parent1, parent2):
    crossover_point = random.randint(1, NUM_GENES - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual):
    mutated_individual = individual.copy()
    for i in range(NUM_GENES):
        if random.random() < MUTATION_RATE:
            mutated_individual[i] = random.randint(1, NUM_MAX)
    return mutated_individual

# Inicializar la población aleatoriamente
population = [create_individual() for _ in range(POPULATION_SIZE)]

# Generar la lista objetivo
number_list_goal = [random.randint(1, NUM_MAX) for _ in range(NUM_GENES)]

generation = 0  # Inicializar la generación en 0

while generation < NUM_GENERATIONS:  # Bucle hasta alcanzar el número máximo de generaciones
    print(f"Generación {generation + 1}")
    
    # Calcular la aptitud de cada individuo en la población
    fitness_scores = [fitness(individual, number_list_goal) for individual in population]
    
    # Imprimir información de la generación actual
    best_fitness = max(fitness_scores)
    avg_fitness = sum(fitness_scores) / len(fitness_scores)
    best_individual = population[fitness_scores.index(best_fitness)]
    print(f"Mejor aptitud: {best_fitness:.4f} | Aptitud promedio: {avg_fitness:.4f}")
    print(f"Mejor individuo: {best_individual}")
    
    # Seleccionar padres basados en la aptitud
    parents = random.choices(population, weights=fitness_scores, k=POPULATION_SIZE // 2)
    
    # Crear una nueva generación a través de cruces y mutaciones
    new_population = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        child1, child2 = crossover(parent1, parent2)
        mutate(child1)
        mutate(child2)
        new_population.extend([child1, child2])
    
    # Reemplazar la población anterior con la nueva generación
    population = new_population
    
    generation += 1  # Incrementar el contador de generaciones

# Encontrar el individuo con la mejor aptitud después de las generaciones
best_individual = max(population, key=lambda ind: fitness(ind, number_list_goal))
print("\nMejor individuo encontrado:", best_individual)

# Generar la lista objetivo
number_list_goal = best_individual
print("Lista objetivo generada:", number_list_goal)
