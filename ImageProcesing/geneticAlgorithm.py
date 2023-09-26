import cv2
from ImageProcesing.preprocessing import *
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import random
from skimage.metrics import structural_similarity as ssim
import os  # Add this import

# Function to flatten an image
def flatten_image(image):
    return image.flatten()

# Function to create an initial population of images based on a reference image
def initialize_population(population_size, reference_image, mutation_rate):
    population = []
    for _ in range(population_size):
        individual = reference_image.copy()
        individual = mutation(individual, mutation_rate)
        population.append(individual)
    return population

# Function to evaluate the fitness of an image
def fitness(image, target_image):
    # Implement your fitness evaluation logic here
    # This could be based on image quality, similarity to target, etc.
    ssim_value = ssim(image, target_image, multichannel=True)  # Set multichannel=True for color images
    # Return a higher value for better fitness.
    return -ssim_value

# Function for one-point crossover
def one_point_crossover(parent1, parent2):
    # Choose a random crossover point
    crossover_point = np.random.randint(0, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Function for mutation
def mutation(individual, mutation_rate):
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if np.random.rand() < mutation_rate:
            # Apply mutation by randomly changing pixel value
            mutated_individual[i] = np.random.randint(0, 256)
    return mutated_individual

def genetic_algorithm(population, target_image, generations, mutation_rate, progressBar):

    progressBar.setValue(0)
    progressBar.setMaximum(generations)
    progressBar.setFormat("0")
    progressBar.update()

    for generation in range(generations):

        progressBar.setFormat(str(generation+1))
        progressBar.update()

        # Evaluate fitness for each individual
        fitness_values = [fitness(individual, target_image) for individual in population]

        # Select indices of parents using tournament selection
        parent_indices = np.random.choice(len(population), len(population), replace=True)
        parents = [population[i] for i in parent_indices]

        # Create the next generation through crossover and mutation
        new_population = []
        while len(new_population) < len(population):
            parent1, parent2 = random.sample(parents, 2)  # Randomly select two parents
            child1, child2 = one_point_crossover(parent1, parent2)
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            new_population.extend([child1, child2])

        # Replace the old population with the new population
        population = new_population

        # Print the best fitness in this generation
        best_fitness = min(fitness_values)
        print(f"Generation {generation+1}, Best Fitness: {best_fitness}")

        progressBar.setValue(generation+1)

    # Return the best individual (image) found
    best_individual = population[np.argmin(fitness_values)]
    return best_individual

## Pyqt progress bar as progressBar
def main(epath, objPath, generations=10, population_size=50, mutation_rate=0.01, ui = None):

    enhancedImage = cv2.imread(epath)
    #objectiveImage = cv2.imread(objPath)

    flattened_enhanced_image = flatten_image(enhancedImage)
    #flattened_objective_image = flatten_image(objectiveImage)

    population = initialize_population(population_size, flattened_enhanced_image, mutation_rate)

    # Run the genetic algorithm to enhance the image
    best_image = genetic_algorithm(population, flattened_enhanced_image, generations, mutation_rate, ui.progressBar)

    # Reshape the best image to its original shape
    best_image = best_image.reshape(enhancedImage.shape)

    ui.generatedPic_View.setPixmap(QtGui.QPixmap(best_image))

    # Display or save the best-enhanced image
    cv2.imshow("Best Enhanced Image", best_image)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    