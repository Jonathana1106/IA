import cv2
from ImageProcesing.preprocessing import *
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import random
from skimage.metrics import structural_similarity as ssim
import os
from ImageProcesing.pointillism import *
import math
os.environ['QT_QPA_PLATFORM'] = 'xcb'

fileName = ""
## TODO: Video
imageGenerationsList = []

##############################################################################################
def load_image(image_path, size_limit=0):
    """
    Carga una imagen desde la ruta especificada y opcionalmente la limita a un tamaño máximo.

    Args:
        image_path (str): Ruta de la imagen a cargar.
        size_limit (int, optional): Tamaño máximo de la imagen. Por defecto, es 0 (sin límites).

    Returns:
        numpy.ndarray: La imagen cargada.
    """
    img = cv2.imread(image_path)

    if size_limit > 0:
        img = limit_size(img, size_limit)

    return img


def calculate_stroke_scale(img, custom_scale=0):
    """
    Calcula la escala de los trazos del pincel automáticamente o utiliza una escala personalizada.

    Args:
        img (numpy.ndarray): La imagen de entrada.
        custom_scale (int, optional): Escala de trazo personalizada. Por defecto, es 0 (automática).

    Returns:
        int: La escala de los trazos del pincel.
    """
    if custom_scale == 0:
        stroke_scale = int(math.ceil(max(img.shape) / 1000))
        print("Escala de trazo automáticamente seleccionada: %d" % stroke_scale)
        return stroke_scale
    else:
        return custom_scale


def calculate_gradient_smoothing_radius(img, custom_radius=0):
    """
    Calcula el radio de suavizado del gradiente automáticamente o utiliza un radio personalizado.

    Args:
        img (numpy.ndarray): La imagen de entrada.
        custom_radius (int, optional): Radio de suavizado personalizado. Por defecto, es 0 (automático).

    Returns:
        int: El radio de suavizado del gradiente.
    """
    if custom_radius == 0:
        gradient_smoothing_radius = int(round(max(img.shape) / 50))
        print("Radio de suavizado de gradientes automáticamente seleccionado: %d" %
              gradient_smoothing_radius)
        return gradient_smoothing_radius
    else:
        return custom_radius


def generate_color_palette(img, palette_size):
    """
    Genera una paleta de colores a partir de la imagen utilizando el algoritmo K-Means.

    Args:
        img (numpy.ndarray): La imagen de entrada.
        palette_size (int): Número de colores en la paleta.

    Returns:
        ColorPalette: Una instancia de ColorPalette basada en los colores extraídos de la imagen.
    """
    print("Calculando paleta de colores...")
    palette = ColorPalette.from_image(img, palette_size)

    print("Extender paleta de colores...")
    palette = palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])

    # display_palette(palette)  # Mostrar la paleta de colores

    return palette


def display_palette(palette):
    """
    Muestra una representación visual de la paleta de colores en una ventana emergente.

    Args:
        palette (ColorPalette): La paleta de colores a mostrar.
    """
    print("Mostrando la paleta de colores")
    cv2.imshow("palette", palette.to_image())
    cv2.waitKey(200)


def calculate_gradients(img):
    """
    Calcula el campo de vectores del gradiente a partir de una imagen en escala de grises.

    Args:
        img (numpy.ndarray): La imagen en escala de grises.

    Returns:
        VectorField: El campo de vectores del gradiente calculado.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Calculando gradientes...")
    gradient = VectorField.from_gradient(gray)

    return gradient


def smooth_gradients(gradient, smoothing_radius):
    """
    Suaviza el campo de vectores del gradiente utilizando un filtro de suavizado.

    Args:
        gradient (VectorField): El campo de vectores del gradiente.
        smoothing_radius (int): El radio del filtro de suavizado.
    """
    print("Suavizando gradientes...")
    gradient.smooth(smoothing_radius)


def create_cartoonized_image(img):
    """
    Crea una versión "cartonizada" de la imagen para usar como base para la pintura.

    Args:
        img (numpy.ndarray): La imagen original.

    Returns:
        numpy.ndarray: La imagen "cartonizada".
    """
    print("Generating cartoonized image...")
    return cv2.medianBlur(img, 11)


def draw_painting(img, grid, palette, gradient, stroke_scale, x_size, y_size):
    """
    Dibuja la pintura en la imagen de acuerdo con una cuadrícula aleatoria de ubicaciones de trazos.

    Args:
        img (numpy.ndarray): La imagen en la que se dibujará la pintura.
        grid (list): Lista de ubicaciones de trazos de pincel.
        palette (ColorPalette): La paleta de colores para la pintura.
        gradient (VectorField): El campo de vectores del gradiente.
        stroke_scale (int): La escala de los trazos del pincel.

    Returns:
        numpy.ndarray: La imagen resultante después de dibujar la pintura.
    """

    grid_size = x_size
    colores = grid  # Tu lista de 100 elementos

    grid_triple = []

    for x in range(grid_size):
        for y in range(grid_size):
            # Obtiene los primeros 3 elementos de la lista colores
            colors = colores[:3]

            # Elimina los primeros 3 elementos de la lista colores
            colores = colores[3:]

            grid_triple.append((x, y, colors))

    result = img.copy()
    batch_size = 10000

    for h in range(0, len(grid_triple), batch_size):
        for i in range(h, min(h + batch_size, len(grid_triple))):
            x, y, color = grid_triple[i]

            angle = math.degrees(gradient.direction(y, x)) + 90
            # print(color)
            length = int(round(stroke_scale + stroke_scale * math.sqrt(gradient.magnitude(y, x))))
            color_tuple = tuple(color)
            # print(color_tuple)
            array_tupla = np.array(color_tuple, dtype=np.float64)
            cv2.ellipse(result, (y, x), (length, stroke_scale),
                        angle, 0, 360, array_tupla, -1, cv2.LINE_AA)

    return result

def save_image(img, output_path):
    """
    Guarda la imagen resultante en el disco.

    Args:
        img (numpy.ndarray): La imagen a guardar.
        output_path (str): La ruta donde se guardará la imagen.
    """
    print("Saving resulting image to:", output_path)
    cv2.imwrite(output_path, img)


def display_result_image(result_img):
    """
    Muestra la imagen resultante en una ventana emergente.

    Args:
        result_img (numpy.ndarray): La imagen resultante.
    """
    print("Mostrando la imagen resultante...")
    cv2.imshow("result", limit_size(result_img, 1080))
    cv2.waitKey(0)


def pintar(individual):
    global fileName
    ##################################################################################
    pixels = np.array(individual)
    palette_size = 20
    stroke_scale = 0
    gradient_smoothing_radius = 0
    limit_image_size = 0
    img_path = "Results/img/" + fileName + "_enhanced.png"
    ##################################################################################
    img = load_image(img_path, limit_image_size)
    stroke_scale = calculate_stroke_scale(img, stroke_scale)
    gradient_smoothing_radius = calculate_gradient_smoothing_radius(
        img, gradient_smoothing_radius)
    palette = generate_color_palette(img, palette_size)
    gradient = calculate_gradients(img)
    smooth_gradients(gradient, gradient_smoothing_radius)
    cartoonized_img = create_cartoonized_image(img)
    # grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
    # display_result_image(painting_result)
    ##################################################################################
    painting_result = draw_painting(cartoonized_img, pixels, palette, gradient, stroke_scale, img.shape[0], img.shape[1])
    return painting_result
##########################################################################################################



# Function to flatten an image
def flatten_image(image):
    return image.flatten()

# Function to create an initial population of images based on a reference image
def initialize_population(population_size, reference_image, mutation_rate):
    global fileName
    population = []
    for i in range(population_size):
        individual = reference_image.copy()
        individual = mutation(individual, mutation_rate)
        painting_result =- pintar(individual)
        ## TODO:
        img_path = "Results/img/" + fileName
        custom_filename = img_path.rsplit(".", -1)[0] + f"_pointillism_{i}.jpg"
        save_image(painting_result, custom_filename)
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

# Genetic Algorithm main loop
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

def main(originalPath, epath, generations=10, population_size=50, mutation_rate=0.01, ui = None):
    global fileName
    enhancedImage = cv2.imread(epath)
    originalImage = cv2.imread(originalPath)
    fileName = originalPath.rsplit("/", -1)[-1]

    flattened_enhanced_image = flatten_image(enhancedImage)
    #flattened_objective_image = flatten_image(objectiveImage)

    population = initialize_population(population_size, flattened_enhanced_image, mutation_rate)

    # Run the genetic algorithm to enhance the image
    best_image = genetic_algorithm(population, flattened_enhanced_image, generations, mutation_rate, ui.progressBar)

    ######################################################################################################
    painting_result =- pintar(best_image)
    img_path = "Results/img/" + fileName + "_enhancedResult.png"
    ##custom_filename = img_path.rsplit(".", -1)[0] + f"_pointillism_final.jpg"
    save_image(painting_result, img_path)
    ######################################################################################################

    # Reshape the best image to its original shape
    best_image = best_image.reshape(enhancedImage.shape)

    ui.generatedPic_View.setPixmap(QtGui.QPixmap(best_image))

    # Display or save the best-enhanced image
    cv2.imshow("Best Enhanced Image", best_image)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    