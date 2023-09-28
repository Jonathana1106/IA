import bisect
import scipy.spatial
import numpy as np
import random
from .utils import regulate, limit_size, clipped_addition, VectorField, ColorPalette
# from .vector_field import VectorField
# from .color_palette import ColorPalette


def compute_color_probabilities(pixels, palette, k=9):
    """
    Calcula las probabilidades de selección de colores para cada píxel en función de una paleta de colores.

    Args:
        pixels (numpy.ndarray): Matriz de píxeles a los que se les asignarán probabilidades de selección de colores.
        palette (ColorPalette): Paleta de colores a partir de la cual se calcularán las probabilidades.
        k (int, optional): Parámetro de control de la distribución de probabilidades. Por defecto es 9.

    Returns:
        numpy.ndarray: Matriz de probabilidades de selección de colores para cada píxel.
    """
    distances = scipy.spatial.distance.cdist(pixels, palette.colors)
    maxima = np.amax(distances, axis=1)

    distances = maxima[:, None] - distances
    summ = np.sum(distances, 1)
    distances /= summ[:, None]

    distances = np.exp(k*len(palette)*distances)
    summ = np.sum(distances, 1)
    distances /= summ[:, None]

    return np.cumsum(distances, axis=1, dtype=np.float32)


def color_select(probabilities, palette):
    """
    Selecciona un color de la paleta basado en probabilidades.

    Args:
        probabilities (numpy.ndarray): Matriz de probabilidades de selección de colores.
        palette (ColorPalette): Paleta de colores desde la cual se hará la selección.

    Returns:
        numpy.ndarray: El color seleccionado de la paleta.
    """
    r = random.uniform(0, 1)
    i = bisect.bisect_left(probabilities, r)
    return palette[i] if i < len(palette) else palette[-1]


def randomized_grid(h, w, scale):
    """
    Genera una cuadrícula aleatoria para muestreo de píxeles en una imagen.

    Args:
        h (int): Alto de la imagen.
        w (int): Ancho de la imagen.
        scale (int): Tamaño de la cuadrícula.

    Returns:
        list: Una lista de coordenadas (y, x) que definen una cuadrícula aleatoria.
    """
    assert (scale > 0)

    r = scale//2

    grid = []
    for i in range(0, h, scale):
        for j in range(0, w, scale):
            y = random.randint(-r, r) + i
            x = random.randint(-r, r) + j

            grid.append((y % h, x % w))

    random.shuffle(grid)
    return grid
