import cv2
import argparse
import math
import progressbar
from pointillism import *


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


def draw_painting(img, grid, palette, gradient, stroke_scale):
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
    result = img.copy()
    batch_size = 10000

    bar = progressbar.ProgressBar()
    for h in bar(range(0, len(grid), batch_size)):
        pixels = np.array([img[x[0], x[1]]
                          for x in grid[h:min(h + batch_size, len(grid))]])

        color_probabilities = compute_color_probabilities(pixels, palette, k=9)

        for i, (y, x) in enumerate(grid[h:min(h + batch_size, len(grid))]):
            color = color_select(color_probabilities[i], palette)
            angle = math.degrees(gradient.direction(y, x)) + 90
            length = int(round(stroke_scale + stroke_scale *
                         math.sqrt(gradient.magnitude(y, x))))

            cv2.ellipse(result, (x, y), (length, stroke_scale),
                        angle, 0, 360, color, -1, cv2.LINE_AA)

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


if __name__ == "__main__":
    # Crear un parser de argumentos para la línea de comandos
    parser = argparse.ArgumentParser(
        description='Generación de Imágenes de Puntillismo')
    parser.add_argument('--palette-size', default=20, type=int,
                        help="Número de colores en la paleta base")
    parser.add_argument('--stroke-scale', default=0, type=int,
                        help="Escala de los trazos del pincel (0 = automático)")
    parser.add_argument('--gradient-smoothing-radius', default=0, type=int,
                        help="Radio del filtro de suavizado aplicado al gradiente (0 = automático)")
    parser.add_argument('--limit-image-size', default=0, type=int,
                        help="Limitar el tamaño de la imagen (0 = sin límites)")
    parser.add_argument('img_path', nargs='?', default="img/sansiro.png",
                        help="Ruta de la imagen de entrada")

    args = parser.parse_args()

    img = load_image(args.img_path, args.limit_image_size)
    stroke_scale = calculate_stroke_scale(img, args.stroke_scale)
    gradient_smoothing_radius = calculate_gradient_smoothing_radius(
        img, args.gradient_smoothing_radius)
    palette = generate_color_palette(img, args.palette_size)
    gradient = calculate_gradients(img)
    smooth_gradients(gradient, gradient_smoothing_radius)
    cartoonized_img = create_cartoonized_image(img)
    grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
    painting_result = draw_painting(
        cartoonized_img, grid, palette, gradient, stroke_scale)
    save_image(painting_result, args.img_path.rsplit(
        ".", -1)[0] + "_pointillism.jpg")
    display_result_image(painting_result)
