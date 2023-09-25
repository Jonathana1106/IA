import cv2
import argparse
import math
import progressbar
from GeneticAlgorithm.pointillism import *

barValue = progressbar.ProgressBar()



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
parser.add_argument('img_path', nargs='?', default="GeneticAlgorithm/img/sansiro.png",
                    help="Ruta de la imagen de entrada")

args = parser.parse_args()

# Definir el nombre del archivo de salida basado en el nombre de la imagen de entrada
res_path = args.img_path.rsplit(".", -1)[0] + "_drawing.jpg"

# Leer la imagen de entrada
img = cv2.imread(args.img_path)

# Limitar el tamaño de la imagen si se especifica un límite
if args.limit_image_size > 0:
    img = limit_size(img, args.limit_image_size)

# Calcular el tamaño de los trazos automáticamente si no se especifica
if args.stroke_scale == 0:
    stroke_scale = int(math.ceil(max(img.shape) / 1000))
    print("Escala de trazo automáticamente seleccionada: %d" % stroke_scale)
else:
    stroke_scale = args.stroke_scale

# Calcular el radio de suavizado del gradiente automáticamente si no se especifica
if args.gradient_smoothing_radius == 0:
    gradient_smoothing_radius = int(round(max(img.shape) / 50))
    print("Radio de suavizado de gradientes automáticamente seleccionado: %d" %
          gradient_smoothing_radius)
else:
    gradient_smoothing_radius = args.gradient_smoothing_radius

# Convertir la imagen a escala de grises para calcular el gradiente
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Calculando paleta de colores...")
# Calcular la paleta de colores
palette = ColorPalette.from_image(img, args.palette_size)

print("Extender paleta de colores...")
# Extender la paleta de colores con ajustes
palette = palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])

# Mostrar la paleta de colores
#cv2.imshow("palette", palette.to_image())
#cv2.waitKey(200)

print("Calculando gradientes...")
# Calcular el campo de vectores del gradiente
gradient = VectorField.from_gradient(gray)

print("Suavizando gradientes...")
# Suavizar el gradiente
gradient.smooth(gradient_smoothing_radius)

print("Generando imagen...")

# Crear una versión "cartonizada" de la imagen para usar como base para la pintura
res = cv2.medianBlur(img, 11)

# Definir una cuadrícula aleatoria de ubicaciones para los trazos de pincel
grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
batch_size = 10000

# Crear una barra de progreso
bar = progressbar.ProgressBar()
for h in bar(range(0, len(grid), batch_size)):
    # Obtener los colores de píxeles en cada punto de la cuadrícula
    pixels = np.array([img[x[0], x[1]]
                      for x in grid[h:min(h + batch_size, len(grid))]])

    # Precalcular las probabilidades para cada color en la paleta
    # Valores más bajos de 'k' significan más aleatoriedad
    color_probabilities = compute_color_probabilities(pixels, palette, k=9)

    for i, (y, x) in enumerate(grid[h:min(h + batch_size, len(grid))]):
        color = color_select(color_probabilities[i], palette)
        angle = math.degrees(gradient.direction(y, x)) + 90
        length = int(round(stroke_scale + stroke_scale *
                     math.sqrt(gradient.magnitude(y, x))))

        # Dibujar el trazo del pincel
        cv2.ellipse(res, (x, y), (length, stroke_scale),
                    angle, 0, 360, color, -1, cv2.LINE_AA)

# Mostrar la imagen resultante y guardarla en un archivo
#cv2.imshow("res", limit_size(res, 1080))
#cv2.imwrite(res_path, res)
#cv2.waitKey(0)
def getbarValue():
    global barValue
    return barValue.finished