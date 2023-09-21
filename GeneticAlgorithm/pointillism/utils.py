import cv2
import numpy as np
import math
from sklearn.cluster import KMeans


class ColorPalette:
    """
    Clase para representar una paleta de colores.

    Args:
        colors (numpy.ndarray): Matriz de colores en formato RGB.
        base_len (int, optional): Longitud base de la paleta. Por defecto, es la longitud de 'colors'.
    """

    def __init__(self, colors, base_len=0):
        self.colors = colors
        self.base_len = base_len if base_len > 0 else len(colors)

    @staticmethod
    def from_image(img, n, max_img_size=200):
        """
        Crea una paleta de colores a partir de una imagen utilizando el algoritmo de K-Means.

        Args:
            img (numpy.ndarray): La imagen de la cual se extraerán los colores.
            n (int): Número de colores a extraer.
            max_img_size (int, optional): Tamaño máximo de la imagen para acelerar K-Means. Por defecto es 200.

        Returns:
            ColorPalette: Una instancia de ColorPalette basada en los colores extraídos de la imagen.
        """
        # Escala la imagen para acelerar k-means
        img = limit_size(img, max_img_size)

        clt = KMeans(n_clusters=n)
        clt.fit(img.reshape(-1, 3))

        return ColorPalette(clt.cluster_centers_)

    def extend(self, extensions):
        """
        Extiende la paleta de colores con colores regulados.

        Args:
            extensions (list): Lista de ajustes de colores para extender la paleta.

        Returns:
            ColorPalette: Una nueva instancia de ColorPalette con la paleta extendida.
        """
        extension = [regulate(self.colors.reshape((1, len(self.colors), 3)).astype(
            np.uint8), *x).reshape((-1, 3)) for x in extensions]

        return ColorPalette(np.vstack([self.colors.reshape((-1, 3))] + extension), self.base_len)

    def to_image(self):
        """
        Convierte la paleta de colores en una imagen representativa.

        Returns:
            numpy.ndarray: Una imagen que representa la paleta de colores.
        """
        cols = self.base_len
        rows = int(math.ceil(len(self.colors) / cols))

        res = np.zeros((rows * 80, cols * 80, 3), dtype=np.uint8)
        for y in range(rows):
            for x in range(cols):
                if y * cols + x < len(self.colors):
                    color = [int(c) for c in self.colors[y * cols + x]]
                    cv2.rectangle(res, (x * 80, y * 80),
                                  (x * 80 + 80, y * 80 + 80), color, -1)

        return res

    def __len__(self):
        return len(self.colors)

    def __getitem__(self, item):
        return self.colors[item]


class VectorField:
    """
    Clase para representar un campo de vectores 2D.

    Args:
        fieldx (numpy.ndarray): Componente x del campo de vectores.
        fieldy (numpy.ndarray): Componente y del campo de vectores.
    """

    def __init__(self, fieldx, fieldy):
        self.fieldx = fieldx
        self.fieldy = fieldy

    @staticmethod
    def from_gradient(gray):
        """
        Crea un campo de vectores a partir del gradiente de una imagen en escala de grises.

        Args:
            gray (numpy.ndarray): Imagen en escala de grises.

        Returns:
            VectorField: Una instancia de VectorField basada en el gradiente de la imagen.
        """
        fieldx = cv2.Scharr(gray, cv2.CV_32F, 1, 0) / 15.36
        fieldy = cv2.Scharr(gray, cv2.CV_32F, 0, 1) / 15.36

        return VectorField(fieldx, fieldy)

    def get_magnitude_image(self):
        """
        Calcula y devuelve una imagen de magnitud basada en el campo de vectores.

        Returns:
            numpy.ndarray: Una imagen que representa la magnitud del campo de vectores.
        """
        res = np.sqrt(self.fieldx**2 + self.fieldy**2)
        return (res * 255/np.max(res)).astype(np.uint8)

    def smooth(self, radius, iterations=1):
        """
        Aplica un suavizado Gaussiano al campo de vectores.

        Args:
            radius (int): Radio del kernel de suavizado.
            iterations (int, optional): Número de veces que se aplica el suavizado. Por defecto es 1.
        """
        s = 2*radius + 1
        for _ in range(iterations):
            self.fieldx = cv2.GaussianBlur(self.fieldx, (s, s), 0)
            self.fieldy = cv2.GaussianBlur(self.fieldy, (s, s), 0)

    def direction(self, i, j):
        """
        Obtiene la dirección del vector en la posición (i, j).

        Args:
            i (int): Índice de fila.
            j (int): Índice de columna.

        Returns:
            float: Ángulo en radianes que representa la dirección del vector.
        """
        return math.atan2(self.fieldy[i, j], self.fieldx[i, j])

    def magnitude(self, i, j):
        """
        Obtiene la magnitud del vector en la posición (i, j).

        Args:
            i (int): Índice de fila.
            j (int): Índice de columna.

        Returns:
            float: Magnitud del vector en la posición dada.
        """
        return math.hypot(self.fieldx[i, j], self.fieldy[i, j])


def limit_size(img, max_x, max_y=0):
    """
    Limita el tamaño de una imagen a un tamaño máximo dado, manteniendo la relación de aspecto.

    Args:
        img (numpy.ndarray): La imagen que se va a redimensionar.
        max_x (int): Ancho máximo deseado.
        max_y (int, optional): Alto máximo deseado. Si no se proporciona, se usa max_x para mantener la relación de aspecto.

    Returns:
        numpy.ndarray: La imagen redimensionada con las dimensiones limitadas por max_x y max_y.
    """
    if max_x == 0:
        return img

    if max_y == 0:
        max_y = max_x

    ratio = min(1.0, float(max_x) / img.shape[1], float(max_y) / img.shape[0])

    if ratio != 1.0:
        shape = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
        return cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    else:
        return img


def clipped_addition(img, x, _max=255, _min=0):
    """
    Realiza una adición de valor 'x' a cada píxel de una imagen, asegurándose de que los valores estén dentro de un rango dado.

    Args:
        img (numpy.ndarray): La imagen a la que se le aplicará la adición.
        x (int): El valor a añadir.
        _max (int, optional): Valor máximo permitido en la imagen resultante. Por defecto es 255.
        _min (int, optional): Valor mínimo permitido en la imagen resultante. Por defecto es 0.
    """
    if x > 0:
        mask = img > (_max - x)
        img += x
        np.putmask(img, mask, _max)
    if x < 0:
        mask = img < (_min - x)
        img += x
        np.putmask(img, mask, _min)


def regulate(img, hue=0, saturation=0, luminosity=0):
    """
    Regula los componentes de color de una imagen en formato BGR ajustando el tono, la saturación y la luminosidad.

    Args:
        img (numpy.ndarray): La imagen de entrada en formato BGR.
        hue (int, optional): Ajuste de tono. Puede ser positivo o negativo. Por defecto es 0.
        saturation (int, optional): Ajuste de saturación. Puede ser positivo o negativo. Por defecto es 0.
        luminosity (int, optional): Ajuste de luminosidad. Puede ser positivo o negativo. Por defecto es 0.

    Returns:
        numpy.ndarray: La imagen regulada con los ajustes de tono, saturación y luminosidad aplicados.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    if hue < 0:
        hue = 255 + hue
    hsv[:, :, 0] += hue
    clipped_addition(hsv[:, :, 1], saturation)
    clipped_addition(hsv[:, :, 2], luminosity)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)
