import os
import cv2

#Parámetros de referencia

#imageRoute es la ruta de la imagen, ejemplo: imageRoute = "C:/Users/Renzo/Downloads/test.jpg"
#resize hace downscale de la imagen a un factor de 0.8
#median, gauss son True/False. Son filtros para reducir el ruido, después de probar median da mejores resultados en general
#Ajuste de contraste, ejemplo: alpha = 1
#Ajuste de Brillo, ejemplo: beta = 15
#esquemaColor nos permite realizar un cambio de color a la imagen en distintos esquemas

def preProcessImage(imageRoute, resize, median, gauss, alpha, beta, esquemaColor=None):
    esquemaColores = {
    "BGR2GRAY": cv2.COLOR_BGR2GRAY,
    "BGR2RGB": cv2.COLOR_BGR2RGB,
    "BGR2HSV": cv2.COLOR_BGR2HSV,
    "BGR2Lab": cv2.COLOR_BGR2Lab,
    "BGR2YUV": cv2.COLOR_BGR2YUV,
    "BGR2XYZ": cv2.COLOR_BGR2XYZ,
    "BGR2HLS": cv2.COLOR_BGR2HLS,
    "BGR2Luv": cv2.COLOR_BGR2Luv,
    "BGR2YCrCb": cv2.COLOR_BGR2YCrCb,
    "BGR2HLS_FULL": cv2.COLOR_BGR2HLS_FULL,
    # Agregar más esquemas de color si es necesario
    }
    
    print("values from preProcessImage:")
    print("Path: " + imageRoute)
    print("Resize: " + str(resize))
    print("Median: " + str(median))
    print("Gauss: " + str(gauss))
    print("Contrast: " + str(alpha))
    print("Brightness: " + str(beta))
    print("Color: " + str(esquemaColor))

    #Carga una imagen
    originalImage = cv2.imread(imageRoute)

    #Comprueba si la imagen se ha cargado correctamente
    if originalImage is None:
        print("No se puede cargar la imagen.")
    else:

        enhancedImage = originalImage.copy()
        #medianBlurredImage = originalImage.copy()
        #gaussBlurredImage = originalImage.copy()
        
        #Muestra la imagen original y las imágenes con los filtros aplicados
        cv2.imshow("Imagen Original", originalImage)
        
        #Redimensionar la imagen "Downscale", la hace pequeña
        if resize:
            #Especifica las nuevas dimensiones o el factor de escala de un 0.8 al tamaño original
            newWidth = 576
            newHeight = 714
            originalImage = cv2.resize(originalImage, (newWidth, newHeight))

        #Aplica filtro de mediana para reducir el ruido
        if median:
            medianBlurredImage = cv2.medianBlur(originalImage, 5)
            enhancedImage = cv2.convertScaleAbs(medianBlurredImage, alpha=alpha, beta=beta)
            cv2.imshow("Imagen con Filtro de Mediana", medianBlurredImage)
        else:
            medianBlurredImage = []
        
         #Aplica filtro Gaussiano para reducir el ruido
        if gauss:
            gaussBlurredImage = cv2.GaussianBlur(originalImage, (5, 5), 0)
            enhancedImage = cv2.convertScaleAbs(gaussBlurredImage, alpha=alpha, beta=beta)
            cv2.imshow("Imagen con Filtro Gaussiano", gaussBlurredImage)
        else:
            gaussBlurredImage = []

        # Aplica el esquema de color si se especifica
        if esquemaColor is not None:
            if esquemaColor in esquemaColores:
                enhancedImage = cv2.cvtColor(enhancedImage, esquemaColores[esquemaColor])
        
        #Quitar estas lineas para que no muestre las ventanas y solo retorne

        cv2.imwrite("imgs/Imagen Mejorada.jpg", enhancedImage)
        return {"original" : originalImage, "median": medianBlurredImage, "gauss" : gaussBlurredImage, "enhanced" : enhancedImage}

#Ruta de la imagen
#Ejemplo:
#imageRoute = "GeneticAlgorithm\img\milano.jpg"


#Sin usar esquema de color
#preProcessImage(imageRoute, True, True, False, 1, 15, None)

#Usando esquema de color válido
#preProcessImage(imageRoute, True, True, False, 1, 15, "BGR2XYZ")