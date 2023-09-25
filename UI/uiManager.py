import sys

from UI.mainUi import Ui_properties_Dialog
from UI.generationUi import Ui_generation_ui
from ImageProcesing.preprocessing import preProcessImage
from GeneticAlgorithm.modular_main import main as objectiveImgMain
from ImageProcesing.geneticAlgorithm import main as geneticAlgorithmMain
from GeneticAlgorithm.main import getbarValue

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QT_TR_NOOP as tr

#def uiManager(ui):
#    ui.imageSearch_Button.clicked.connect(lambda: searchFile(ui))

path = ""
filterPath = ""
objPath = ""
iterations = -1
population = -1
gauss = False
median = False
resize = False
gama = ""
gama_list =  ["BGR2GRAY", "BGR2RGB", "BGR2HSV", "BGR2Lab", "BGR2YUV", "BGR2XYZ", "BGR2HLS", "BGR2Luv", "BGR2YCrCb", "BGR2HLS_FULL"]

## Main Dialog, properties Dialog
def init():
    app = QtWidgets.QApplication(sys.argv)
 
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_properties_Dialog()
 
    ui.setupUi(MainWindow)
    ui.gama_combobox.addItems(gama_list)
    ui.gama_combobox.setCurrentIndex(0)
    ui.population_spinBox.setValue(100)
    ui.iteration_spinBox.setValue(50)
    MainWindow.show()
    ui.imageSearch_Button.clicked.connect(lambda: searchFile(ui))
    ui.generate_Button.clicked.connect(lambda: getValues(ui))  
    
    sys.exit(app.exec_())

## Properties Dialog, Image Selection
def searchFile(ui):

    dlg = QtWidgets.QFileDialog()
    dlg.setFileMode(QtWidgets.QFileDialog.AnyFile)
    dlg.setNameFilter(tr("Images (*.png *.xpm *.jpg)"))
    filenames = QtCore.QStringListModel()

    if dlg.exec_():
        filenames = dlg.selectedFiles()
        ui.imagePath_textInput.setText(filenames[0])

        ui.imagePreview_View.setPixmap(QtGui.QPixmap(filenames[0]))

## Properties Dialog, Generate Button
def getValues(ui):
    global path, iterations, population, gauss, median, resize, gama
    if ui.imagePath_textInput.text() == "":
        ui.imagePath_textInput.setText("No se ha seleccionado una imagen")
    else:
        path = ui.imagePath_textInput.text()
        iterations = ui.iteration_spinBox.value()
        population = ui.population_spinBox.value()
        gama = ui.gama_combobox.currentText()
        gauss = ui.gauss_checkb.isChecked()
        median = ui.median_checkb.isChecked()
        resize = ui.resize_checkb.isChecked()

        generationUi(ui)


## Generation Dialog
## Close Properties Dialog and Open Generation Dialog
def generationUi(ui):
    global path, iterations, population, gauss, median, resize, gama, filterPath, objPath

    dlg = QtWidgets.QDialog()
    gen_ui = Ui_generation_ui()
    gen_ui.setupUi(dlg)

    dlg.show()
    imageProcessing()
    setCOntent(gen_ui)

    print("Values from ui:")
    
    print("Path: " + path)
    print("Iterations: " + str(iterations))
    print("Population: " + str(population))
    print("Gama: " + gama)
    print("Gauss: " + str(gauss))
    print("Median: " + str(median))
    print("Resize: " + str(resize))
    print("Filter Path: " + filterPath)
    print("Obj Path: " + objPath)
    
    dlg.exec_()

    #ui.properties_Dialog.close() 
    #ui.window.close()

## Generation Dialog, Set Content
def setCOntent(ui):
    global path, iterations, population, gauss, median, resize, gama, filterPath, objPath
    ui.path_Label.setText("Path: " + path)
    ui.iterations_Label.setText("Iteraciones: " + str(iterations))
    ui.population_Label.setText("Población: " + str(population))
    ui.originalPic_Viewer.setPixmap(QtGui.QPixmap(filterPath))
    ui.generatedPic_View.setPixmap(QtGui.QPixmap(objPath))

    ui.progressBar.setMaximum(iterations)
    ui.progressBar.setValue(0)
    ui.progressBar.setFormat("0%")

    #geneticAlgorithmMain(epath = filterPath, objPath = objPath, generations=iterations, population_size=population, progressBar = ui.progressBar)

def imageProcessing():
    global path, iterations, population, gauss, median, resize, gama, filterPath, objPath
    preProcessImageDicc = preProcessImage(path, resize, median, gauss, 1, 15, gama)
    filterPath = preProcessImageDicc["epath"]
    objPath = objectiveImgMain(img_path = filterPath)
