import sys
from mainUi import Ui_properties_Dialog
from generationUi import Ui_generation_ui
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QT_TR_NOOP as tr

#def uiManager(ui):
#    ui.imageSearch_Button.clicked.connect(lambda: searchFile(ui))

path = ""
iterations = -1
population = -1


## Main Dialog, properties Dialog
def init():
    app = QtWidgets.QApplication(sys.argv)
 
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_properties_Dialog()
 
    ui.setupUi(MainWindow)
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
    global path, iterations, population
    if ui.imagePath_textInput.text() == "":
        ui.imagePath_textInput.setText("No se ha seleccionado una imagen")
    else:
        path = ui.imagePath_textInput.text()
        iterations = ui.iteration_spinBox.value()
        population = ui.population_spinBox.value()
        print("Path: " + path)
        print("Iterations: " + str(iterations))
        print("Population: " + str(population))
        generationUi(ui)


## Generation Dialog
## Close Properties Dialog and Open Generation Dialog
def generationUi(ui):

    dlg = QtWidgets.QDialog()
    gen_ui = Ui_generation_ui()
    gen_ui.setupUi(dlg)

    dlg.show()
    setCOntent(gen_ui)


    dlg.exec_()

    #ui.properties_Dialog.close() 
    #ui.window.close()

## Generation Dialog, Set Content
def setCOntent(ui):
    global path, iterations, population
    ui.path_Label.setText("Path: " + path)
    ui.iterations_Label.setText("Iteraciones: " + str(iterations))
    ui.population_Label.setText("Población: " + str(population))
    ui.originalPic_Viewer.setPixmap(QtGui.QPixmap(path))
    ui.generatedPic_View.setPixmap(QtGui.QPixmap(path))