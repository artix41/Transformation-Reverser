#!/usr/bin/python2.7

import sys
from PyQt4 import QtGui

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

import sklearn.datasets

import numpy as np
import cmath

from train import *

# =========== Constants =============
NB_POINTS = 100

# ============ Classes ===============

class Window(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # General configurations
        self.setWindowTitle("Transformation generator")

        # A combobox to choose the input space

        self.label_input = QtGui.QLabel(self)
        self.label_input.setText("Input space :")
        self.combo_input = QtGui.QComboBox(self)
        self.combo_input.addItem("Mesh")
        self.combo_input.addItem("Gaussian Data Points")
        self.combo_input.addItem("Manual Data Points")

        self.combo_input.activated[str].connect(self.change_input_panel)

        # Input options box (numbers of points, lines, etc.)
        self.input_options_box, self.input_params = self.options_mesh()

        # Generation button
        self.button_generate = QtGui.QPushButton('Generate')
        self.button_generate.clicked.connect(self.change_graph_panel)

        # A combobox to choose the transformations

        self.label_transformation = QtGui.QLabel(self)
        self.label_transformation.setText("Transformation :")
        self.combo_t = QtGui.QComboBox(self)
        self.combo_t.addItem("Translation")
        self.combo_t.addItem("Rotation")
        self.combo_t.addItem("Scaling")
        self.combo_t.addItem("Shear")
        self.combo_t.addItem("Compression")
        self.combo_t.addItem("Polynomial (CM)")
        self.combo_t.addItem("Log")
        self.combo_t.addItem("Matrix product")
        self.combo_t.addItem("Exponential Kernel")
        self.combo_t.activated[str].connect(self.change_options_panel)

        self.options_func = {"Translation": self.options_translation,
                             "Rotation": self.options_rotation,
                             "Scaling": self.options_scaling,
                             "Shear": self.options_shear,
                             "Compression": self.options_compression,
                             "Polynomial (CM)": self.options_polynomial,
                             "Log": self.options_log,
                             "Matrix product": self.options_matrix_product,
                             "Exponential Kernel": self.options_kernel}

        self.input_options_func = {"Mesh": self.options_mesh,
                                   "Gaussian Data Points": self.options_gaussian_points,
                                   "Manual Data Points": self.options_manual_points}

        self.trans_func = {"Translation": self.translation,
                           "Rotation": self.rotation,
                           "Scaling": self.scaling,
                           "Shear": self.shear,
                           "Compression": self.compression,
                           "Polynomial (CM)": self.polynomial,
                           "Log": self.log,
                           "Matrix product": self.matrix_product,
                           "Exponential Kernel": self.kernel}

        # A figure instance to plot on
        self.figure_before = plt.figure()
        self.toolbar_before = NavigationToolbar(self.figure_before.canvas, self)

        # A figure instance to plot on
        self.figure_after = plt.figure()
        self.toolbar_after = NavigationToolbar(self.figure_after.canvas, self)

        # Transformation options (depends on the transformation)
        self.options_box, self.trans_params = self.options_translation()

        # Transform button
        self.button_transform = QtGui.QPushButton('Transform')
        self.button_transform.clicked.connect(self.transform)

        # Undo button
        self.button_undo = QtGui.QPushButton('Undo')
        self.button_undo.clicked.connect(self.undo)

        # Train button
        self.button_train = QtGui.QPushButton('Train')
        self.button_train.clicked.connect(self.train)

        # Neural network options

        self.label_NN = QtGui.QLabel(self)
        self.label_NN.setText("Neural network :")

        self.combo_NN = QtGui.QComboBox(self)
        self.combo_NN.addItem("Corrector")
        self.combo_NN.addItem("Symetric")

        self.options_NN = QtGui.QGridLayout()

        self.check_onlyP = QtGui.QCheckBox('Only positive pairs', self)
        self.check_onlyP.setChecked(False)
        self.nbr_iter_label = QtGui.QLabel(self)
        self.nbr_iter_label.setText("Number of iterations :")
        self.nbr_iter_text = QtGui.QLineEdit(self)
        self.nbr_iter_text.setText('300')

        self.options_NN.addWidget(self.check_onlyP, 1, 0)
        self.options_NN.addWidget(self.nbr_iter_label, 2, 0)
        self.options_NN.addWidget(self.nbr_iter_text, 2, 1)

        self.text_results_nn = QtGui.QLabel(self)
        self.text_results_nn.setText("Loss:\nValid Loss:\n\nWeights:")

        # Set the layout
        self.grid = QtGui.QGridLayout()
        self.grid.setSpacing(10)

        self.grid.addWidget(self.label_input, 0, 0)
        self.grid.addWidget(self.combo_input, 1, 0)
        self.grid.addLayout(self.input_options_box, 2, 0)
        self.grid.addWidget(self.button_generate, 3, 0)
        self.grid.addWidget(self.label_transformation, 0, 1)

        self.grid.addWidget(self.combo_t, 1, 1)
        self.grid.addLayout(self.options_box, 2, 1)
        self.grid.addWidget(self.button_transform, 3, 1)

        self.grid.addWidget(self.label_NN, 0, 2)
        self.grid.addWidget(self.combo_NN, 1, 2)
        self.grid.addLayout(self.options_NN, 2, 2)
        self.grid.addWidget(self.button_train, 3, 2)
        self.grid.addWidget(self.text_results_nn, 5, 2)

        self.grid.addWidget(self.toolbar_before, 4, 0, 1, 2)
        self.grid.addWidget(self.figure_before.canvas, 5, 0, 1, 1)
        self.grid.addWidget(self.toolbar_after, 4, 1, 1, 1)
        self.grid.addWidget(self.figure_after.canvas, 5, 1, 1, 1)
        self.grid.addWidget(self.button_undo, 6, 0, 1, 2)

        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)

        self.setLayout(self.grid)

        # Init the graph
        self.mesh_z = [] # Stack of matrix of complex numbers
        self.target_z = [] # Stack of list of complex numbers
        self.source_z = 0 # List of complex numbers
        self.labels = [] # list of binary numbers
        self.ax_before = self.figure_before.add_subplot(111)
        self.ax_before.tick_params(labelbottom=True, labelleft=True, bottom=True, right=False, top=False, left=True)
        self.ax_before.set_aspect('equal', adjustable='box')
        self.ax_before.axis((-3, 3, -1.5, 1.5))

        self.ax_after = self.figure_after.add_subplot(111)
        self.ax_after.tick_params(labelbottom=True, labelleft=True, bottom=True, right=False, top=False, left=True)
        self.ax_after.set_aspect('equal', adjustable='box')
        self.ax_after.axis((-3, 3, -1.5, 1.5))

    def undo(self):
        if len(self.mesh_z) > 1:
            self.mesh_z.pop()
        if len(self.target_z) > 1:
            self.target_z.pop()

        graph_type = str(self.combo_input.currentText())
        if graph_type == "Mesh":
            self.draw_mesh(ax)
        elif graph_type == "Gaussian Data Points":
            self.draw_data_points(self.figure_before)

    def train(self):
        model = Model(network=str(self.combo_NN.currentText()), nb_iter=int(self.nbr_iter_text.text()))

        print("=================== Preprocess the data ===================")
        X_source = np.array([[z.real, z.imag] for z in self.source_z])
        X_target = np.array([[z.real, z.imag] for z in self.target_z[-1]])
        X_mesh = np.array([[[self.mesh_z[-1][i, j].real, self.mesh_z[-1][i, j].imag]\
                            for j in range(self.mesh_z[-1].shape[1])]
                            for i in range(self.mesh_z[-1].shape[0])])

        if self.check_onlyP.isChecked():
            pairs, sim = createPairsPositive(X_source, X_target, self.labels)
        else:
            pairs, sim = createPairs(X_source, X_target, self.labels)

        print("=================== Training and testing ===================")

        hist = model.fit(pairs[:,0], pairs[:,1], sim)

        # Display the loss and the weights
        loss = hist.history['loss'][-1]
        val_loss = hist.history['val_loss'][-1]
        model.model.save_weights("weights.h5", overwrite=True)
        f = h5py.File('weights.h5', 'r')

        if str(self.combo_NN.currentText()) == "Corrector":
            W_source = np.array([[1,0],[0,1]])
            b_source = np.array([0,0])
            W_target = f['graph']['param_0'].value
            b_target = f['graph']['param_1'].value
        if str(self.combo_NN.currentText()) == "Symetric":
            W_source = f['graph']['param_0'].value
            b_source = f['graph']['param_1'].value
            W_target = f['graph']['param_2'].value
            b_target = f['graph']['param_3'].value

        results = "Loss: " + str(loss) + "\nValid Loss: " + str(val_loss)
        results += '\n\nSource Weights:\n' + str(W_source)
        results += '\n\nSource bias:\n' + str(b_source)
        results += '\n\nTarget Weights:\n' + str(W_target)
        results += '\n\nTarget bias:\n' + str(b_target)

        self.text_results_nn.setText(results)

        feat_target = model.get_feat(X_target)
        feat_source = model.get_source(X_source)

        self.mesh_z.append(np.array([np.vectorize(complex)(model.get_source(X_mesh[i, :])[:,0], model.get_source(X_mesh[i, :])[:,1])\
                            for i in range(X_mesh.shape[0])]))
        self.mesh_z.append(np.array([np.vectorize(complex)(model.get_feat(X_mesh[i, :])[:,0], model.get_feat(X_mesh[i, :])[:,1])\
                            for i in range(X_mesh.shape[0])]))

        print("=================== Display the results ===================")

        ax = self.figure_after.axes[0]
        ax.clear()
        ax.axis((-3, 3, -1.5, 1.5))

        if str(self.combo_NN.currentText()) == "Corrector":
            self.draw_mesh(self.figure_after, {'target':-1, 'source':0})
        elif str(self.combo_NN.currentText()) == "Symetric":
            self.draw_mesh(self.figure_after, {'target':-1, 'source':-2})

        self.mesh_z.pop()
        self.mesh_z.pop()

        colors = ["blue", "red", "green"]
        for i in range(feat_target.shape[0]):
            ax.scatter(feat_target[i][0], feat_target[i][1], color=colors[self.labels[i]])
            ax.scatter(feat_source[i][0], feat_source[i][1], facecolor="#ffffff", s=30, color=colors[self.labels[i]])
            #ax.scatter(self.source_z[i].real, self.source_z[i].imag, facecolor="#ffffff", s=30, color=colors[self.labels[i]])
        self.figure_after.canvas.draw()

    def init_mesh(self):
        try:
            nbr_lines = float(self.input_params["nbr_lines"].text())
        except ValueError:
            nbr_lines = 30
        x = np.linspace(-10, 10, nbr_lines)
        y = np.linspace(-10, 10, nbr_lines)
        mesh_x, mesh_y = np.meshgrid(x, y)
        self.mesh_z = [np.vectorize(complex)(mesh_x, mesh_y)]

        self.draw_mesh(self.figure_before)
        self.figure_before.canvas.draw()

    def draw_mesh(self, figure, index={'target':-1, 'source':0}):
        ax = figure.axes[0]
        ax.clear()
        ax.axis((-3, 3, -1.5, 1.5))
        ax.axhline(y=0, color='k', linewidth=2.0)
        ax.axvline(x=0, color='k', linewidth=2.0)

        ax2 = self.figure_after.axes[0]
        ax2.clear()
        ax2.axis((-3, 3, -1.5, 1.5))
        ax2.axhline(y=0, color='k', linewidth=2.0)
        ax2.axvline(x=0, color='k', linewidth=2.0)

        for i in range(self.mesh_z[-1].shape[0]):
            ax.add_line(Line2D(self.mesh_z[index['target']].real[i, :], self.mesh_z[index['target']].imag[i, :], color="red")) # draw horizontal line
            ax.add_line(Line2D(self.mesh_z[index['source']].real[i, :], self.mesh_z[index['source']].imag[i, :], color="red", linestyle="--")) # draw horizontal line
        for j in range(self.mesh_z[-1].shape[1]):
            ax.add_line(Line2D(self.mesh_z[index['target']].real[:, j], self.mesh_z[index['target']].imag[:, j], color="blue")) # draw vertical line
            ax.add_line(Line2D(self.mesh_z[index['source']].real[:, j], self.mesh_z[index['source']].imag[:, j], color="blue", linestyle="--")) # draw vertical line

        # refresh canvas
        figure.canvas.draw()

    def init_gaussian_data_points(self):
        self.init_mesh()
        try:
            nbr_red_points = int(self.input_params["nbr_red_points"].text())
        except ValueError:
            nbr_red_points = 50
        try:
            nbr_blue_points = int(self.input_params["nbr_blue_points"].text())
        except ValueError:
            nbr_blue_points = 50
        try:
            nbr_green_points = int(self.input_params["nbr_green_points"].text())
        except ValueError:
            nbr_green_points = 50
        try:
            std_blue = float(self.input_params["std_blue"].text())
        except ValueError:
            std_blue = 0.01 #0.01
        try:
            std_red = float(self.input_params["std_red"].text())
        except ValueError:
            std_red = 0.01
        try:
            std_green = float(self.input_params["std_green"].text())
        except ValueError:
            std_green = 0.01
        try:
            mean_blue = [float(self.input_params["mean_blue"][0].text()),
                        float(self.input_params["mean_blue"][1].text())]
        except ValueError:
            mean_blue = [-0.5, -0.5]
        try:
            mean_red = [float(self.input_params["mean_red"][0].text()),
                        float(self.input_params["mean_red"][1].text())]
        except ValueError:
            mean_red = [0.5, -0.5]
        try:
            mean_green = [float(self.input_params["mean_green"][0].text()),
                        float(self.input_params["mean_green"][1].text())]
        except ValueError:
            mean_green = [0, 0.5]

        source_x_blue, source_y_blue = np.random.multivariate_normal(mean_blue, [[std_blue,0],[0,std_blue]], nbr_blue_points).T
        source_x_red, source_y_red = np.random.multivariate_normal(mean_red, [[std_red,0],[0,std_red]], nbr_red_points).T
        source_x_green, source_y_green = np.random.multivariate_normal(mean_green, [[std_green,0],[0,std_green]], nbr_green_points).T
        target_x_blue, target_y_blue = np.random.multivariate_normal(mean_blue, [[std_blue,0],[0,std_blue]], nbr_blue_points).T
        target_x_red, target_y_red = np.random.multivariate_normal(mean_red, [[std_red,0],[0,std_red]], nbr_red_points).T
        target_x_green, target_y_green = np.random.multivariate_normal(mean_green, [[std_green,0],[0,std_green]], nbr_green_points).T

        source_blue_z, source_red_z, source_green_z = [],[],[]
        target_blue_z, target_red_z, target_green_z = [],[],[]

        if len(source_x_blue) != 0:
            source_blue_z = np.vectorize(complex)(source_x_blue, source_y_blue)
            target_blue_z = np.vectorize(complex)(target_x_blue, target_y_blue)
        if len(source_x_red) != 0:
            source_red_z = np.vectorize(complex)(source_x_red, source_y_red)
            target_red_z = np.vectorize(complex)(target_x_red, target_y_red)
        if len(source_x_green) != 0:
            source_green_z = np.vectorize(complex)(source_x_green, source_y_green)
            target_green_z = np.vectorize(complex)(target_x_green, target_y_green)


        self.source_z = np.concatenate([source_blue_z, source_red_z, source_green_z])
        self.target_z = [np.concatenate([target_blue_z, target_red_z, target_green_z])]
        self.labels = [0]*nbr_blue_points + [1]*nbr_red_points + [2]*nbr_green_points
        self.draw_data_points(self.figure_before)

        self.figure_before.canvas.draw()

    def draw_data_points(self, figure):
        ax = figure.axes[0]
        ax.clear()
        ax.axis((-3, 3, -1.5, 1.5))
        ax.axhline(y=0, color='k', linewidth=2.0)
        ax.axvline(x=0, color='k', linewidth=2.0)

        self.draw_mesh(figure)

        colors = ["blue", "red", "green"]
        for i in range(self.source_z.shape[0]):
            ax.scatter(self.target_z[-1][i].real, self.target_z[-1][i].imag, color=colors[self.labels[i]])
            ax.scatter(self.source_z[i].real, self.source_z[i].imag, facecolor="#ffffff", s=30, color=colors[self.labels[i]])

        # Display legend
        fontP = FontProperties()
        fontP.set_size('small')
        full_points = plt.scatter(-1000, -1000, color="black")
        hollow_points = plt.scatter(-1000, -1000, facecolor="#ffffff", s=30, color="black")
        ax.legend([hollow_points, full_points], ["Source", "Target"], bbox_to_anchor=(1, 1), loc=2, prop= fontP)

        self.figure_before.canvas.draw()

    def change_graph_panel(self):
        """ Results of button 'Generate : choose the good function to init the graph"""

        graph_type = str(self.combo_input.currentText())
        if graph_type == "Mesh":
            self.init_mesh()
        elif graph_type == "Gaussian Data Points":
            self.init_gaussian_data_points()
        elif graph_type == "Manual Data Points":
            self.init_mesh()

    def change_input_panel(self):
        """ Change the panel with the options of initialization, depending on
        whether we have choosen Mesh, Gaussian, Manual, etc.
        The panel contains options such as 'Number of mesh', 'Number of points'..."""

        # Remove the old panel
        while self.input_options_box.count():
            item = self.input_options_box.takeAt(0)
            widget = item.widget()
            widget.deleteLater()

        # Create the new panel
        self.input_options_box, self.input_params = self.input_options_func[str(self.combo_input.currentText())]()
        self.grid.addLayout(self.input_options_box, 2, 0)
        self.setLayout(self.grid)
        self.change_options_panel()

    def change_options_panel(self):
        # Remove the old panel
        while self.options_box.count():
            item = self.options_box.takeAt(0)
            widget = item.widget()
            widget.deleteLater()

        # Create the new panel
        self.options_box, self.trans_params = self.options_func[str(self.combo_t.currentText())]()
        self.grid.addLayout(self.options_box, 2, 1)
        self.setLayout(self.grid)

    def transform(self):
        self.ax_before.clear()
        self.options_func[str(self.combo_t.currentText())]()
        graph_type = str(self.combo_input.currentText())

        self.mesh_z.append(np.array([[self.trans_func[str(self.combo_t.currentText())](self.mesh_z[-1][i, j], "mesh")\
                            for j in range(self.mesh_z[-1].shape[1])]
                            for i in range(self.mesh_z[-1].shape[0])]))

        if graph_type == "Mesh":
            self.draw_mesh(self.figure_before)

        if graph_type == "Gaussian Data Points":
            self.target_z.append(np.array([self.trans_func[str(self.combo_t.currentText())](self.target_z[-1][i], "target")\
                            for i in range(self.target_z[-1].shape[0])]))
            self.draw_data_points(self.figure_before)

    def options_mesh(self):
        grid = QtGui.QGridLayout()

        nbr_lines_label = QtGui.QLabel("Number of mesh lines (default : 100) :", self)
        nbr_lines_text = QtGui.QLineEdit(self)

        grid.addWidget(nbr_lines_label, 1, 0)
        grid.addWidget(nbr_lines_text, 1, 1)

        params = {"nbr_lines": nbr_lines_text}
        return grid, params

    def options_gaussian_points(self):
        grid = QtGui.QGridLayout()

        nbr_lines_label = QtGui.QLabel("Number of mesh lines:", self)
        nbr_lines_text = QtGui.QLineEdit(self)
        nbr_lines_text.setText('30')

        nbr_points_label = QtGui.QLabel("Number of points (red/blue/green):", self)
        nbr_red_points_text = QtGui.QLineEdit(self)
        nbr_red_points_text.setText('50')

        nbr_blue_points_text = QtGui.QLineEdit(self)
        nbr_blue_points_text.setText('50')

        nbr_green_points_text = QtGui.QLineEdit(self)
        nbr_green_points_text.setText('50')

        std_blue_label = QtGui.QLabel("STD for blue:", self)
        std_blue_text = QtGui.QLineEdit(self)
        std_blue_text.setText('0.01')

        std_red_label = QtGui.QLabel("STD for red:", self)
        std_red_text = QtGui.QLineEdit(self)
        std_red_text.setText('0.01')

        std_green_label = QtGui.QLabel("STD for green:", self)
        std_green_text = QtGui.QLineEdit(self)
        std_green_text.setText('0.01')

        mean_blue_label = QtGui.QLabel("Mean for blue:", self)
        mean_blue_x_text = QtGui.QLineEdit(self)
        mean_blue_x_text.setText('-0.5')
        mean_blue_y_text = QtGui.QLineEdit(self)
        mean_blue_y_text.setText('-0.5')

        mean_red_label = QtGui.QLabel("Mean for red:", self)
        mean_red_x_text = QtGui.QLineEdit(self)
        mean_red_x_text.setText('0.5')
        mean_red_y_text = QtGui.QLineEdit(self)
        mean_red_y_text.setText('-0.5')

        mean_green_label = QtGui.QLabel("Mean for green:", self)
        mean_green_x_text = QtGui.QLineEdit(self)
        mean_green_x_text.setText('1.2')
        mean_green_y_text = QtGui.QLineEdit(self)
        mean_green_y_text.setText('0.5')

        mean_green_label = QtGui.QLabel("Mean for green:", self)
        mean_green_x_text = QtGui.QLineEdit(self)
        mean_green_x_text.setText('0')
        mean_green_y_text = QtGui.QLineEdit(self)
        mean_green_y_text.setText('0.5')

        std_label = QtGui.QLabel("STD (blue/red/green):", self)
        std_blue_text = QtGui.QLineEdit(self)
        std_blue_text.setText('0.01')
        std_red_text = QtGui.QLineEdit(self)
        std_red_text.setText('0.01')
        std_green_text = QtGui.QLineEdit(self)
        std_green_text.setText('0.01')


        grid.addWidget(nbr_lines_label, 1, 0)
        grid.addWidget(nbr_lines_text, 1, 1)
        grid.addWidget(nbr_points_label, 2, 0)
        grid.addWidget(nbr_red_points_text, 2, 1)
        grid.addWidget(nbr_blue_points_text, 2, 2)
        grid.addWidget(nbr_green_points_text, 2, 3)
        grid.addWidget(mean_blue_label, 3, 0)
        grid.addWidget(mean_blue_x_text, 3, 1)
        grid.addWidget(mean_blue_y_text, 3, 2)
        grid.addWidget(mean_red_label, 4, 0)
        grid.addWidget(mean_red_x_text, 4, 1)
        grid.addWidget(mean_red_y_text, 4, 2)
        grid.addWidget(mean_green_label, 5, 0)
        grid.addWidget(mean_green_x_text, 5, 1)
        grid.addWidget(mean_green_y_text, 5, 2)
        grid.addWidget(std_label, 6, 0)
        grid.addWidget(std_blue_text, 6, 1)
        grid.addWidget(std_red_text, 6, 2)
        grid.addWidget(std_green_text, 6, 3)

        params = {"nbr_lines": nbr_lines_text, "nbr_red_points": nbr_red_points_text,
                "nbr_blue_points": nbr_blue_points_text, "nbr_green_points": nbr_green_points_text,
                "std_blue": std_blue_text, "std_red": std_red_text,"std_green": std_green_text,
                "mean_blue": [mean_blue_x_text, mean_blue_y_text],
                "mean_red": [mean_red_x_text, mean_red_y_text],
                "mean_green": [mean_green_x_text, mean_green_y_text]}

        return grid, params

    def options_manual_points(self):
        grid = QtGui.QGridLayout()

        color_label = QtGui.QLabel("Color :", self)
        color_radio = QtGui.QButtonGroup()

        red_radio = QtGui.QRadioButton("red")
        red_radio.setChecked(True)
        color_radio.addButton(red_radio)
        blue_radio = QtGui.QRadioButton("blue")
        color_radio.addButton(blue_radio)

        grid.addWidget(color_label, 0, 0)
        grid.addWidget(red_radio, 0, 1)
        grid.addWidget(blue_radio, 1, 1)
        grid.horizontalSpacing()

        color_text = "red"
        if blue_radio.isChecked():
            color_text = "blue"

        params = {"color": color_text}
        return grid, params

    def draw_manual_point(self):
        pass

    def options_translation(self):
        grid = QtGui.QGridLayout()

        vector_label = QtGui.QLabel("Vector :", self)
        vector_x_text = QtGui.QLineEdit(self)
        vector_x_text.setText('-2')
        vector_y_text = QtGui.QLineEdit(self)
        vector_y_text.setText('0')

        grid.addWidget(vector_label, 1, 0)
        grid.addWidget(vector_x_text, 1, 1)
        grid.addWidget(vector_y_text, 1, 2)

        params = {"vector": [vector_x_text, vector_y_text]}
        return grid, params

    def translation(self, z, data_type):
        try:
            vector = [float(self.trans_params["vector"][0].text()),
                        float(self.trans_params["vector"][1].text())]
        except ValueError:
            vector = [0, 0]

        return z + complex(vector[0], vector[1])

    def options_rotation(self):
        grid = QtGui.QGridLayout()

        angle_label = QtGui.QLabel("Angle :", self)
        angle_text = QtGui.QLineEdit(self)

        grid.addWidget(angle_label, 1, 0)
        grid.addWidget(angle_text, 1, 1)

        params = {"angle": angle_text}
        return grid, params

    def rotation(self, z, data_type):
        try:
            theta = float(self.trans_params["angle"].text())
        except ValueError:
            theta = 0

        return np.exp(complex(0, theta))*z

    def options_scaling(self):
        grid = QtGui.QGridLayout()

        scaling_label = QtGui.QLabel("Factor :", self)
        scaling_text = QtGui.QLineEdit(self)

        grid.addWidget(scaling_label, 1, 0)
        grid.addWidget(scaling_text, 1, 1)

        params = {"factor": scaling_text}
        return grid, params

    def scaling(self, z, data_type):
        try:
            factor = float(self.trans_params["factor"].text())
        except ValueError:
            factor = 1
        return z*factor

    def options_shear(self):
        grid = QtGui.QGridLayout()

        scaling_label = QtGui.QLabel("Coeff :", self)
        scaling_text = QtGui.QLineEdit(self)

        grid.addWidget(scaling_label, 1, 0)
        grid.addWidget(scaling_text, 1, 1)

        params = {"coeff": scaling_text}
        return grid, params

    def shear(self, z, data_type):
        try:
            coeff = float(self.trans_params["coeff"].text())
        except ValueError:
            coeff = 0
        return complex(z.real+coeff*z.imag, z.imag)

    def options_compression(self):
        grid = QtGui.QGridLayout()

        x_compression_label = QtGui.QLabel("Coeff along x :", self)
        x_compression_text = QtGui.QLineEdit(self)
        y_compression_label = QtGui.QLabel("Coeff along y :", self)
        y_compression_text = QtGui.QLineEdit(self)

        grid.addWidget(x_compression_label, 1, 0)
        grid.addWidget(x_compression_text, 1, 1)
        grid.addWidget(y_compression_label, 2, 0)
        grid.addWidget(y_compression_text, 2, 1)

        params = {"x_coeff": x_compression_text, "y_coeff": y_compression_text}
        return grid, params

    def compression(self, z, data_type):
        try:
            x_coeff = float(self.trans_params["x_coeff"].text())
        except ValueError:
            x_coeff = 1
        try:
            y_coeff = float(self.trans_params["y_coeff"].text())
        except ValueError:
            y_coeff = 1

        return complex(*tuple(np.multiply([z.real, z.imag], [x_coeff, y_coeff])))

    def options_polynomial(self):
        grid = QtGui.QGridLayout()

        degree_label = QtGui.QLabel("Degree (z -> z^k) :", self)
        degree_text = QtGui.QLineEdit(self)

        grid.addWidget(degree_label, 1, 0)
        grid.addWidget(degree_text, 1, 1)

        params = {"degree": degree_text}
        return grid, params

    def polynomial(self, z, data_type):
        try:
            degree = float(self.trans_params["degree"].text())
        except ValueError:
            degree = 1
        return z**degree

    def options_log(self):
        grid = QtGui.QGridLayout()

        coeff_label = QtGui.QLabel("Coeff (z -> k*log(z)) :", self)
        coeff_text = QtGui.QLineEdit(self)

        grid.addWidget(coeff_label, 1, 0)
        grid.addWidget(coeff_text, 1, 1)

        params = {"coeff": coeff_text}
        return grid, params

    def log(self, z, data_type):
        try:
            coeff = float(self.trans_params["coeff"].text())
        except ValueError:
            coeff = 1
        return coeff*np.log(z)

    def options_matrix_product(self):
        grid = QtGui.QGridLayout()

        matrix_label = QtGui.QLabel("Matrix :", self)

        x11_text = QtGui.QLineEdit(self)
        #x11_text.setFixedWidth(10)
        x11_text.setText('1')
        x12_text = QtGui.QLineEdit(self)
        #x12_text.setFixedWidth(10)
        x12_text.setText('0')
        x21_text = QtGui.QLineEdit(self)
        #x21_text.setFixedWidth(10)
        x21_text.setText('0')
        x22_text = QtGui.QLineEdit(self)
        #x22_text.setFixedWidth(10)
        x22_text.setText('1')


        grid.addWidget(matrix_label, 1, 0)
        grid.addWidget(x11_text, 1, 1)
        grid.addWidget(x12_text, 1, 2)
        grid.addWidget(x21_text, 2, 1)
        grid.addWidget(x22_text, 2, 2)

        params = {"x11": x11_text, "x12": x12_text, "x21": x21_text, "x22":x22_text}
        return grid, params

    def matrix_product(self, z, data_type):
        try:
            x11 = float(self.trans_params["x11"].text())
        except ValueError:
            x11 = 1
        try:
            x12 = float(self.trans_params["x12"].text())
        except ValueError:
            x12 = 0
        try:
            x21 = float(self.trans_params["x21"].text())
        except ValueError:
            x21 = 0
        try:
            x22 = float(self.trans_params["x22"].text())
        except ValueError:
            x22 = 1

        M = np.array([[x11, x12], [x21, x22]])

        return complex(*tuple(np.dot(M, np.array([z.real, z.imag]))))

    def options_kernel(self):
        nbr_noeuds = 4
        grid = QtGui.QGridLayout()

        gamma_label = QtGui.QLabel("Gamma :", self)
        gamma_text = QtGui.QLineEdit(self)
        gamma_text.setText('1')
        noeud_label = [QtGui.QLabel("Noeud " + str(i) + " (alpha, beta, x, y):", self) for i in range(nbr_noeuds)]
        alpha_text = [0]*nbr_noeuds
        beta_text = [0]*nbr_noeuds
        x_text = [0]*nbr_noeuds
        y_text = [0]*nbr_noeuds

        for i in range(nbr_noeuds):
            alpha_text[i] = QtGui.QLineEdit(self)
            alpha_text[i].setText('1')
            beta_text[i] = QtGui.QLineEdit(self)
            beta_text[i].setText('1')
            x_text[i] = QtGui.QLineEdit(self)
            x_text[i].setText('0')
            y_text[i] = QtGui.QLineEdit(self)
            y_text[i].setText('0')


        grid.addWidget(gamma_label, 1, 0)
        grid.addWidget(gamma_text, 1, 1)
        for i in range(nbr_noeuds):
            grid.addWidget(noeud_label[i], 2+i, 0)
            grid.addWidget(alpha_text[i], 2+i, 1)
            grid.addWidget(beta_text[i], 2+i, 2)
            grid.addWidget(x_text[i], 2+i, 3)
            grid.addWidget(y_text[i], 2+i, 4)

        params = {"gamma": gamma_text, "alpha": alpha_text, "beta": beta_text, "x": x_text, "y": y_text}
        return grid, params

    def kernel (self, z, data_type):
        nbr_noeuds = 4
        try:
            gamma = float(self.trans_params["gamma"].text())
        except ValueError:
            gamma = 1
        try:
            alpha = [float(self.trans_params["alpha"][i].text()) for i in range(nbr_noeuds)]
        except ValueError:
            alpha = [0 for i in range(nbr_noeuds)]
        try:
            beta = [float(self.trans_params["beta"][i].text()) for i in range(nbr_noeuds)]
        except ValueError:
            beta = [0 for i in range(nbr_noeuds)]
        try:
            x = [float(self.trans_params["x"][i].text()) for i in range(nbr_noeuds)]
        except ValueError:
            x = [0 for i in range(nbr_noeuds)]
        try:
            y = [float(self.trans_params["y"][i].text()) for i in range(nbr_noeuds)]
        except ValueError:
            y = [0 for i in range(nbr_noeuds)]

        vect_z = np.array([z.real, z.imag])
        vect_x = [np.array([x[i], y[i]]) for i in range(nbr_noeuds)]
        X = np.sum([alpha[i] * np.exp(-gamma * np.linalg.norm(vect_z - vect_x[i])**2) for i in range(nbr_noeuds)])
        Y = np.sum([beta[i] * np.exp(-gamma * (vect_z - vect_x[i])**2) for i in range(nbr_noeuds)])
        return complex(*tuple(np.array([X, Y])))


# =========== Main Program ==============

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
