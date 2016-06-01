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
        self.combo_input.addItem("Multidim Gaussian Data Points")
        self.combo_input.addItem("Manual Data Points")

        self.combo_input.activated[str].connect(self.change_input_panel)

        # Input options box (numbers of points, lines, etc.)
        self.input_options_box, self.input_params = self.options_mesh()

        # Transformation combo box (to choose translation, rotation, etc.)
        self.combo_transform = QtGui.QComboBox(self)
        self.combo_transform.addItem("Translation")
        self.combo_transform.addItem("Rotation")
        self.combo_transform.addItem("Scaling")
        self.combo_transform.addItem("Shear")
        self.combo_transform.addItem("Compression")
        self.combo_transform.addItem("Polynomial (CM)")
        self.combo_transform.addItem("Log")
        self.combo_transform.addItem("Matrix product")
        self.combo_transform.addItem("Exponential Kernel")
        self.combo_transform.addItem("Exponential Kernel Normalized")

        # Generation button
        self.button_generate = QtGui.QPushButton('Generate')
        self.button_generate.clicked.connect(self.change_graph_panel)

        # A combobox to choose the transformations

        self.label_transformation = QtGui.QLabel(self)
        self.label_transformation.setText("Transformation :")

        self.options_func = {"Translation": self.options_translation,
                             "Rotation": self.options_rotation,
                             "Scaling": self.options_scaling,
                             "Shear": self.options_shear,
                             "Compression": self.options_compression,
                             "Polynomial (CM)": self.options_polynomial,
                             "Log": self.options_log,
                             "Matrix product": self.options_matrix_product,
                             "Exponential Kernel": self.options_kernel,
                             "Exponential Kernel Normalized": self.options_normalized_kernel,
                             "Linear Transform": self.options_linear_transform,
                             "Non Linear Transform": self.options_non_linear_transform}

        self.input_options_func = {"Mesh": self.options_mesh,
                                   "Gaussian Data Points": self.options_gaussian_points,
                                   "Multidim Gaussian Data Points": self.options_multidim_gaussian_points,
                                   "Manual Data Points": self.options_manual_points}

        self.trans_func = {"Translation": self.translation,
                           "Rotation": self.rotation,
                           "Scaling": self.scaling,
                           "Shear": self.shear,
                           "Compression": self.compression,
                           "Polynomial (CM)": self.polynomial,
                           "Log": self.log,
                           "Matrix product": self.matrix_product,
                           "Exponential Kernel": self.kernel,
                           "Exponential Kernel Normalized": self.normalized_kernel,
                           "Linear Transform": self.linear_transform}

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

        self.nbr_iter_label = QtGui.QLabel(self)
        self.nbr_iter_label.setText("Number of iterations :")
        self.nbr_iter_text = QtGui.QLineEdit(self)
        self.nbr_iter_text.setText('300')
        self.nbr_pairs_label = QtGui.QLabel(self)
        self.nbr_pairs_label.setText("Number of pairs :")
        self.nbr_pairs_text = QtGui.QLineEdit(self)
        self.nbr_pairs_text.setText('2000')
        self.proportion_positive_label = QtGui.QLabel(self)
        self.proportion_positive_label.setText("Proportion of + pairs :")
        self.proportion_positive_text = QtGui.QLineEdit(self)
        self.proportion_positive_text.setText('0.5')


        self.options_NN.addWidget(self.nbr_iter_label, 1, 0)
        self.options_NN.addWidget(self.nbr_iter_text, 1, 1)
        self.options_NN.addWidget(self.nbr_pairs_label, 2, 0)
        self.options_NN.addWidget(self.nbr_pairs_text, 2, 1)
        self.options_NN.addWidget(self.proportion_positive_label, 3, 0)
        self.options_NN.addWidget(self.proportion_positive_text, 3, 1)


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
        self.grid.addWidget(self.combo_transform, 1, 1)
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
        graph_type = str(self.combo_input.currentText())

        if graph_type == "Gaussian Data Points" or graph_type == "Mesh":
            self.train_onedim()
        elif graph_type == "Multidim Gaussian Data Points":
            self.train_multidim()

    def train_onedim(self):
        model = Model(network=str(self.combo_NN.currentText()))

        print("=================== Preprocess the data ===================")
        X_source = np.array([[z.real, z.imag] for z in self.source_z])
        X_target = np.array([[z.real, z.imag] for z in self.target_z[-1]])
        X_mesh = np.array([[[self.mesh_z[-1][i, j].real, self.mesh_z[-1][i, j].imag]\
                            for j in range(self.mesh_z[-1].shape[1])]
                            for i in range(self.mesh_z[-1].shape[0])])

        nbr_pairs = int(self.nbr_pairs_text.text())
        proportion_positive = float(self.proportion_positive_text.text())
        
        pairs, sim = createPairs(X_source, X_target, self.labels, nbr_pairs, proportion_positive)

        print("=================== Training and testing ===================")

        hist = model.fit(pairs[:,0], pairs[:,1], sim, nb_iter=int(self.nbr_iter_text.text()))

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

        feat_target = model.get_target(X_target)
        feat_source = model.get_source(X_source)

        self.mesh_z.append(np.array([np.vectorize(complex)(model.get_source(X_mesh[i, :])[:,0], model.get_source(X_mesh[i, :])[:,1])
                            for i in range(X_mesh.shape[0])]))
        self.mesh_z.append(np.array([np.vectorize(complex)(model.get_target(X_mesh[i, :])[:,0], model.get_target(X_mesh[i, :])[:,1])
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

    def train_multidim(self):
        self.model.compile()
        nbr_pairs = int(self.nbr_pairs_text.text())
        proportion_positive = float(self.proportion_positive_text.text())
        if self.check_onlyP.isChecked():
            pairs, sim = createPairsPositive(self.X_source[-1], self.X_target[-1], self.labels, nbr_pairs, proportion_positive)
        else:
            pairs, sim = createPairs(self.X_source[-1], self.X_target[-1], self.labels, nbr_pairs, proportion_positive)

        print("=================== Training and testing ===================")

        hist = self.model.fit(pairs[:,0], pairs[:,1], sim, nb_iter=int(self.nbr_iter_text.text()))

        # Display the loss and the weights
        loss = hist.history['loss'][-1]
        val_loss = hist.history['val_loss'][-1]
        self.model.model.save_weights("weights.h5", overwrite=True)
        f = h5py.File('weights.h5', 'r')

        if str(self.combo_NN.text()) == "Siamese":
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

        self.draw_multidim_points(self.figure_after)


    def init_mesh(self):
        try:
            nbr_lines = float(self.input_params["nbr_lines"].text())
        except ValueError:
            nbr_lines = 30
        x = np.linspace(-10, 10, nbr_lines)
        x = np.concatenate([x, [0]])
        y = np.linspace(-10, 10, nbr_lines)
        y = np.concatenate([y, [0]])
        mesh_x, mesh_y = np.meshgrid(x, y)
        self.mesh_z = [np.vectorize(complex)(mesh_x, mesh_y)]

        self.draw_mesh(self.figure_before)
        self.figure_before.canvas.draw()

    def draw_mesh(self, figure, index={'target':-1, 'source':0}):
        ax = figure.axes[0]
        ax.clear()
        ax.axis((-3, 3, -1.5, 1.5))
        #ax.axhline(y=0, color='k', linewidth=2.0)
        #ax.axvline(x=0, color='k', linewidth=2.0)

        ax2 = self.figure_after.axes[0]
        ax2.clear()
        ax2.axis((-3, 3, -1.5, 1.5))
        #ax2.axhline(y=0, color='k', linewidth=2.0)
        #ax2.axvline(x=0, color='k', linewidth=2.0)

        color, linewidth = "red", 1.0
        for i in range(self.mesh_z[-1].shape[0]):
            if i == self.mesh_z[-1].shape[0]-1: # draw the black axis
                color, linewidth = "k", 2.0
            ax.add_line(Line2D(self.mesh_z[index['target']].real[i, :], self.mesh_z[index['target']].imag[i, :],
                                color=color, linewidth=linewidth))
            ax.add_line(Line2D(self.mesh_z[index['source']].real[i, :], self.mesh_z[index['source']].imag[i, :],
                                color=color, linestyle="--", linewidth=linewidth))

        color, linewidth = "blue", 1
        for j in range(self.mesh_z[-1].shape[1]):
            if j == self.mesh_z[-1].shape[0]-1:
                color, linewidth = "k", 2.0
            ax.add_line(Line2D(self.mesh_z[index['target']].real[:, j], self.mesh_z[index['target']].imag[:, j],
                                color=color, linewidth=linewidth)) # draw vertical line
            ax.add_line(Line2D(self.mesh_z[index['source']].real[:, j], self.mesh_z[index['source']].imag[:, j],
                                color=color, linestyle="--", linewidth=linewidth)) # draw vertical line

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

    def init_multidim_gaussian_data_points(self):
        try:
            nbr_points = int(self.input_params["nbr_points"].text())
        except ValueError:
            nbr_points = 50
        try:
            dimension = int(self.input_params["nbr_dim"].text())
        except ValueError:
            dimension = 3
        try:
            nbr_classes = int(self.input_params["nbr_classes"].text())
        except ValueError:
            nbr_classes = 4
        try:
            mean_factor = float(self.input_params["mean"].text())
        except ValueError:
            mean_factor = 1
        try:
            std = float(self.input_params["std"].text())
        except ValueError:
            std = 0.01
        try:
            network = str(self.input_params["network"].currentText())
        except ValueError:
            network = "Siamese"

        # mean : list of vectors (the mean for each class)
        # mean for class i is the binary conversion of i (taken as a vector)
        def compute_mean(i):
            if i != 0 and i % (2**dimension) == 0: # if the result would be 0 but it's not 0
                return -compute_mean(i-1)
            res = np.array(list(bin(i_class & (2**dimension-1))[2:].zfill(dimension)), dtype=float)
            res = mean_factor * res
            res += i_class // (2**dimension)
            return res
        mean = np.array([compute_mean(i_class) for i_class in range(nbr_classes)])
        print("mean :", mean)

        # covariance : list of covariance matrix. We take the same for all classes
        covariance = np.array([std * np.array([[int(i==j) for j in range(dimension)] for i in range(dimension)])
                        for i_class in range(nbr_classes)])

        # X_source[i] : matrix (nbr_points*nbr_classes X dimension) after i transformations.
        self.X_source, self.X_target = np.empty((0,dimension)), np.empty((0,dimension))
        print self.X_source
        for i_class in range(nbr_classes):
            gaussian_source = np.random.multivariate_normal(mean[i_class], covariance[i_class], nbr_points).T
            gaussian_source = np.concatenate(list(gaussian_source.reshape(dimension, nbr_points,1)), axis=1)
            self.X_source = np.concatenate([self.X_source, gaussian_source], axis=0)

            gaussian_target = np.random.multivariate_normal(mean[i_class], covariance[i_class], nbr_points).T
            gaussian_target = np.concatenate(list(gaussian_target.reshape(dimension, nbr_points,1)), axis=1)
            self.X_target = np.concatenate([self.X_target, gaussian_target], axis=0)
        self.X_source = [self.X_source]
        self.X_target = [self.X_target]

        self.labels = np.array([i_point//nbr_points for i_point in range(nbr_classes*nbr_points)])
        #print ("labels : ", self.labels)

        self.model = Model(network=network, input_size=dimension)
        self.draw_multidim_points(self.figure_before)

    def draw_data_points(self, figure):
        ax = figure.axes[0]
        ax.clear()
        ax.axis((-3, 3, -1.5, 1.5))
        #ax.axhline(y=0, color='k', linewidth=2.0)
        #ax.axvline(x=0, color='k', linewidth=2.0)

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

    def draw_multidim_points(self, figure):
        ax = figure.axes[0]
        ax.clear()
        ax.axis((-3, 3, -1.5, 1.5))
        ax.axhline(y=0, color='k', linewidth=2.0)
        ax.axvline(x=0, color='k', linewidth=2.0)
        source = self.model.get_source(self.X_source[-1])
        target = self.model.get_target(self.X_target[-1])

        ax.scatter(target[:,0], target[:,1], c=self.labels, alpha=0.8)
        ax.scatter(source[:,0], source[:,1], c=self.labels, marker="x")

        figure.canvas.draw()

    def change_graph_panel(self):
        """ Results of button 'Generate : choose the good function to init the graph"""

        graph_type = str(self.combo_input.currentText())
        if graph_type == "Mesh":
            self.init_mesh()
        elif graph_type == "Gaussian Data Points":
            self.init_gaussian_data_points()
        elif graph_type == "Multidim Gaussian Data Points":
            self.init_multidim_gaussian_data_points()
            self.change_options_panel() # to reload the number of dim for the matrix for example
            self.change_transform_combo()
            self.change_nn_combo()
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
        self.change_nn_combo()
        self.change_transform_combo()
        self.change_options_panel()

    def change_options_panel(self):
        """ Change the panel with the options of the transformation"""

        # Remove the old panel
        while self.options_box.count():
            item = self.options_box.takeAt(0)
            widget = item.widget()
            widget.deleteLater()

        # Create the new panel
        self.options_box, self.trans_params = self.options_func[str(self.combo_transform.currentText())]()
        self.grid.addLayout(self.options_box, 2, 1)
        self.setLayout(self.grid)

    def change_transform_combo(self):
        graph_type = str(self.combo_input.currentText())
        self.combo_transform.deleteLater()
        self.combo_transform = QtGui.QComboBox(self)
        if graph_type == "Mesh" or graph_type == "Gaussian Data Points" or graph_type == "Manual Data Points":
            self.combo_transform.addItem("Translation")
            self.combo_transform.addItem("Rotation")
            self.combo_transform.addItem("Scaling")
            self.combo_transform.addItem("Shear")
            self.combo_transform.addItem("Compression")
            self.combo_transform.addItem("Polynomial (CM)")
            self.combo_transform.addItem("Log")
            self.combo_transform.addItem("Matrix product")
            self.combo_transform.addItem("Exponential Kernel")
            self.combo_transform.addItem("Exponential Kernel Normalized")
            self.combo_transform.activated[str].connect(self.change_options_panel)
        elif graph_type == "Multidim Gaussian Data Points" :
            self.combo_transform.addItem("Linear Transform")
            self.combo_transform.addItem("Non Linear Transform")

        self.grid.addWidget(self.combo_transform, 1, 1)

    def change_nn_combo(self):
        graph_type = str(self.combo_input.currentText())
        self.combo_NN.deleteLater()
        if graph_type == "Multidim Gaussian Data Points":
            self.combo_NN = QtGui.QLabel(self)
            self.combo_NN.setText(str(self.input_params["network"].currentText()))
        else:
            self.combo_NN = QtGui.QComboBox(self)
            self.combo_NN.addItem("Corrector")
            self.combo_NN.addItem("Symetric")

        self.grid.addWidget(self.combo_NN, 1, 2)


    def transform(self):
        self.ax_before.clear()
        self.options_func[str(self.combo_transform.currentText())]()
        graph_type = str(self.combo_input.currentText())

        if graph_type != "Multidim Gaussian Data Points":
            self.mesh_z.append(np.array([[self.trans_func[str(self.combo_transform.currentText())](self.mesh_z[-1][i, j], "mesh")\
                                for j in range(self.mesh_z[-1].shape[1])]
                                for i in range(self.mesh_z[-1].shape[0])]))

        if graph_type == "Mesh":
            self.draw_mesh(self.figure_before)

        if graph_type == "Gaussian Data Points":
            self.target_z.append(np.array([self.trans_func[str(self.combo_transform.currentText())](self.target_z[-1][i], "target")\
                            for i in range(self.target_z[-1].shape[0])]))
            self.draw_data_points(self.figure_before)
        if graph_type == "Multidim Gaussian Data Points":
            self.draw_multidim_points(self.figure_before)
            self.X_target.append(self.trans_func[str(self.combo_transform.currentText())]())
            self.draw_multidim_points(self.figure_before)

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

    def options_multidim_gaussian_points(self):
        grid = QtGui.QGridLayout()

        nbr_points_label = QtGui.QLabel("Number of points: ", self)
        nbr_points_text = QtGui.QLineEdit(self)
        nbr_points_text.setText('50')

        nbr_dim_label = QtGui.QLabel("Number of dimensions: ", self)
        nbr_dim_text = QtGui.QLineEdit(self)
        nbr_dim_text.setText('3')

        nbr_classes_label = QtGui.QLabel("Number of classes: ", self)
        nbr_classes_text = QtGui.QLineEdit(self)
        nbr_classes_text.setText('4')

        mean_label = QtGui.QLabel("Mean: ", self)
        mean_text = QtGui.QLineEdit(self)
        mean_text.setText('1')

        std_label = QtGui.QLabel("STD:", self)
        std_text = QtGui.QLineEdit(self)
        std_text.setText('0.01')

        network_label = QtGui.QLabel(self)
        network_label.setText("Neural Network :")
        network_combo = QtGui.QComboBox(self)
        network_combo.addItem("Siamese")
        network_combo.addItem("DANN")
        network_combo.addItem("XANN")


        grid.addWidget(nbr_points_label, 1, 0)
        grid.addWidget(nbr_points_text, 1, 1)
        grid.addWidget(nbr_dim_label, 2, 0)
        grid.addWidget(nbr_dim_text, 2, 1)
        grid.addWidget(nbr_classes_label, 3, 0)
        grid.addWidget(nbr_classes_text, 3, 1)
        grid.addWidget(mean_label, 4, 0)
        grid.addWidget(mean_text, 4, 1)
        grid.addWidget(std_label, 5, 0)
        grid.addWidget(std_text, 5, 1)
        grid.addWidget(network_label, 6, 0)
        grid.addWidget(network_combo, 6, 1)

        params = {"nbr_points": nbr_points_text, "nbr_dim": nbr_dim_text,
                "nbr_classes": nbr_classes_text, "mean": mean_text,
                "std": std_text, "network": network_combo}

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
            beta_text[i] = QtGui.QLineEdit(self)
            beta_text[i].setText('1')
            x_text[i] = QtGui.QLineEdit(self)
            x_text[i].setText('0')
            y_text[i] = QtGui.QLineEdit(self)
            y_text[i].setText('0')
        alpha_text[0].setText('1')
        alpha_text[1].setText('-1')
        alpha_text[2].setText('-1')
        alpha_text[3].setText('1')
        beta_text[0].setText('1')
        beta_text[1].setText('-1')
        beta_text[2].setText('1')
        beta_text[3].setText('-1')
        x_text[0].setText('0')
        x_text[1].setText('0')
        x_text[2].setText('1')
        x_text[3].setText('-1')
        y_text[0].setText('1')
        y_text[1].setText('-1')
        y_text[2].setText('0')
        y_text[3].setText('0')
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
        Y = np.sum([beta[i] * np.exp(-gamma * np.linalg.norm(vect_z - vect_x[i])**2) for i in range(nbr_noeuds)])
        return complex(*tuple(np.array([X, Y])))

    def options_normalized_kernel(self):
        nbr_noeuds = 4
        grid = QtGui.QGridLayout()

        gamma_label = QtGui.QLabel("Gamma :", self)
        gamma_text = QtGui.QLineEdit(self)
        gamma_text.setText('1')
        noeud_label = [QtGui.QLabel("Noeud " + str(i) + " (x, y):", self) for i in range(nbr_noeuds)]

        x_text = [0]*nbr_noeuds
        y_text = [0]*nbr_noeuds

        for i in range(nbr_noeuds):
            x_text[i] = QtGui.QLineEdit(self)
            x_text[i].setText('0')
            y_text[i] = QtGui.QLineEdit(self)
            y_text[i].setText('0')
        x_text[0].setText('0')
        x_text[1].setText('0')
        x_text[2].setText('1')
        x_text[3].setText('-1')
        y_text[0].setText('1')
        y_text[1].setText('-1')
        y_text[2].setText('0')
        y_text[3].setText('0')
        grid.addWidget(gamma_label, 1, 0)
        grid.addWidget(gamma_text, 1, 1)
        for i in range(nbr_noeuds):
            grid.addWidget(noeud_label[i], 2+i, 0)
            grid.addWidget(x_text[i], 2+i, 1)
            grid.addWidget(y_text[i], 2+i, 2)

        params = {"gamma": gamma_text, "x": x_text, "y": y_text}
        return grid, params

    def normalized_kernel (self, z, data_type):
        nbr_noeuds = 4
        try:
            gamma = float(self.trans_params["gamma"].text())
        except ValueError:
            gamma = 1
        try:
            x = [float(self.trans_params["x"][i].text()) for i in range(nbr_noeuds)]
        except ValueError:
            x = [0 for i in range(nbr_noeuds)]
        try:
            y = [float(self.trans_params["y"][i].text()) for i in range(nbr_noeuds)]
        except ValueError:
            y = [0 for i in range(nbr_noeuds)]

        def K(x1,x2):
            return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

        vect_z = np.array([z.real, z.imag])
        vect_x = [np.array([x[i], y[i]]) for i in range(nbr_noeuds)]
        X = np.sum([x[i] * K(vect_z, vect_x[i]) for i in range(nbr_noeuds)])
        Y = np.sum([y[i] * K(vect_z, vect_x[i]) for i in range(nbr_noeuds)])
        X *= 1/(np.sum([K(vect_z, vect_x[i]) for i in range(nbr_noeuds)]))
        Y *= 1/(np.sum([K(vect_z, vect_x[i]) for i in range(nbr_noeuds)]))
        return complex(*tuple(np.array([X, Y])))

    def options_linear_transform(self):
        grid = QtGui.QGridLayout()
        dimension = int(self.input_params["nbr_dim"].text())

        matrix_label = QtGui.QLabel("Matrix:", self)
        matrix_text = [[0 for i in range(dimension)] for j in range(dimension)]
        for i in range(dimension):
            for j in range(dimension):
                matrix_text[i][j] = QtGui.QLineEdit(self)
                matrix_text[i][j].setText(str(int(i==j)))

        biais_label = QtGui.QLabel("Biais:", self)
        biais_text = [0]*dimension
        for i in range(dimension):
            biais_text[i] = QtGui.QLineEdit(self)
            biais_text[i].setText('0')

        grid.addWidget(matrix_label, 1, 0)
        for i in range(dimension):
            for j in range(dimension):
                grid.addWidget(matrix_text[i][j], 1+i, 1+j)
        grid.addWidget(biais_label, 1+dimension, 0)
        for i in range(dimension):
            grid.addWidget(biais_text[i], 1+dimension, 1+i)

        params = {"matrix": matrix_text, "biais": biais_text}
        return grid, params

    def linear_transform(self):
        """ Return the new value of X_source
        (matrix of shape (nbr_points*nbr_classes, dimension)))"""

        dimension = int(self.input_params["nbr_dim"].text())
        try:
            matrix_widget = self.trans_params["matrix"]
            print ("Shape matrix", np.array(matrix_widget).shape)
            matrix = np.array([[float(matrix_widget[i][j].text()) for j in range(dimension)]
                        for i in range(dimension)])
        except ValueError:
            matrix = np.eye(dimension)
            print ("Error   ")
        try:
            biais_widget = self.trans_params["biais"]
            biais = [float(biais_widget[i].text()) for i in range(dimension)]
        except ValueError:
            biais = np.zeros(dimension)
            print("Error")

        new_X = [np.dot(matrix, self.X_target[-1][i_point]) + biais for i_point in range(len(self.X_target[-1]))]
        print ("Matrix", matrix)
        print ("Biais", biais)
        new_X = np.array(new_X).reshape(len(self.X_target[-1]),dimension)
        return new_X

    def options_non_linear_transform(self):
        pass
# =========== Main Program ==============

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
