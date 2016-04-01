# Transformation-Reverser

## Description
Neural networks learning to reverse 2D space transformations (translation, rotations, scaling). A GUI allows the users to define the datasets and the transformations, and to see the result. You can choose to use two kinds of neural networks :

* Corrector :

![Corrector](/schema_corrector.png)

* Symetric :

![Symetric](/schema_symetric.png)

## Installation

It has been tested with python2.7 and the libraries below, which can be installed with pip for example :
* Numpy 1.10.4
* Matplotlib 1.5.1
* Scikit-learn 0.16.1
* PyQt4
* Theano 0.7.0
* Keras 0.3.2
* h5py 2.5.0
 
Once you have all these libraries installed, you can run the program with :
```python gui.py```
