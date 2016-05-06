import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from random import randint

os.environ["KERAS_BACKEND"] = "theano"
import theano

from keras.models import Sequential, Graph
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Dense, Merge, Lambda
from keras import backend as K

# ================= General functions ===================

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs.values()
    return K.sqrt((K.square(u - v)).sum(axis=1, keepdims=True))

def contrastive_loss(y, d):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''

    margin = 1
    return K.mean(y * K.square(d) + (1 - y) * K.maximum(margin - d, 0))
    #return K.mean(y * K.square(d) + (1 - y) * K.exp(- K.square(d)))

def createPairs(X_source, X_target, labels, nbr_pairs=2000, proportion_positive=0.5):
    """ We create a dataset of 2000 pairs taken randomly
    We consider len(X_source) == len(X_target) and the labels are the same
    between source and target """

    pairs = []
    sim = []
    i_pair_negative = 0
    i_pair_positive = 0
    nbr_pairs_positive = np.floor(nbr_pairs * proportion_positive)
    nbr_pairs_negative = nbr_pairs - nbr_pairs_positive
    while i_pair_positive < nbr_pairs_positive or i_pair_negative < nbr_pairs_negative:
        index_source = randint(0,len(X_source)-1)
        index_target = randint(0,len(X_target)-1)
        cur_sim = int(labels[index_source] == labels[index_target])
        if cur_sim and i_pair_positive < nbr_pairs_positive:
            sim.append(1)
            pairs.append([X_source[index_source], X_target[index_target]])
            i_pair_positive += 1
        elif not cur_sim and i_pair_negative < nbr_pairs_negative:
            sim.append(0)
            pairs.append([X_source[index_source], X_target[index_target]])
            i_pair_negative += 1

    print("bincount(sim) : ", np.bincount(sim))
    pairs = np.array(pairs)
    sim = np.array(sim)

    return pairs, sim

def createPairsPositive(X_source, X_target, labels, nbr_pairs=2000):
    """ We create a dataset of 2000 pairs taken randomly
    We consider len(X_source) == len(X_target) and the labels are the same
    between source and target """

    pairs = []
    sim = []
    iPair = 0
    while iPair < nbr_pairs:
        index_source = randint(0,len(X_source)-1)
        index_target = randint(0,len(X_target)-1)

        if int(labels[index_source] == labels[index_target]):
            pairs.append([X_source[index_source], X_target[index_target]])
            sim.append(int(labels[index_source] == labels[index_target]))
            iPair += 1
    print "Nbr pairs : ", len(sim)
    pairs = np.array(pairs)
    sim = np.array(sim)

    return pairs, sim

# ================ Neural network class ==================

class Model:
    def __init__(self, network, input_size=2, batch_size=20):
        """ Initiate the neural network and compile it
        Input : network : string in ['Corrector', 'Symetric']"""

        self.batch_size = 20
        self.network = network
        self.input_size = input_size

        # Instantiation the NN
        self.model = Graph()

        if self.network == 'Symetric':
            self.buildSymetricNN()
        elif self.network == 'Corrector':
            self.buildCorrectorNN()
        elif self.network == 'Siamese':
            self.buildSymetricNN()
        self.compile()

    def compile (self):
        # Compile the NN
        rms = RMSprop()
        self.model.compile(loss={'output': contrastive_loss}, optimizer=rms)

    def buildSymetricNN(self):
        self.model.add_input(name='source', input_shape=(self.input_size,))
        self.model.add_input(name='target', input_shape=(self.input_size,))

        if self.input_size == 2:
            init_func = "identity"
            print ("identity initialization")
        else:
            init_func = "lecun_uniform"

        self.model.add_node(Dense(2, init=init_func), name='corrector_s', input='source')
        self.model.add_node(Dense(2, init=init_func), name='corrector_t', input='target')

        self.model.add_node(Lambda(euclidean_distance),
           inputs=['corrector_s', 'corrector_t'],
           merge_mode='join',
           name='d')
        self.model.add_output(name='output', input='d')

    def buildCorrectorNN(self):
        self.model.add_input(name='source', input_shape=(2,))
        self.model.add_input(name='target', input_shape=(2,))

        self.model.add_node(Dense(2, init='identity'), name='corrector_t', input='target')

        self.model.add_node(Lambda(euclidean_distance),
           inputs=['source', 'corrector_t'],
           merge_mode='join',
           name='d')
        self.model.add_output(name='output', input='d')

    def buildDeepCorrectorNN(self):
        self.model.add_input(name='source', input_shape=(2,))
        self.model.add_input(name='target', input_shape=(2,))


        self.model.add_node(Dense(10), name='ip1', input='target')
        self.model.add_node(Dense(2), name='corrector_t', input='ip1')

        self.model.add_node(Lambda(euclidean_distance),
           inputs=['source', 'corrector_t'],
           merge_mode='join',
           name='d')
        self.model.add_output(name='output', input='d')

    def fit(self, source, target, sim, nb_iter=150):
        return self.model.fit({'source': source[:len(source)//2], 'target': target[:len(source)//2], 'output': sim[:len(source)//2]},
            validation_data={'source': source[len(source)//2:], 'target': target[len(source)//2:], 'output': sim[len(source)//2:]},
            batch_size=self.batch_size,
            nb_epoch=nb_iter)

    def get_target(self, target):
        get_feature_target = theano.function([self.model.inputs['target'].input],
            self.model.nodes['corrector_t'].get_output(train=False),
            allow_input_downcast=False)
        feat_target = get_feature_target(np.array(target, dtype="float32"))

        return feat_target

    def get_source(self, source):
        if self.network == 'Symetric':
            get_feature_source = theano.function([self.model.inputs['source'].input],
                self.model.nodes['corrector_s'].get_output(train=False),
                allow_input_downcast=False)

        elif self.network == 'Corrector':
            get_feature_source = theano.function([self.model.inputs['source'].input],
                self.model.inputs['source'].get_output(train=False),
                allow_input_downcast=False)

        elif self.network == 'Siamese':
            get_feature_source = theano.function([self.model.inputs['source'].input],
                self.model.nodes['corrector_s'].get_output(train=False),
                allow_input_downcast=False)

        feat_source = get_feature_source(np.array(source, dtype="float32"))

        return feat_source
