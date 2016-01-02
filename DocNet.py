__author__ = 'ericrincon'

import matplotlib

import numpy

from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.core import Flatten

from keras.optimizers import SGD

from keras.models import Graph
from keras.layers import containers

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import MaxPooling1D

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))


    def on_epoch_end(self, epoch, logs={}):
        loss_sum = 0

        for mini_batch_loss in self.batch_losses:
            loss_sum += mini_batch_loss
        average_loss = loss_sum/len(self.batch_losses)

        self.losses.append(average_loss)
        self.val_losses.append(logs.get('val_loss'))
        self.batch_losses.clear()


class DocNet:
    def __init__(self, doc_vector_size=100, filter_sizes=[2, 3, 4, 5, 6], dropout_p=0.5, doc_max_size=50,
                 n_feature_maps=2, n_classes=2, embedding=False, hidden_layer_sizes=[100], convolution=2,
                 activation_func='relu', model_path=None):
        if not model_path:
            if convolution == 1:
                filter_sizes = [20, 30, 40, 50]
            self.model = self.create_model(doc_vector_size=doc_vector_size, filter_sizes=filter_sizes, dropout_p=dropout_p,
                                           doc_max_size=doc_max_size, n_feature_maps=n_feature_maps, n_classes=n_classes,
                                           embedding=embedding, nn_layer_sizes=hidden_layer_sizes, convolution=convolution,
                                           activation=activation_func)

    def create_model(self, doc_vector_size, filter_sizes, dropout_p, doc_max_size,
                     n_feature_maps, n_classes, activation, embedding, nn_layer_sizes, convolution):
        model = Graph()

        if convolution == 2:
            model.add_input(name='data', input_shape=(1, doc_max_size, doc_vector_size))
        elif convolution == 1:
            model.add_input(name='data', input_shape=(1, doc_vector_size))

        model.add_node(Dropout(.20), input='data', name='input_dropout')

        for filter_size in filter_sizes:
            node = containers.Sequential()

            if convolution == 2:
                node.add(Convolution2D(n_feature_maps, filter_size, doc_vector_size, input_shape=(1, doc_max_size,
                                                                                                  doc_vector_size)))
            elif convolution == 1:
                node.add(Convolution1D(n_feature_maps, filter_size, input_dim=doc_vector_size))

            node.add(Activation(activation))

            if convolution == 2:
                node.add(MaxPooling2D(pool_size=(doc_max_size - filter_size + 1, 1)))
            elif convolution == 1:
                node.add(MaxPooling1D(pool_length=doc_vector_size - filter_size + 1))

            node.add(Flatten())
            model.add_node(node, name='filter_unit_' + str(filter_size), input='input_dropout')

        fully_connected_nn = containers.Sequential()

        # Add hidden layers for the final nn layers.

        for i, layer_size in enumerate(nn_layer_sizes):
            if i == 0:
                if convolution == 2:
                    fully_connected_nn.add(Dense(layer_size, input_dim=n_feature_maps * len(filter_sizes)))

                elif convolution == 1:
                    fully_connected_nn.add(Dense(layer_size, input_dim=n_feature_maps * len(filter_sizes)))
            else:
                fully_connected_nn.add(Dense(layer_size))
            fully_connected_nn.add(Activation(activation))
            fully_connected_nn.add(Dropout(dropout_p))
        fully_connected_nn.add(Dense(n_classes))
        fully_connected_nn.add(Activation('softmax'))

        model.add_node(fully_connected_nn, name='fully_connected_nn',
                       inputs=['filter_unit_' + str(n) for n in filter_sizes])
        model.add_output(name='nn_output', input='fully_connected_nn')
        return model

    def train(self, X_train, Y_train, batch_size=32, n_epochs=10, model_name='cnn.h5py', plot=True, learning_rate=.1,
              lr_decay=1e-6, momentum=.5, nesterov=True, valid_split=.1, verbose=1, optimization_method='adagrad',
              plot_headless=False):

        if optimization_method == 'sgd':
            optim = SGD(lr=learning_rate, decay=lr_decay, momentum=momentum, nesterov=nesterov)
        else:
            optim = optimization_method

        loss_history = LossHistory()
        checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)

        self.model.compile(optimizer=optim, loss={'nn_output': 'categorical_crossentropy'})
        self.model.fit({'data': X_train, 'nn_output': Y_train}, batch_size=batch_size, nb_epoch=n_epochs,
                       callbacks=[loss_history, checkpointer], validation_split=valid_split, shuffle=True,
                       verbose=verbose)

        # Plot the the validation and training loss

        if plot:
            if plot_headless:
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            train_loss_history = loss_history.losses
            valid_loss_history = loss_history.val_losses

            epochs_axis = numpy.arange(1, n_epochs + 1)
            train_loss, = plt.plot(epochs_axis, train_loss_history, label='Train')
            val_loss, = plt.plot(epochs_axis, valid_loss_history, label='Validation')
            plt.legend(handles=[train_loss, val_loss])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')

            plt.savefig(model_name + '_loss_plot.png')

        self.model.save_weights('final_' + model_name, overwrite=True)

    def test(self, X_test, Y_test, print_metrics=True, print_output=True):
        predictions = self.predict_classes(X_test)

        # Get metrics
        accuracy = accuracy_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions)
        auc = roc_auc_score(Y_test, predictions)
        precision = precision_score(Y_test, predictions)
        recall = recall_score(Y_test, predictions)

        if print_metrics:
            print('Accuracy: ', accuracy)
            print('F1: ', f1)
            print('AUC: ', auc)
            print('Precision: ', precision)
            print('Recall: ', recall)

        if print_output:
            for i in range(Y_test.shape[0]):
                print('Predicted: {} | True: {}'.format(predictions[i], Y_test[i]))

        return accuracy, f1, auc, precision, recall

    def predict_classes(self, x):
        predictions = self.model.predict({'data': x})

        predicted_classes = numpy.argmax(predictions['nn_output'], axis=1)

        return predicted_classes