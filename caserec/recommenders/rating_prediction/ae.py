# coding=utf-8
"""
    Autoencoder
    [Rating Prediction]

"""

# © 2018. Case Recommender (MIT License)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import optimizers
from caserec.recommenders.rating_prediction.base_rating_prediction import BaseRatingPrediction
from caserec.utils.extra_functions import timed

__author__ = 'Joao Felipe Guedes <guedes.joaofelipe@poli.ufrj.br>'


class AE(BaseRatingPrediction):
    def __init__(self, train_file=None, test_file=None, output_file=None, sep='\t', output_sep='\t',
                 random_seed=None, num_neurons=10, epochs = 2000, learning_rate=0.001, loss_metric='mse', training_metrics=['mae'], patience=10, kernel_initializer='random_uniform'):
        """
        Autoencoder for rating prediction

        An autoencoder model map both users and items to a joint latent factor space of dimensionality num_neurons. 
        It does so by having a neural network with an n_columns-sized input layer and n_columns sized-output layer. 
        The neural network compresses the input matrix to the hidden layer and decompresses it to the output layer. 
        
        Different from othe rating-based prediction models, such neural network scales the input matrix to [-1,1]. 
        
        Usage::

            >> AE(train, test).compute()

        :param train_file: File which contains the train set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type train_file: str

        :param test_file: File which contains the test set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type test_file: str, default None

        :param output_file: File with dir to write the final predictions
        :type output_file: str, default None

        :param factors: Number of latent factors per user/item
        :type factors: int, default 10

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param output_sep: Delimiter for output file
        :type output_sep: str, default '\t'

        :param random_seed: Number of seed. Lock random numbers for reproducibility of experiments.
        :type random_seed: int, default None

        :param num_neurons: number of neurons to be allocated for the single hidden layer
        :param epochs: maximum number of epochs to train
        
        :param learning_rate: to be used on the adam optimizer for the neural network's training phase
        
        :param loss_metric: a sklearn-based metric to be minimized during training

        :param training_metrics: metrics to be measured during training time, epoch by epoch
        
        :param patience: how many epochs to be used on EarlyStopping. This is used to stop training due to loss-metric stability

        """
        super(AE, self).__init__(train_file=train_file, test_file=test_file, output_file=output_file, sep=sep,
                                  output_sep=output_sep)

        self.recommender_name = 'AE'        
        self.learning_rate = learning_rate        
        self.num_neurons = num_neurons
        self.epochs = epochs
        self.loss_metric = loss_metric
        self.training_metrics = training_metrics
        self.patience = patience
        self.kernel_initializer = kernel_initializer

        if random_seed is not None:
            np.random.seed(random_seed)

        # internal vars
        self.feedback_triples = None
        self.prediction_matrix = None

    def init_model(self):
        """
        Method to treat and initialize the model

        """

        self.feedback_triples = []

        # Map interaction with ids
        for user in self.train_set['feedback']:
            for item in self.train_set['feedback'][user]:
                self.feedback_triples.append((self.user_to_user_id[user], self.item_to_item_id[item],
                                              self.train_set['feedback'][user][item]))

        self.create_matrix()

        #Setting autoencoder
        self.norm_scaler = MinMaxScaler(feature_range = (-1,1))
        self.input_dim = self.matrix.shape[1] 

        self.autoencoder = Sequential([
                                  Dense(self.num_neurons, activation = 'selu', kernel_initializer = self.kernel_initializer, input_dim = self.input_dim),
                                  Dense(self.input_dim, activation = 'tanh', kernel_initializer = self.kernel_initializer)
                                 ])

        self.earlyStopping = EarlyStopping(monitor = 'loss', patience = self.patience, mode = 'auto')

    def fit(self):
        """
        This method performs an Autoencoder to reconstruct the utility matrix over the training data.
        """

        # Fitting matrix scaler for the range [-1, 1]        
        self.norm_scaler.fit(self.matrix)
        self.matrix_normalized = self.norm_scaler.transform(self.matrix)                

        self.Adam = optimizers.Adam(lr = self.learning_rate)

        self.autoencoder.compile(optimizer = self.Adam, loss = self.loss_metric, metrics=self.training_metrics)

        self.fit_history = self.autoencoder.fit(
                                    self.matrix_normalized, 
                                    self.matrix_normalized, 
                                    epochs = 2000,
                                    verbose = 0,
                                    shuffle = True,
                                    #validation_data = (self.x_norm_test, self.x_norm_test), 
                                    callbacks = [self.earlyStopping])

        # Predicting output for normalized input matrix
        output_normalized = self.autoencoder.predict(self.matrix_normalized);

        # Denormalizing output
        output_denormalized = self.norm_scaler.inverse_transform(output_normalized);

        self.prediction_matrix = output_denormalized

    def predict_score(self, u, i, cond=True):
        """
        Method to predict a single score for a pair (user, item)

        :param u: User ID
        :type u: int

        :param i: Item ID
        :type i: int

        :param cond: Use max and min values of train set to limit score
        :type cond: bool, default True

        :return: Score generate for pair (user, item)
        :rtype: float

        """

        rui = self.train_set["mean_value"] + self.prediction_matrix[u][i]

        if cond:
            if rui > self.train_set["max_value"]:
                rui = self.train_set["max_value"]
            elif rui < self.train_set["min_value"]:
                rui = self.train_set["min_value"]

        return rui

    def predict(self):
        """
        This method computes a final rating for unknown pairs (user, item)

        """

        if self.test_file is not None:
            for user in self.test_set['users']:
                for item in self.test_set['feedback'][user]:
                    self.predictions.append((user, item, self.predict_score(self.user_to_user_id[user],
                                                                            self.item_to_item_id[item], True)))
        else:
            raise NotImplementedError

    def compute(self, verbose=True, metrics=None, verbose_evaluation=True, as_table=False, table_sep='\t'):
        """
        Extends compute method from BaseRatingPrediction. Method to run recommender algorithm

        :param verbose: Print recommender and database information
        :type verbose: bool, default True

        :param metrics: List of evaluation measures
        :type metrics: list, default None

        :param verbose_evaluation: Print the evaluation results
        :type verbose_evaluation: bool, default True

        :param as_table: Print the evaluation results as table
        :type as_table: bool, default False

        :param table_sep: Delimiter for print results (only work with verbose=True and as_table=True)
        :type table_sep: str, default '\t'

        """

        super(AE, self).compute(verbose=verbose)

        if verbose:
            self.init_model()
            print("training_time:: %4f sec" % timed(self.fit))
            if self.extra_info_header is not None:
                print(self.extra_info_header)

            print("prediction_time:: %4f sec" % timed(self.predict))

            print('\n')

        else:
            # Execute all in silence without prints
            self.init_model()
            self.fit()
            self.predict()

        self.write_predictions()

        if self.test_file is not None:
            self.evaluate(metrics, verbose_evaluation, as_table=as_table, table_sep=table_sep)


if __name__ == "__main__":

    import os 
    import pandas as pd 

    datasets_folder = './../../../../../Datasets/MovieLens/100k_raw/u.data'

    # Testing train/test data as filepaths
    tr = datasets_folder
    te = None #datasets_folder

    # Testing train/test data as dataframes 
    tr = pd.read_csv(datasets_folder, sep='\t', header=None, names = ['user', 'item', 'feedback_value', 'timestamp'])
    te = tr
    
    model = AE(tr, te)
    model.compute(verbose=True)

