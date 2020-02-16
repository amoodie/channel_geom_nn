import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


class ChannelDataset(object):
    """
    Data handler for channel geometry dataset work.

    Wrap all the normalization and such away.
    """
    def __init__(self, datasrc, scale=None, test_split=0.3, summary=True):

        self.datasrc = datasrc
        self.test_split = test_split

        # read the data using pandas
        if self.datasrc == 'mcelroy':
            self.df = pd.read_csv('data/mcelroy_dataclean.csv')
        elif self.datasrc == 'wilkerson':
            self.df = pd.read_csv('data/wilkerson_dataclean.csv') 
        elif self.datasrc == 'trampush':
            self.df = pd.read_csv('data/ShieldsJHRData.csv') # read data set using pandas
        else:
            raise ValueError('Bad flag supplied for datasrc: %s' % self.datasrc)

        # clean dataset
        self.df = self.df.dropna(inplace = False)  # Remove all nan entries.

        if summary:
            print('Data summary:\n')
            print(self.df.describe(), '\n\n') # Overview of dataset

        # subset for train and test and rescale all values
        self.df_train, self.df_test = train_test_split(self.df, test_size=self.test_split)

        self.X_train_orig = (self.df_train.drop(['Bbf.m', 'Hbf.m', 'S'], axis=1).values)
        self.y_train_orig = (self.df_train[['Bbf.m', 'Hbf.m', 'S']].values)
        self.X_test_orig = (self.df_test.drop(['Bbf.m', 'Hbf.m', 'S'], axis=1).values)
        self.y_test_orig = (self.df_test[['Bbf.m', 'Hbf.m', 'S']].values)
        self.logged = False
        self.normed = False

        # do some normalization if requested
        if scale == 'minmax':
            # min max normalization
            self.X_scaler = MinMaxScaler() # For normalizing dataset
            self.y_scaler = MinMaxScaler() # For normalizing dataset
            self.normed = True
        elif scale == 'log':
            # log(x) normalization
            self.X_scaler = LogScaler()
            self.y_scaler = LogScaler()

            self.logged = True
        else:
            self.X_scaler = DummyScaler()
            self.y_scaler = DummyScaler()

        self.X_train = self.X_scaler.fit_transform(self.X_train_orig)
        self.y_train = self.y_scaler.fit_transform(self.y_train_orig)
        self.X_test = self.X_scaler.transform(self.X_test_orig)
        self.y_test = self.y_scaler.transform(self.y_test_orig)

        # set up data for mini-batching during training
        self.batch_size = 1
        self.buffer_size = 15
        self.batches_per_epoch = int( np.floor(self.X_train.shape[0] / self.batch_size) )
        self.ds_train = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train)).repeat().batch(self.batch_size).shuffle(self.buffer_size)
        self.it_train = self.ds_train.make_one_shot_iterator()
        
    def get_next(self):
        return self.it_train.get_next()

    def shuffle(self):
        self.ds_train.shuffle(self.buffer_size)


class LogScaler(object):
    """
    Log scaler object for consistent API
    """
    def __init__(self):
        pass

    def fit_transform(self, X):
        return self.transform(X)

    def transform(self, X):
        return np.log10(X)

    def inverse_transform(self, X):
        return np.power(10, X)



class DummyScaler(object):
    """
    Dummy scaler object for consistent API
    """
    def __init__(self):
        pass

    def fit_transform(self, X):
        return self.transform(X)

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X
