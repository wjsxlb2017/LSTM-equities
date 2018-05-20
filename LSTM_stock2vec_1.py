#!/usr/bin/env python
import numpy as np
from random import shuffle
#import itertools
#from operator import itemgetter
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import ModelCheckpoint
import h5py
from copy import deepcopy
import threading
#from timeit import default_timer as timer

# decorator for threadsafe generators
class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


BATCH_SIZE = 32 # smaller gives possibly higher accuracy but more gradient variation
VAL_BUFFER = 4 # number of samples to exclude from gradient update and reserve for validation
TIMESTEPS = 32
NUM_LABELS = 26 # number of categories for classification
#NUM_LABELS = 1 # for regression experiment

## set up fast way to get staggered sequences of training and validation data in batches
master_indices = np.zeros((BATCH_SIZE + VAL_BUFFER, TIMESTEPS), dtype=np.int64)
for i in range(master_indices.shape[0]):
    master_indices[i] = i + np.arange(TIMESTEPS)
val_subset = np.linspace(1, BATCH_SIZE + VAL_BUFFER - 1, num=VAL_BUFFER, dtype=np.int64)
VAL_INDICES = master_indices[val_subset, :]
unused_subset = np.setdiff1d(np.arange(BATCH_SIZE + VAL_BUFFER), val_subset)
X_INDICES = master_indices[unused_subset, :]
Y_TRAIN_LOCS = np.array([n * TIMESTEPS - 1 for n in range(1, BATCH_SIZE + 1)], dtype=np.uint32)
Y_VAL_LOCS = np.array([n * TIMESTEPS - 1 for n in range(1, VAL_BUFFER + 1)], dtype=np.uint32)
VAL_INDICES = VAL_INDICES.flatten()
X_INDICES = X_INDICES.flatten()
del val_subset, unused_subset, master_indices


with h5py.File('stockVectorEmbeddings.h5', 'r') as f:
    embedding_matrix = f['embedding_matrix'][:] # approx 200MB

with h5py.File('LSTM_data.h5', 'r') as f:
    sym_list = list(f.keys())


## generator for making batches of training data
#@threadsafe_generator
def train_gen():
    while True:
        shuffle(sym_list)
        for sym in sym_list:
            X_indices = np.copy(X_INDICES)
            with h5py.File('LSTM_data.h5', 'r') as f:
                embd_rows = f[sym][:-120, 0] ## hold out final 120 days for test only
                labels = f[sym][:-120, 1]
                num_batches = embd_rows.shape[0] // BATCH_SIZE
                for batch in range(num_batches):
                    while X_indices[-1] < embd_rows.shape[0]:
                        X_train = embedding_matrix[embd_rows[X_indices]]
                        X_train = np.reshape(X_train, (BATCH_SIZE, TIMESTEPS, embedding_matrix.shape[1]))
                        y_train = np_utils.to_categorical(labels[X_indices[Y_TRAIN_LOCS]])
                        #y_train = labels[X_indices[Y_TRAIN_LOCS]] # regression experiment
                        delta = NUM_LABELS - y_train.shape[1]
                        if delta > 0:
                            y_train = np.pad(y_train, ((0,0),(0,delta)), 'constant')
                        X_indices = X_indices + BATCH_SIZE + VAL_BUFFER
                        yield (X_train, y_train)


## generator for making batches of validation data in "roll-forward" cross validation scheme
#@threadsafe_generator
def validation_gen():
    while True:
        for sym in deepcopy(sym_list):
            val_indices = np.copy(VAL_INDICES)
            with h5py.File('LSTM_data.h5', 'r') as f:
                embd_rows = f[sym][:-120, 0]
                labels = f[sym][:-120, 1]
                while val_indices[-1] < embd_rows.shape[0]:
                    X_val = embedding_matrix[embd_rows[val_indices]]
                    X_val = np.reshape(X_val, (VAL_BUFFER, TIMESTEPS, embedding_matrix.shape[1]))
                    y_val = np_utils.to_categorical(labels[val_indices[Y_VAL_LOCS]])
                    #y_val = labels[val_indices[Y_VAL_LOCS]] # regression experiment
                    delta = NUM_LABELS - y_val.shape[1]
                    if delta > 0:
                        y_val = np.pad(y_val, ((0,0),(0,delta)), 'constant')
                    val_indices = val_indices + BATCH_SIZE + VAL_BUFFER
                    yield (X_val, y_val)


model = Sequential()
model.add(LSTM(64,
               input_shape=(TIMESTEPS, embedding_matrix.shape[1]),
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=regularizers.l2(0.005),
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=regularizers.l2(0.005),
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.05,
               recurrent_dropout=0.0,
               implementation=1,
               return_sequences=True, # needed for stacking
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False))
#model.add(Dropout(0.1))
model.add(LSTM(32,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=regularizers.l2(0.005),
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=regularizers.l2(0.005),
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.05,
               recurrent_dropout=0.0,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False))
model.add(Dropout(0.1))
model.add(Dense(32,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.005),
                activity_regularizer=regularizers.l2(0.005)))
model.add(Dense(NUM_LABELS, activation='softmax'))

#model.compile(loss='categorical_crossentropy', optimizer='adam')
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

# save best-performing model parameters during training
chk = ModelCheckpoint(filepath='LSTM_stock2vec_1.hdf5',
                      monitor='loss',
                      verbose=2,
                      save_best_only=True,
                      save_weights_only=False,
                      mode='auto',
                      period=1)

train_generator = train_gen()
val_generator = validation_gen()

model.fit_generator(train_generator,
                    epochs=50,
                    steps_per_epoch=250000, # alternatively use utils.sequence class
                    verbose=2,
                    callbacks=[chk],
                    validation_data=val_generator,
                    validation_steps=250000,
                    class_weight=None,
                    max_queue_size=256,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True)
