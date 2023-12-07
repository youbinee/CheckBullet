__author__ = "YOUBIN JEON"
__copyright__ = "YOUBIN JEON 2023"
__version__ = "1.0.0"
__license__ = "MNC lab"

# ============================================================
# File name: model.py
# Versions: v1.0.0
# 
# Date     Description
# 230223   init development
# ============================================================

import os
import sys
import math
import time
import numpy as np
import datetime as dt
import tensorflow as tf
import pandas as pd
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM, CuDNNLSTM, Bidirectional, TimeDistributed, RepeatVector
from keras.models import Sequential, load_model, Model, model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, RMSprop

from core.utils import Timer
from core.cbcallback import CBCallback

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)

class Model():

    #=========================================================================
    # INIT MODEL
    #=========================================================================
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.model = Sequential()

    #=========================================================================
    # Log File
    #=========================================================================
    def log_file(self, dir_name):
        root_logdir = os.path.join(os.curdir, dir_name)
        sub_dir = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join(root_logdir, sub_dir)

    #=========================================================================
    # Load MODEL
    #=========================================================================
    def load_model(self, filepath):
        print('========== [Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    #=========================================================================
    # Load Weights
    #=========================================================================
    def load_weights(self, filepath):
        print('========== [Model] Loading weights from fils %s' % filepath)
        
        self.model.load_weights(filepath)

    #=========================================================================
    # BUILD MODEL
    #=========================================================================
    def build_model(self, configs):
        # set Timer
        timer = Timer()
        timer.start()

        # create model(LSTM, Dense, Dropout)
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            repeat_num = layer['repeat_num'] if 'repeat_num' in layer else None

            if layer['type'] == 'dense':
                with tf.name_scope('layer_Dense'):
                    self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                with tf.name_scope('layer_LSTM'):
                    self.model.add(CuDNNLSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                with tf.name_scope('layer_Dropout'):
                    self.model.add(Dropout(dropout_rate))
            if layer['type'] == 'repeat':
                with tf.name_scope('layer_Repeat'):
                    self.model.add(RepeatVector(repeat_num))
            if layer['type'] == 'time':
                with tf.name_scope('layer_time'):
                    self.model.add(TimeDistributed(Dense(neurons, activation=activation)))
        
        # model compile
        opt = Adam(lr=0.0008)
        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'], metrics=["accuracy"])

        print('========== [Model] Model Compiled')
        timer.stop()

    #=========================================================================
    # Model Training
    #=========================================================================
    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir, save_dir_w, test_gen, is_retrain):
        timer = Timer()
        timer.start()
        print('========== [Model] Training Started')
        
        s_time = time.time()
        model = Model()
        board_log = model.log_file('log')
        board_CB = TensorBoard(log_dir=board_log)

        if is_retrain == 1:
            print('========== [Model] Load Weights Started')
            #json_file = open("./saved_models/json_model-1-e1.h5", "r")
            #l_model_json = json_file.read()
            #json_file.close()
            #self.model = model_from_json(l_model_json)
            #self.model.load_weights('./saved_models/saved_weights.h5')
            #self.model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
        elif is_retrain == 2:
            print('========== [Model] Load Models Started')
            self.model = load_model('./saved_models/model-1-e13.h5')

        # save model & weights
        save_fname = os.path.join(save_dir, '%s-e%s-{epoch:02d}.h5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S'), str(epochs)))
        save_fname_w = os.path.join(save_dir_w, '%s-e%s-{epoch:02d}.h5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S'), str(epochs)))
       
        cbcallbacks = CBCallback()
        callbacks = [ # save model
                board_CB,
                cbcallbacks
        ]

        hist = self.model.fit_generator(
            generator = data_gen,
            steps_per_epoch = steps_per_epoch,
            epochs = epochs,
            callbacks = callbacks,
            validation_data = test_gen,
            workers = 1
        )
        self.model.save_weights(save_fname_w)
        self.model.summary()

        timer.stop()
        return model

    #=========================================================================
    # Evaluate Val_loss
    #=========================================================================
    def evaluate(self, x_test, y_test, batch_size):
        values= self.model.evaluate(x_test, y_test, batch_size)
        return values

    #=========================================================================
    # Model Prediction
    #=========================================================================
    def predict_sequence_full(self, x_data, y_data, window_size):
        print('========== [Model] Predicting Sequences Full')

        curr_frame = x_data[0]
        predicted = []

        for i in range(len(x_data)):
            predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)

        return predicted

    #=========================================================================
    # Log SAVE
    #=========================================================================
    def log_save(self, predict_data, true_data):
        with tf.name_scope('true'):
            true_data = np.reshape(true_data, [-1])

        with tf.name_scope('predict'):
            predict_data = np.reshape(predict_data, [-1]) # -1, 1

        diff_data = []
        for j in zip(true_data, predict_data):
            diff_data.append(j[0] - j[1])

        diff_data = np.array(diff_data)
        # print('diff_data : %s' % diff_data)

        result_dataframe = pd.DataFrame(list(zip(true_data, predict_data, diff_data)), columns=['true_data', 'predict_data','diff_data'])
        result_dataframe.to_csv('./results/predicted_result.csv', sep=',')
