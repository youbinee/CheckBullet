__author__ = "YOUBIN JEON"
__copyright__ = "YOUBIN JEON 2023"
__version__ = "1.0.0"
__license__ = "MNC lab"

# ============================================================
# File name: cbcallback.py
# Versions: v1.0.0
# 
# Date     Description
# 230224   [Init] development(on_epoch_end, on_epoch_begin)
# ============================================================

import tensorflow as tf
import datetime as dt
import time
import requests
import json
import os
import sys
import numpy as np
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from multiprocessing import Process, Manager
import threading

sys.path.append('/root/CheckBullet')
from cb_controller.Controller import controller
from cb_checkpoint.Checkpoint import checkpoint

class CBCallback(tf.keras.callbacks.Callback):
    #============================================================
    # [Init] CBCallback
    #============================================================
    def __init__(self):
        super(CBCallback, self).__init__()
        self.times = 0
        self.ckpt_interval = 0
        self.avg_epoch_time = 0
        self.avg_total_time = 0
        self.l_incre_count = 0
        self.s_incre_count = 0
        self.res_merge_point = 0
        self.is_merge = False
        self.last_start = 0
        self.ts_snap = 0
        self.tf_snap = 0
        self.loss = []

    #=============================================================
    # [After] On_epoch_end
    #=============================================================
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys()) # ['loss', 'val_acc', 'val_loss', 'acc']
        e_time = time.time() - self.epoch_time_start # calculate epoch time
        self.times = self.times + e_time # training runtime

        cont = controller() # setting Controller
        configs = json.load(open("config.json", 'r'))

        epochs = configs['training']['epochs']
        save_dir = configs['model']['save_dir']

        print("==================== [CB_Training][START] on_epoch_end ================")
        
        # [GPU] set data
        data = {
                "model": 1,
                "e_time": e_time,
                "epochs": epochs
                }

        #======================================================
        # [GPU] If epoch = 1, full checkpoint
        #======================================================
        if epoch == 0: # (end)init epoch = 1
            
            #==========================================================
            # [GPU -> Controller] Call Checkpoint Interval
            #==========================================================
            res_controller = cont.set_interval(data)
            self.ckpt_interval = res_controller['ckpt_interval']
            self.avg_epoch_time = res_controller['avg_epoch_time']
            self.avg_total_time = res_controller['avg_total_time']

            print("========== [CB_Training] First Model Full Checkpoint")
            
            #===========================================================
            # [GPU -> CPU] model full checkpoint
            #===========================================================
            self.ts_snap = time.time()
            p_cpu = threading.Thread(target=self.save_full, args=(data, epoch, save_dir, self.model))
            p_cpu.daemon = True
            p_cpu.start()
            p_cpu.join()
            self.tf_snap = time.time()

        #======================================================
        # [GPU] if epoch >= 2, partial checkpoint
        #======================================================
        else:
            print("========== [CB_Training] Model Weights Incremental Checkpoint")
            print("========== [CB_Training][EPOCH] current epoch: ", epoch+1)

            #============================================================
            # [GPU] Checkpoint Interval
            #============================================================
            if epoch % self.ckpt_interval == 0:
                self.loss.append(logs.get("loss"))

                self.l_incre_count = self.l_incre_count + 1
                    
                #=========================================================
                # [GPU -> Controller] merging point
                #=========================================================
                if self.l_incre_count == 1:
                    self.last_start = epoch+1
                    t_snap = self.tf_snap - self.ts_snap
                    self.res_merge_point = cont.set_merging_point(self.ckpt_interval, t_snap)
                    
                if self.l_incre_count % self.res_merge_point == 0:
                    self.is_merge = True
                else:
                    self.is_merge = False
                
                #=========================================================
                # [CPU -> CPU] Call weight partial checkpoint
                #=========================================================
                p_cpu = Process(target=self.save_only_weights, args=(data, epoch, self.ckpt_interval, save_dir, self.model.get_weights(), self.is_merge, self.res_merge_point, self.last_start, self.loss))
                p_cpu.daemon = True
                p_cpu.start()
                p_cpu.join()

                #===========================================================
                # [GPU -> CPU] model full checkpoint
                #===========================================================
                #p_cpu = threading.Thread(target=self.save_full, args=(data, epoch, save_dir, self.model))
                #p_cpu.daemon = True
                #p_cpu.start()
                #p_cpu.join()
            else:
                print("========== [CB_Training] No Checkpoint Interval")
        print("==================== [CB_Training][END] on_epoch_end ================")
    
    #================================================================
    # [Before] On_epoch_begin
    #================================================================
    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys()) # ['loss', 'val_acc', 'val_loss', 'acc']
        self.epoch_time_start = time.time() # set timer
         
        #======================================================
        # [GPU] if epoch = 1, check re-training
        #======================================================
        if epoch == 0:
            configs = json.load(open("config.json", 'r'))

            epochs = configs['training']['epochs']
            save_dir = configs['model']['save_dir']

            file_name = []
            for file in os.listdir(save_dir):
                if file.startswith("merge"):
                    file_name.append(file)

            # call weights file list
            for file in os.listdir(save_dir):
                if file.startswith("w-only"):
                    file_name.append(file)

            file_name.sort(reverse=True)
            w_count = 1
            w_list = []
        
            #==================================================================
            # [GPU] if re-training  = existing saved checkpoint
            #==================================================================
            weights_origin = []
            if len(file_name) == 0:
                print("========== [cb_training][Start] no merge file -> start first training ==========")

            elif len(file_name) == 1 or file_name[0].startswith('merge'):
                save_fname = os.path.join(save_dir, file_name[0])
                weights_origin = np.load(save_fname, allow_pickle=True)
                
                m_weights = self.model.get_weights()

                for w_array in weights_origin:
                    w_list.append('w_array_{}'.format(w_count))
                    locals()['w_array_{}'.format(w_count)] = w_array
                    w_count = w_count + 1
                w_list.sort()

                for i in range(1,len(w_list)): # call only element
                    e_num = 'w_array_{}'.format(i)
                    for j in range(len(weights_origin[e_num])): # call element's variables
                        if isinstance(weights_origin[e_num][j], (list, np.ndarray)) == True:
                            for k in range(len(weights_origin[e_num][j])):
                                m_weights[i-1][j][k] = weights_origin[e_num][j][k]

                        else:
                            m_weights[i-1][j] = weights_origin[e_num][j]

                dequantized_value = m_weights
                q_count = len(weights_origin['w_array_{}'.format(len(w_list))])
                for n in range(q_count):
                    scale = weights_origin['w_array_{}'.format(len(w_list))][n]
                    dequantized_value[n] = m_weights[n] / scale
                
                self.model.set_weights(dequantized_value)
                self.model.save_weights(save_fname)

            elif len(file_name) > 1 and file_name[0].startswith('w-only'):
                for m in range(1,2):
                    save_fname = os.path.join(save_dir, file_name[m])
                    weights_new = np.load(save_fname, allow_pickle=True)

                    m_weights = self.model.get_weights()

                    #======================================================
                    # [GPU] Comapre file
                    #======================================================
                    if m <= 1:
                        save_fname = os.path.join(save_dir, file_name[0])
                        weights_origin = np.load(save_fname, allow_pickle=True)

                        for w_array in weights_origin:
                            w_list.append('w_array_{}'.format(w_count))
                            locals()['w_array_{}'.format(w_count)] = w_array
                            w_count = w_count + 1
                        w_list.sort()
                        
                        for i in range(1,len(w_list)): # call only element
                            e_num = 'w_array_{}'.format(i)
                            for j in range(len(weights_origin[e_num])): # call element's variables
                                if isinstance(weights_origin[e_num][j], (list, np.ndarray)) == True:
                                    for k in range(len(weights_origin[e_num][j])):
                                        if np.isnan(weights_origin[e_num][j][k]):
                                            m_weights[i-1][j][k] = weights_new[e_num][j][k]
                                        elif weights_origin[e_num][j][k] != weights_new[e_num][j][k]:
                                            m_weights[i-1][j][k] = weights_origin[e_num][j][k]
                                        else:
                                            m_weights[i-1][j][k] = weights_origin[e_num][j][k]

                                else:
                                    if np.isnan(weights_origin[e_num][j]):
                                        m_weights[i-1][j] = weights_new[e_num][j]
                                    elif weights_origin[e_num][j] != weights_new[e_num][j]:
                                        m_weights[i-1][j] = weights_origin[e_num][j]
                                    else:
                                        m_weights[i-1][j] = weights_origin[e_num][j]
                    
                    #======================================================
                    # [GPU] Compare file > 2
                    #======================================================
                    else:
                        for i in range(1,len(w_list)): # call only element
                            e_num = 'w_array_{}'.format(i)
                            for j in range(len(weights_new[e_num])): # call element's variables
                                if isinstance(weights_new[e_num][j], (list, np.ndarray)) == True:
                                    for k in range(len(weights_new[e_num][j])):
                                        if np.isnan(weights_new[e_num][j][k]) == False:
                                            m_weights[i-1][j][k] = weights_new[e_num][j][k]
                                else:
                                    if np.isnan(weights_new[e_num][j]) == False:
                                        m_weights[i-1][j] = weights_new[e_num][j]
                    
                dequantized_value = m_weights
                q_count = len(weights_origin['w_array_{}'.format(len(w_list))])
                for n in range(q_count):
                    scale = weights_origin['w_array_{}'.format(len(w_list))][n]
                    dequantized_value[n] = m_weights[n] / scale
 
                save_fname = os.path.join(save_dir, 'saved_weights.h5')
                self.model.set_weights(dequantized_value)
                self.model.save_weights(save_fname)
               
        self.epoch_time_end = time.time()
        print("========== [cb_training][Duration] Recovery Time: ", self.epoch_time_end-self.epoch_time_start)

    #===================================================================
    # [Save] Full Checkpoint
    #===================================================================
    def save_full(self, data, epoch, save_dir, cpu_model):
        print("========== [CB_Training][PID] full checkpoint CPU: ", os.getpid())
        ckpt = checkpoint()
        full_res = ckpt.save_f_model(data, epoch, save_dir, cpu_model)

    #===================================================================
    # [Save] Weights Checkpoint
    #===================================================================
    def save_weights(self, data, epoch, save_dir, cpu_model):
        print("========== [CB_Training][PID] weights checkpoint CPU: ", os.getpid())
        ckpt = checkpoint()
        incre_res = ckpt.save_w_model(data, epoch, save_dir, cpu_model)

    #===================================================================
    # [Save] Weights-only Checkpoint
    #===================================================================
    def save_only_weights(self, data, epoch, interval, save_dir, weights, is_merge, merge_point, last_start, loss):
        ckpt = checkpoint()
        incre_res = ckpt.save_only_weights(data, epoch, interval, save_dir, weights, is_merge, merge_point, last_start, loss)

