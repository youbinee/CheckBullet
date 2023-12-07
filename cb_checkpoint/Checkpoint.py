__author__ = "YOUBIN JEON"
__copyright__ = "YOUBIN JEON 2023"
__version__ = "1.0.0"
__license__ = "MNC lab"

# ============================================================
# File name: cbcallback.py
# Versions: v1.0.0
# 
# Date     Description
# 230407   [Init] development(create full checkpoint)
# ============================================================

import os
import sys
import json
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model

class checkpoint():
    #===========================================================
    # [Init] Checkpoint
    #===========================================================
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        super().__init__()
        self.model = Sequential()
        self.save_point = 0
        self.merge = 0
        self.test_count = 0
        self.test_total_count = 0

    #=======================================================
    # [Define] The weights of the current epoch for each array.
    #=======================================================
    def list_index(self, weights):
        w_count = 1
        w_list = []
        for w_array in weights:
            w_list.append('w_array_{}'.format(w_count))
            locals()['w_array_{}'.format(w_count)] = w_array
            w_count = w_count + 1
        return w_list

    #===========================================================
    # [Save] Full Checkpoint
    #===========================================================
    def save_f_model(self, data, epoch, save_dir, cpu_model):
        print("========== [CB_Checkpoint][START][CPU] Save full model ==========")
        
        self.model = cpu_model
        save_fname = os.path.join(save_dir, 'model-%s-e%s.h5' % (str(data['model']), str(epoch+1)))
        self.model.save(filepath=save_fname, overwrite=True)
        
        print("========== [CB_Checkpoint][END][CPU] Save full model ==========")

    #========================================================
    # [Save] Save weights model
    #========================================================
    def save_w_model(self, data, epoch, save_dir, cpu_model):
        print("========== [CB_Checkpoint][START][CPU] Save model weight ==========")
        
        self.model = cpu_model
        save_fname = os.path.join(save_dir, 'weight-1-e%s.h5' % (str(epoch+1)))
        cpu_model.save_weights(filepath=save_fname, overwrite=True)

        print("========== [CB_Checkpoint][END][CPU] Save model weight ==========")

    #=========================================================
    # [Save] Merging Weights
    #=========================================================
    def merge_weights(self, data, epoch, save_dir, interval, merge_point, weights, loss):
        print("========== [CB_Checkpoint][START][CPU] Merge model weight ==========")
        
        m_weights = weights
        start = epoch+1-interval
        stop = (epoch+1)-(merge_point*interval)
        load_count = 1
        w_list = []
        file_name = []
        m_count = 0
        epoch_name = ""

        w_list = self.list_index(weights)

        # Call weights file list
        for file in os.listdir(save_dir):
            if file.startswith("w-only"):
                file_name.append(file)
        file_name.sort(reverse=False)

        #=====================================================
        # previous weights merging
        #=====================================================
        for m in range(start,stop,-(interval)):
            if m_count == 1:
                break
            
            if m >= 1 and m <= 9:
                epoch_name = "0" + str(m)
            else:
                epoch_name = str(m)

            m_count = m_count + 1

            # [Merge] Load weights saved in previous interval
            save_fname_l = os.path.join(save_dir, 'w-only-model-%s-e%s.npz' % (str(data['model']), epoch_name))
            pre_weights = np.load(save_fname_l, allow_pickle=True)

            # [Merge] Merge by comparing previous weights based on current weights
            for i in range(len(w_list)): # call only element
                for j in range(len(weights[i])): # call element's variables
                    if isinstance(weights[i][j], (list, np.ndarray)) == True: # k element in  array
                        for k in range(len(weights[i][j])):
                            if np.isnan(weights[i][j][k]):
                                m_weights[i][j][k] = pre_weights['w_array_{}'.format(i+1)][j][k]
                            else:
                                m_weights[i][j][k] = weights[i][j][k]

                    else:
                        if np.isnan(weights[i][j]):
                            m_weights[i][j] = pre_weights['w_array_{}'.format(i+1)][j]
                        else:
                            m_weights[i][j] = weights[i][j]
            sys.stdout.flush()

        #===================================================
        # Save Merged file
        #===================================================
        try:
            if epoch+1 >= 1 and epoch+1 <= 9:
                epoch_name = "0" + str(epoch+1)
            else:
                epoch_name = str(epoch+1)

            save_fname = os.path.join(save_dir, 'merge-model-%s-e%s.npz' % (str(data['model']), epoch_name))
            np.savez(save_fname, **{name:value for name, value in zip(w_list, m_weights)})
        except:
            print("========== [CB_Checkpoint][ERROR] save new merge weights. Please check the merging weights")

        #===================================================
        # Delete previously saved weights file
        #===================================================
        for m in range(epoch+1,stop,-(interval)):
            if m >= 1 and m <= 9:
                epoch_name = "0" + str(m)
            else:
                epoch_name = str(m)

            save_fname_l = os.path.join(save_dir, 'w-only-model-%s-e%s.npz' % (str(data['model']), epoch_name))
            os.remove(save_fname_l)

        #===================================================
        # Delete previously saved merging files
        #===================================================
        m_interval = merge_point * interval
        remove_file = epoch+1-m_interval

        if remove_file >= 1 and remove_file <= 9:
            epoch_name = "0" + str(remove_file)
        else:
            epoch_name = str(remove_file)

        if remove_file > 1:
            save_fname_l = os.path.join(save_dir, 'merge-model-%s-e%s.npz' % (str(data['model']), epoch_name))
            os.remove(save_fname_l)

        print("========== [CB_Checkpoint][END][CPU] Merge model weight ==========")

    #===============================================
    # [Save] weights-only Checkpoint
    #==============================================
    def save_only_weights(self, data, epoch, interval, save_dir, weights, is_merge, merge_point, last_start, loss):
        print("========== [CB_Checkpoint][START][CPU] Save incremental checkpoint ==========")
       
        epoch_name = ""
        if epoch+1 >= 1 and epoch+1 <= 9:
            epoch_name = "0" + str(epoch+1)
        else:
            epoch_name = str(epoch+1)

        avg_weights = []
        save_fname = os.path.join(save_dir, 'w-only-model-%s-e%s.npz' % (str(data['model']), epoch_name))

        w_list = []
        w_list = self.list_index(weights)

        #=======================================================
        # [Add] Weights only Quantization
        #=======================================================
        q_weights = weights
        q_scale = []
        i = 0
        for name, value in zip(w_list, weights):
            bits = 6
            scale = (2 ** bits - 1) / (np.max(np.abs(value)) - np.min(np.abs(value)))
            q_scale.append(scale)

            quantized_value = np.round(value * scale + 0.5)
            q_weights[i] = quantized_value
            i = i + 1

        #====================================================
        # Save weight tensor
        #====================================================
        if (epoch+1) == (interval+1):
            q_weights[len(w_list)-1] = q_scale
            np.savez(save_fname, **{name:value for name, value in zip(w_list, q_weights)})

        else:
            print("========== [CB_Checkpoint][SAVE] Incremental weights checkpoint in CPU")
            u_epoch = (epoch+1) - last_start
                
            if u_epoch == 0:
                q_weights[len(w_list)-1] = q_scale
                np.savez(save_fname, **{name:value for name, value in zip(w_list, q_weights)})
            else:
                if (u_epoch+1) % (interval*merge_point) == 1:
                    file_name = 'merge-model-%s-e%s.npz'
                else:  
                    file_name = 'w-only-model-%s-e%s.npz'
                
                if (epoch+1-interval) >= 1 and (epoch+1-interval) <= 9:
                    epoch_name = "0" + str(epoch+1-interval)
                else:
                    epoch_name = str(epoch+1-interval)

                save_fname_l = os.path.join(save_dir, file_name % (str(data['model']), epoch_name))
                pre_weights = np.load(save_fname_l, allow_pickle=True)
                com_weights = weights

                #============================================================
                # [SAVE] Make partial q_weights
                #============================================================
                n_weights = q_weights
                for i in range(len(w_list)): # call only element
                    for j in range(len(q_weights[i])): # call element's variables
                        if isinstance(q_weights[i][j], (list, np.ndarray)) == True:
                            for k in range(len(q_weights[i][j])):
                                self.test_total_count += 1
                                if q_weights[i][j][k] == pre_weights['w_array_{}'.format(i+1)][j][k]: # q_weights == pre_weights
                                    n_weights[i][j][k] = None
                                    self.test_count += 1
                                    sys.stdout.flush()
                                else: # q_weights != pre_weights
                                    n_weights[i][j][k] = q_weights[i][j][k]
                        else:
                            self.test_total_count += 1
                            if q_weights[i][j] == pre_weights['w_array_{}'.format(i+1)][j]:
                                n_weights[i][j] = None
                                self.test_count += 1
                            else:
                                n_weights[i][j] = q_weights[i][j]
                
                n_weights[len(w_list)-1] = q_scale
                np.savez(save_fname, **{name:value for name, value in zip(w_list, n_weights)})

                sys.stdout.flush()
        
        #================================================
        # Merging point
        #================================================
        if is_merge:
           # incremental chekpoint merging
           self.merge_weights(data, epoch, save_dir, interval, merge_point, n_weights, loss)
           
        print("========== [CB_Checkpoint][END][CPU] Save incremental checkpoint ==========")

    def retrain_merge(self, save_dir, len_file):
        print("========== [CB_Checkpoint][START][CPU] Re-training model merge  ==========")
        file_name = []
        
        # call weights file list
        for file in os.listdir(save_dir):
            if file.startswith("w-only"):
                file_name.append(file)

        # call merge file list
        for file in os.listdir(save_dir):
            if file.startswith("merge"):
                file_name.append(file)
        file_name.sort(reverse=True)

        print("========== [CB_Checkpoint][END][CPU] Re-training model merge  ==========")
