__author__ = "YOUBIN JEON"
__copyright__ = "YOUBIN JEON 2023"
__version__ = "1.0.0"
__license__ = "MNC lab"

# ============================================================
# File name: training.py
# Versions: v1.0.0
# 
# Date     Description
# 230223   init development
# ============================================================

import os
import json
import time
import math
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model

def model_training():
    print("========= [TRAINING] START training =========")
    
    if os.path.isfile('./saved_models/throughput.h5'):
        print("========= [TRAINING] Use existing model =========")
        model = load_model('./saved_models/throughput.h5')
        
    else:
        print("========== [TRAINING] New training model ==========")

        # ======================================================
        # Data Preprocessing
        # ======================================================
        configs = json.load(open('config.json', 'r'))
        if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

        data = DataLoader(
            os.path.join('data', configs['data']['filename']),
            configs['data']['train_test_split'],
            configs['data']['columns']
        )

        # ======================================================
        # Model Training
        # ======================================================
        model = Model()
        model.build_model(configs)

        steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
        
        model.train_generator(
            data_gen = data.generate_train_batch(
                seq_len = configs['data']['sequence_length'],
                batch_size = configs['training']['batch_size'],
                normalise = configs['data']['normalise']
            ),
            epochs = configs['training']['epochs'],
            batch_size = configs['training']['batch_size'],
            steps_per_epoch = steps_per_epoch,
            save_dir = configs['model']['save_dir'],
            save_dir_w = configs['model']['save_dir_w'],
            test_gen = data.get_test_data(
                seq_len = configs['data']['sequence_length'],
                normalise = configs['data']['normalise']
            ),
            is_retrain = 0
        )
        
    print("=========== [TRAINING] END trainig ===========")

if __name__ == '__main__':
    model_training()
