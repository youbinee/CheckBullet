__author__ = "YOUBIN JEON"
__copyright__ = "YOUBIN JEON 2023"
__version__ = "1.0.0"
__license__ = "MNC lab"

# ============================================================
# File name: Interval.py
# Versions: v1.0.0
#
# Date     Description
# 230228   [Init] development(set_interval)
# ============================================================

import json
import math

class controller:
    #======================================================
    # [Init] Contoller
    #======================================================
    def __init__(self):
        self.avg_epoch_time = 10
        self.avg_total_time = 440
        self.f_ckpt_over = 0.45
        self.ckpt_partial_cost = 5
        self.re_threshold = 12
        self.re_full_cost = 0.25
        self.re_partial_cost = 5
        self.ckpt_interval = 1

    #======================================================
    # [Cal] Calculate Checkpoint Interval
    #======================================================
    def set_interval(self, data):
        print("========== [CB_Controller][START] Set Interval ==========")

        epochs = data['epochs']
        e_time = data['e_time']

        mtbf_time = (e_time**2)/(2*self.f_ckpt_over)
        self.ckpt_interval = math.ceil((e_time*epochs)/mtbf_time)

        res_data = {
                "ckpt_interval": self.ckpt_interval,
                "avg_epoch_time": self.avg_epoch_time,
                "avg_total_time": self.avg_total_time}
        
        print("========== [CB_Controller][END] Set Interval ==========")
        return res_data

    #======================================================
    # [Cal] Calculate Merging Point
    #======================================================
    def set_merging_point(self, interval, snap):
        print("========== [CB_Controller][START] Set Merging Point  ==========")
        
        mc_point = math.floor((self.ckpt_interval * self.avg_epoch_time)-snap+self.ckpt_partial_cost/self.ckpt_partial_cost)
        mr_point = math.floor((self.re_threshold-self.re_full_cost+self.re_partial_cost)/self.re_partial_cost)
        
        m_point = min(mc_point, mr_point)
        print("========== [CB_Controller][END] Set Merging Point ==========")
        return m_point

