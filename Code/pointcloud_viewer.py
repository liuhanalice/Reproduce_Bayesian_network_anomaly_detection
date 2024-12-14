import datetime

import numpy as np
import open3d as o3d
import time
from Generate_Defects_v2 import depression_circle_v2
import metrics
from my_funcs import *
from Directed_Graphical_Model import EM_Directed_Graphical_Model
import sys
from sklearn.metrics import confusion_matrix
import warnings

# This is a demo for single experiment
def POINTCLOUD_VIEWER():
    
    pcd =  o3d.io.read_point_cloud( "Reproduce_corner_result/Depth_7_R4/data3/pcd_19.pcd")
    label = np.load("Reproduce_corner_result/Depth_7_R4/data3/label_19.npy")
    est_label =  np.load("Reproduce_corner_result/Depth_7_R4/result3/EM_DRG_est_label_19.npy")

    Visualization_pcd(np.asarray(pcd.points[:]).astype(np.float64), label, scale=1)
    Visualization_pcd(np.asarray(pcd.points[:]).astype(np.float64), est_label, scale=1)

if __name__ == '__main__':
    # random seed
    np.random.seed(1)
    random.seed(1)

    # single test experiment
    POINTCLOUD_VIEWER()
