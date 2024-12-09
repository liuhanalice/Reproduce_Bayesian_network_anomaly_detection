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
def SINGLE_EXPERIMENT():
    defect_pos = np.array([[-0.0], [-0.0]])
    options = EM_options(0.0004, 0.01, 3, 2.4, 1.5, defect_pos, bg_std_depth=0.10, step=-0.4, spline_flag=False)
    print(options.bg_k, options.bg_std_depth)
    pcd, label, _, _ = depression_circle_v2(options, num_p=150)

    Visualization_pcd(np.asarray(pcd.points[:]).astype(np.float64), label, scale=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est_label = EM_Directed_Graphical_Model(pcd, label, NN=35, n_knot=10, epoch_EM=16, C_f=1 / 4, C_ns=1 / 48)
    Visualization_pcd(np.asarray(pcd.points[:]).astype(np.float64), est_label, scale=1)
    confs_mat = metrics.ConfusionMatrix(number_of_labels=3)
    confs_mat.count_predicted_batch_hard(label, est_label)
    print("confs_mat:", confs_mat.confusion_matrix)
    confusion_matrix = confs_mat.confusion_matrix
    FPR = (confusion_matrix[0, 1] + confusion_matrix[2, 1]) / (
            np.sum(confusion_matrix[0, :]) + np.sum(confusion_matrix[2, :]))
    FNR = (confusion_matrix[1, 0] + confusion_matrix[1, 2]) / np.sum(confusion_matrix[1, :])
    print("FPR,FNR", FPR, FNR)


if __name__ == '__main__':
    # random seed
    np.random.seed(1)
    random.seed(1)

    # single test experiment
    SINGLE_EXPERIMENT()
