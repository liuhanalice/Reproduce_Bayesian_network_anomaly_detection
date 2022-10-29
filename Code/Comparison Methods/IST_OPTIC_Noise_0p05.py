import time

import numpy as np
import open3d as o3d
from Generate_Defects_v2 import depression_circle_v2
import metrics
from my_funcs import EM_options,Visualization_pcd
from funcs_IST_OPTIC import window_plane_V1
import random


def Experiments_of_Noise_Level(noise):
    print("Experiments of Noise Level (IST_OPTIC_2008)!")
    print("Current Noise Level:", noise)

    num_experiments = 30

    positions = np.random.uniform(-0.5, 0.5, (2, num_experiments))
    FNR_vec = np.zeros((num_experiments,))
    FPR_vec = np.zeros((num_experiments,))
    for i in range(num_experiments):
        print("Current i is: ", i)
        cur_pos = positions[:, i: i + 1]
        # options = EM_options(0.0004, 0.01, depth, depth * 0.8, 1.5, cur_pos, bg_std_depth=0.10, step=-0.35,
        #                      spline_flag=False)
        # print("options", options.bg_k, options.bg_std_depth)
        # pcd, label, _, _ = depression_circle_v2(options, num_p=150)
        pcd = o3d.io.read_point_cloud('../EM_DRG_result/Noise_0p05/data3/pcd_' + str(i) + '.pcd')
        label = np.load('../EM_DRG_result/Noise_0p05/data3/label_' + str(i) + '.npy')
        est_label = window_plane_V1(pcd, label)
        np.save('../EM_DRG_result/Noise_0p05/result3/IST_OPTIC_est_label_' + str(i), est_label)
        confs_mat = metrics.ConfusionMatrix(number_of_labels=3)
        confs_mat.count_predicted_batch_hard(label, est_label)
        print("confs_mat:", confs_mat.confusion_matrix)
        confusion_matrix = confs_mat.confusion_matrix
        FPR = (confusion_matrix[0, 1] + confusion_matrix[2, 1]) / (
                np.sum(confusion_matrix[0, :]) + np.sum(confusion_matrix[2, :]))
        FNR = (confusion_matrix[1, 0] + confusion_matrix[1, 2]) / np.sum(confusion_matrix[1, :])
        print("FPR,FNR", FPR, FNR)
        FPR_vec[i] = FPR
        FNR_vec[i] = FNR


    print("FPR_vec", FPR_vec)
    print("Average FPR", np.mean(FPR_vec))
    print("FPR_vec", FNR_vec)
    print("Average FPR", np.mean(FNR_vec))
    np.save('./EM_DRG_result/Noise_0p05/result3/IST_OPTIC_FPR', FPR_vec)
    np.save('./EM_DRG_result/Noise_0p05/result3/IST_OPTIC_FNR', FNR_vec)


if __name__ == '__main__':
    # random seed
    seed = np.random.seed(1)
    random.seed(seed)

    Experiments_of_Noise_Level(noise=0.05)