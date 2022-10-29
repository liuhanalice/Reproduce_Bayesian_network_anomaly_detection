import numpy as np
import open3d as o3d
import time
from Generate_Defects_v2 import depression_circle_v2
import metrics
from my_funcs import *
from Directed_Graphical_Model import EM_Directed_Graphical_Model
import sys


def Experiments_of_Noise_Level(noise):
    print("Experiments of Noise Level (EM_DIRECTED_GRAPH)!")
    print("Current Noise Level:", noise)
    num_experiments = 30
    positions = np.random.uniform(-0.5, 0.5, (2, num_experiments))
    FNR_vec = np.zeros((num_experiments,))
    FPR_vec = np.zeros((num_experiments,))
    for i in range(num_experiments):
        print("Current i is: ", i)
        cur_pos = positions[:, i: i + 1]
        options = EM_options(0.0004, 0.01, 5, 4, 1.5, cur_pos, bg_std_depth=noise, step=-0.35, spline_flag=False)
        print("options", options.bg_k, options.bg_std_depth)
        pcd, label, _, _ = depression_circle_v2(options, num_p=150)

        o3d.io.write_point_cloud('./EM_DRG_result/Noise_0p15/data3/pcd_' + str(i) + '.pcd', pcd)
        np.save('./EM_DRG_result/Noise_0p15/data3/label_' + str(i), label)

        est_label = EM_Directed_Graphical_Model(pcd, label, NN=35, n_knot=10, epoch_EM=16, C_f=1 / 4,
                                                C_ns=1 / 48)

        np.save('./EM_DRG_result/Noise_0p15/result3/EM_DRG_est_label_' + str(i), est_label)

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

    np.save('./EM_DRG_result/Noise_0p15/result3/EM_DRG_FPR', FPR_vec)
    np.save('./EM_DRG_result/Noise_0p15/result3/EM_DRG_FNR', FNR_vec)


if __name__ == '__main__':
    # random seed
    seed = np.random.seed(1)
    random.seed(seed)

    Experiments_of_Noise_Level(noise=0.15)
