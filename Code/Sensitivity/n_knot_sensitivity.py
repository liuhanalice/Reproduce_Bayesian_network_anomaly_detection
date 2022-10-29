import numpy as np
import open3d as o3d
import time
from Generate_Defects_v2 import depression_circle_v2
import metrics
from my_funcs import *
from Directed_Graphical_Model import EM_Directed_Graphical_Model
import sys
from sklearn.metrics import confusion_matrix


def n_knot_experiments(n_knot):
    print("number of knots:", n_knot)
    num_experiments = 6
    positions = np.random.uniform(-0.5, 0.5, (2, num_experiments))
    final_confusion_matrix = np.zeros((3, 3))
    for i in range(num_experiments):
        print("Current i is: ", i)
        cur_pos = positions[:, i: i + 1]
        options = EM_options(0.0004, 0.01, 5, 4, 1.4, cur_pos, bg_std_depth=0.10, step=-0.35, spline_flag=False)
        print(options.bg_k, options.bg_std_depth)
        pcd, label, _, _ = depression_circle_v2(options, num_p=150)
        est_label, distance_label = EM_Directed_Graphical_Model(pcd, label, NN=35, n_knot=n_knot, epoch_EM=16, C_f=1 / 4, num_H=int(n_knot / 9 * 2),
                                                                C_ns=1 / 48, visualization=0)
        confusion_mat = confusion_matrix(label, est_label)
        final_confusion_matrix += confusion_mat
        print("confusion mat:", confusion_mat)
        np.save('./parameter sensitivity/num_knot_sensitivity_' + str(n_knot) + '_epoch_' + str(i), confusion_mat)
    np.save('./parameter sensitivity/num_knot_sensitivity_' + str(n_knot), final_confusion_matrix)



def data_analysis():
    nb_list = [9, 10, 11, 12, 13, 14, 15, 16]
    for i in range(len(nb_list)):
        confusion_mat = np.load('./parameter sensitivity/num_knot_sensitivity_' + str(nb_list[i]) + '.npy')
        print("mat:", confusion_mat)
        FPR = (confusion_mat[0, 1] + confusion_mat[2, 1]) / (
                np.sum(confusion_mat[0, :]) + np.sum(confusion_mat[2, :]))
        FNR = (confusion_mat[1, 0] + confusion_mat[1, 2]) / np.sum(confusion_mat[1, :])
        print("FPR:", FPR)
        print("FNR:", FNR)


if __name__ == '__main__':
    # random seed
    np.random.seed(1)
    random.seed(1)

    # single test experiment
    # n_knot_list = 9
    # n_knot_list = 10
    # n_knot_list = 11
    # n_knot_list = 12
    # n_knot_list = 13
    # n_knot_list = 14
    # n_knot_list = 15
    # n_knot_list = 16
    # n_knot_list = 20

    # n_knot_experiments(n_knot_list)
    data_analysis()
