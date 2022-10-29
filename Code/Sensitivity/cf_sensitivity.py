from Generate_Defects_v2 import depression_circle_v2
import metrics
from my_funcs import *
from Directed_Graphical_Model import EM_Directed_Graphical_Model
import sys
from sklearn.metrics import confusion_matrix


def cf_experiment(cf):
    print("value of cf", cf)
    num_experiments = 6
    positions = np.random.uniform(-0.5, 0.5, (2, num_experiments))
    final_confusion_matrix = np.zeros((3, 3))
    for i in range(num_experiments):
        print("Current i is: ", i)
        cur_pos = positions[:, i: i + 1]
        options = EM_options(0.0004, 0.01, 5, 4, 1.4, cur_pos, bg_std_depth=0.10, step=-0.35, spline_flag=False)
        print(options.bg_k, options.bg_std_depth)
        pcd, label, _, _ = depression_circle_v2(options, num_p=150)
        est_label, distance_label = EM_Directed_Graphical_Model(pcd, label, NN=35, n_knot=10, epoch_EM=16, C_f=cf,
                                                                C_ns=1 / 48, visualization=0)
        confusion_mat = confusion_matrix(label, est_label)
        print("confusion mat:", confusion_mat)
        final_confusion_matrix += confusion_mat
        np.save('./parameter sensitivity/cf_sensitivity_' + str(cf) + '_epoch_' + str(i), confusion_mat)
    np.save('./parameter sensitivity/cf_sensitivity_' + str(cf), final_confusion_matrix)


def data_analysis():
    cf_list = [1 / 96, 1 / 48, 1 / 24, 1 / 12, 1 / 8, 1 / 4, 1 / 2]
    for i in range(len(cf_list)):
        confusion_mat = np.load('./parameter sensitivity/cf_sensitivity_' + str(cf_list[i]) + '.npy')
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
    # cf_list = 1/2
    # cf_list = 1/4
    # cf_list = 1 / 8
    # cf_list = 1 / 12
    # cf_list = 1 / 16
    # cf_list = 1 / 20
    # cf_list = 1 / 96
    # cf_experiment(cf_list)
    data_analysis()
