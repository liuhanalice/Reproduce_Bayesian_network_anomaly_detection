from Generate_Defects_v2 import depression_circle_v2
import metrics
from my_funcs import *
from Directed_Graphical_Model import EM_Directed_Graphical_Model
import sys
from sklearn.metrics import confusion_matrix


def k_experiment(k):
    print("value of k", k)
    num_experiments = 6
    positions = np.random.uniform(-0.5, 0.5, (2, num_experiments))
    final_confusion_matrix = np.zeros((3, 3))
    for i in range(num_experiments):
        print("Current i is: ", i)
        cur_pos = positions[:, i: i + 1]
        options = EM_options(0.0004, 0.01, 5, 4, 1.4, cur_pos, bg_std_depth=0.10, step=-0.4, spline_flag=False)
        print(options.bg_k, options.bg_std_depth)
        pcd, label, _, _ = depression_circle_v2(options, num_p=150)
        est_label, distance_label = EM_Directed_Graphical_Model(pcd, label, NN=k, n_knot=10, epoch_EM=16, C_f=1 / 8,
                                                                C_ns=1 / 48, visualization=0)
        confusion_mat = confusion_matrix(label, est_label)
        print("confusion mat:", confusion_mat)
        final_confusion_matrix += confusion_mat
        np.save('./parameter sensitivity/k_sensitivity_' + str(k) + '_epoch_' + str(i), confusion_mat)
    np.save('./parameter sensitivity/k_sensitivity_' + str(k), final_confusion_matrix)


if __name__ == '__main__':
    # random seed
    np.random.seed(1)
    random.seed(1)
    k_list = 25
    k_experiment(k_list)
