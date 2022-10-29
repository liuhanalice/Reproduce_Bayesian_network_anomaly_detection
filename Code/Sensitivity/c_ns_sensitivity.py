from Generate_Defects_v2 import depression_circle_v2
import metrics
from my_funcs import *
from Directed_Graphical_Model import EM_Directed_Graphical_Model
import sys
from sklearn.metrics import confusion_matrix


def c_ns_experiment(c_ns):
    print("value of c_ns", c_ns)

    num_experiments = 6
    positions = np.random.uniform(-0.5, 0.5, (2, num_experiments))
    final_confusion_matrix = np.zeros((3, 3))
    for i in range(num_experiments):
        print("Current i is: ", i)
        cur_pos = positions[:, i: i + 1]
        options = EM_options(0.0004, 0.01, 5, 4, 1.4, cur_pos, bg_std_depth=0.10, step=-0.4, spline_flag=False)
        print(options.bg_k, options.bg_std_depth)
        pcd, label, _, _ = depression_circle_v2(options, num_p=150)
        est_label, distance_label = EM_Directed_Graphical_Model(pcd, label, NN=35, n_knot=10, epoch_EM=16, C_f=1 / 4, C_ns=c_ns, visualization=0)
        confusion_mat = confusion_matrix(label, est_label)
        print("confusion mat:", confusion_mat)
        final_confusion_matrix += confusion_mat
        np.save('./parameter sensitivity/c_ns_sensitivity_' + str(np.round(c_ns, 4)) + '_epoch_' + str(i), confusion_mat)
    np.save('./parameter sensitivity/c_ns_sensitivity_' + str(np.round(c_ns, 4)), final_confusion_matrix)


def data_analysis():
    c_ns_list = [1 / 36, 1 / 48, 1 / 64, 1 / 80]
    for i in range(len(c_ns_list)):
        confusion_mat = np.load('./parameter sensitivity/c_ns_sensitivity_' + str(np.round(c_ns_list[i], 4)) + '.npy')
        print("mat:", confusion_mat)
        fpr = (confusion_mat[0, 2] + confusion_mat[1, 2]) / (
                np.sum(confusion_mat[0, :]) + np.sum(confusion_mat[1, :]))
        fnr = (confusion_mat[2, 0] + confusion_mat[2, 1]) / np.sum(confusion_mat[2, :])
        print("fpr:", fpr)
        print("fnr:", fnr)


if __name__ == '__main__':
    # random seed
    np.random.seed(1)
    random.seed(1)

    # c_ns_list = 1 / 24
    # c_ns_experiment(c_ns_list)
    data_analysis()
