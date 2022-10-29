import time

import numpy as np
import open3d as o3d

import metrics
from funcs_GRAPHICAL_MODEL import feature_extraction_v3
import random


def remove_outlier(pcd, label):
    pts = np.array(pcd.points)
    cond = np.where(label <=1)[0]
    new_pcd = o3d.geometry.PointCloud()

    new_pcd.points = o3d.utility.Vector3dVector(pts[cond])
    return new_pcd, label[cond]


def Experiments_of_Noise_Level(noise):
    print("Experiments of Noise Level (GRAPHICAL_MODEL)!")
    print("Current Noise Level:", noise)

    num_experiments = 5
    start_time=time.perf_counter()
    positions = np.random.uniform(-0.5, 0.5, (2, num_experiments))
    FNR_vec = np.zeros((num_experiments,))
    FPR_vec = np.zeros((num_experiments,))
    for i in range(num_experiments):
        print("Current i is: ", i)
        pcd = o3d.io.read_point_cloud('../EM_DRG_result/Noise_0p05/data3/pcd_' + str(i) + '.pcd')
        label = np.load('../EM_DRG_result/Noise_0p05/data3/label_' + str(i) + '.npy')
        pcd, label = remove_outlier(pcd, label)

        est_label = feature_extraction_v3(pcd, label,w_min=0.55, w_max=0.67)
        np.save('../EM_DRG_result/Noise_0p05/result3/GRAPHICAL_MODEL_est_label_' + str(i), est_label)
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
    end_time=time.perf_counter()
    print("running time",end_time-start_time)
    print("FPR_vec", FPR_vec)
    print("Average FPR", np.mean(FPR_vec))
    print("FPR_vec", FNR_vec)
    print("Average FPR", np.mean(FNR_vec))
    np.save('../EM_DRG_result/Noise_0p05/result3/GRAPHICAL_MODEL_FPR', FPR_vec)
    np.save('../EM_DRG_result/Noise_0p05/result3/GRAPHICAL_MODEL_FNR', FNR_vec)


if __name__ == '__main__':
    # random seed
    seed = np.random.seed(1)
    random.seed(seed)

    Experiments_of_Noise_Level(noise=0.05)
