import numpy as np
import open3d as o3d
import time
from Generate_Defects_v2 import depression_circle_v2
import metrics
from my_funcs import *
from Directed_Graphical_Model import EM_Directed_Graphical_Model
import sys
import argparse
import os
from tqdm import tqdm

# Sample command line: python Reproduce_30Experiments.py --bg_k 0.0004 --x -0.7 --y -1.4 --d 5 --r 2.4 --randrange 0.5

def experiments(bg_k, x, y, d, r, randrange):
    # create data dirs
    folderpath = './Reproduce_corner_result/Depth_' + str(d) + '_R' + str(r) + '_bgk' + str(bg_k) 
    os.makedirs(folderpath, exist_ok=True)
    data_path = os.path.join(folderpath, 'data3')
    result_path = os.path.join(folderpath, 'result3')
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    # start experiment
    print("Experiments of Depth (EM_DIRECTED_GRAPH)!")
    print("depth:", d)
    num_experiments = 30
    positions = np.array([[x], [y]]) + np.random.uniform(-randrange, randrange, (2, num_experiments))
    FNR_vec = np.zeros((num_experiments,))
    FPR_vec = np.zeros((num_experiments,))
    for i in tqdm(range(num_experiments)):
        print("Current i is: ", i)
        cur_pos = positions[:, i: i + 1]
        options = EM_options(bg_k, 0.01, d, r, 1.5, cur_pos, bg_std_depth=0.10, step=-0.35, spline_flag=False)
        print("options", options.bg_k, options.bg_std_depth)
        pcd, label, _, _ = depression_circle_v2(options, num_p=150)

        o3d.io.write_point_cloud(folderpath + '/data3/pcd_' + str(i) + '.pcd', pcd)
        np.save(folderpath + '/data3/label_' + str(i), label)

        est_label = EM_Directed_Graphical_Model(pcd, label, NN=35, n_knot=10, epoch_EM=16, C_f=1 / 4, C_ns=1 / 48)

        np.save(folderpath + '/result3/EM_DRG_est_label_' + str(i), est_label)

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
    print("FPR_vec_Ratio" + str(d), FPR_vec)
    print("Average FPR", np.mean(FPR_vec))
    print("FNR_vec_Ratio" + str(d), FNR_vec)
    print("Average FNR", np.mean(FNR_vec))

    np.save(folderpath + '/result3/EM_DRG_FPR', FPR_vec)
    np.save(folderpath + '/result3/EM_DRG_FNR', FNR_vec)


if __name__ == '__main__':
    # random seed
    seed = np.random.seed(1)
    random.seed(seed)

    parser = argparse.ArgumentParser(description="arguments for generating experiment data, refer to EM_options")

    parser.add_argument('--bg_k', type=float, required=True, help='surface curvature')
    parser.add_argument('--x', type=float, required=True, help='x-coordinate center value')
    parser.add_argument('--y', type=float, required=True, help='y-coordinate center value')
    parser.add_argument('--d', type=float, required=True, help='defect depth')
    parser.add_argument('--r', type=float, required=True, help='defect radius on surface')
    parser.add_argument('--randrange', type=float, required=True, help='random range for 30 experiments')

    args = parser.parse_args()

    experiments(bg_k=args.bg_k, x=args.x, y=args.y, d=args.d, r=args.r, randrange=args.randrange)
