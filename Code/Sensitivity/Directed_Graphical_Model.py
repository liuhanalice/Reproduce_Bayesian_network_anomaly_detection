import numpy as np
import open3d as o3d
import time
from Generate_Defects_v2 import depression_circle_v2
import metrics
from my_funcs import *
import sys


def Parameter_update(epoch_i, h, m, spline_paras, tensor_diff, miu, points, weight0, weight1, alpha,
                     neighbor, sigma_s_sq, n_knot, C_f, C_ns, visualization, num_H):
    new_sigma_sq, delta_m, B_b, R, pts = Update_sigma_sq(weight0, points, h, m, spline_paras, n_knot)
    # update weight0 and weight1
    new_weight0, new_weight1, new_weight2 = Posterior_Prob(tensor_diff, new_sigma_sq, sigma_s_sq, neighbor, miu, points,
                                                           h, m,
                                                           delta_m, alpha, max_iter=8, C_f=C_f, C_ns=C_ns)

    new_label = Label_Assignment(new_weight0, new_weight1)
    new_miu_diff = Update_miu_diff(new_weight0, new_weight1, new_weight2, tensor_diff, neighbor)
    new_sigma_s_sq = Update_sigma_s(new_weight0, new_weight1, new_weight2, tensor_diff, miu, neighbor)
    new_alpha = Update_alpha(np.sum(new_weight0), np.sum(new_weight1), np.sum(new_weight2))
    new_h_p, new_m_p, new_paras = Update_h_m_paras_Linear(epoch_i, pts, new_weight0, B_b, spline_paras.shape[0], num_H)
    new_h = np.dot(R, new_h_p)
    new_m = new_m_p
    if epoch_i == 1 and visualization:
        Visualization_pcd(points, new_label, scale=5)
    return new_sigma_sq, new_sigma_s_sq, new_miu_diff, new_alpha, new_weight0, new_weight1, \
           new_h, new_m, new_paras


def EM_Directed_Graphical_Model(pcd, true_label, NN, n_knot, epoch_EM, C_f, C_ns, visualization=False, NN_TV=100, num_H=2, sigma_d=1.2):
    points = np.asarray(pcd.points[:]).astype(np.float64)
    print("Number of points:", points.shape[0])
    print("points:", points)
    print("C_f,C_ns", C_f, C_ns)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    num = points.shape[0]
    K = np.zeros((num, 3, 3), np.float64)
    for i in range(num):
        K[i] = np.eye(3).astype(np.float64)
    Neighbor_Inf = Neigbor_Information(pcd_tree, NN, points)
    Neighbor_Inf_TV = Neigbor_Information(pcd_tree, NN_TV, points)
    K_f = Computation_Of_K_Converged(K, NN_TV, points, Neighbor_Inf_TV, sigma_d, MaxIter=5)
    tensor_diff = Tensor_Diff_K(K_f, Neighbor_Inf)
    outlier_rate = 1 / 3
    defect_rate = (1 - outlier_rate) / 2
    weight0 = (1 - outlier_rate - defect_rate) * np.ones((num,), np.float64)
    weight1 = defect_rate * np.ones((num, 2), np.float64)
    h, m = Initialization_h_m_MS(points)
    print("initial h and m are:", h.T, m)
    spline_paras = np.zeros(((n_knot - 4) ** 2, 1), np.float64)
    alpha = np.array([defect_rate, defect_rate, outlier_rate])
    miu = 0
    sigma_s_sq = 0.3
    for epoch in range(epoch_EM):
        print("current epoch is ", epoch)
        print("Cf in current epoch:", C_f)
        sigma_sq, sigma_s_sq, miu, alpha, weight0, weight1, h, m, spline_paras = Parameter_update(
            epoch_EM - epoch, h, m, spline_paras, tensor_diff, miu, points, weight0, weight1, alpha, Neighbor_Inf,
            sigma_s_sq, n_knot, C_f, C_ns, visualization, num_H)
        print("estimated parameters in current epoch", sigma_sq, sigma_s_sq, miu, alpha,
              h, m, spline_paras.T)
    label = Label_Assignment(weight0, weight1)

    return label
