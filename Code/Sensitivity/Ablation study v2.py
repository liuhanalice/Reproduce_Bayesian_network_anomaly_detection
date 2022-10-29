from Generate_Defects_v2 import depression_circle_v2
import metrics
from my_funcs import *
from Directed_Graphical_Model import EM_Directed_Graphical_Model
import sys
from sklearn.metrics import confusion_matrix


def predefined_distance_thresholding(pts, B_b, h, m, paras, weight0, weight1, tau=0.4):
    weight_label = Label_Assignment(weight0, weight1)
    distance_label = weight_label.copy()
    delta = B_b.dot(paras)
    for i in range(distance_label.shape[0]):
        cur_distance = h.T.dot(pts[i].reshape(3, 1)) - (m + delta[i])
        if weight_label[i] <= 1:
            if np.abs(cur_distance) <= tau:
                distance_label[i] = 0
            else:
                distance_label[i] = 1
    return distance_label


def direct_reconstruction(pcd, var_epsilon, n_knot=10, num_H=2):
    points = np.asarray(pcd.points[:]).astype(np.float64)
    num = points.shape[0]
    outlier_rate = 1 / 3
    defect_rate = (1 - outlier_rate) / 2
    weight0 = (1 - outlier_rate - defect_rate) * np.ones((num,), np.float64)
    weight1 = defect_rate * np.ones((num,), np.float64)
    h, m = Initialization_h_m_MS(points)
    spline_paras = np.zeros(((n_knot - 4) ** 2, 1), np.float64)
    _, delta_m, B_b, R, pts = Update_sigma_sq(weight0, points, h, m, spline_paras, n_knot)
    new_h_p, new_m_p, spline_paras = Update_h_m_paras_Linear(1, pts, weight0, B_b, spline_paras.shape[0], num_H, 1)
    distance_label = predefined_distance_thresholding(pts, B_b, new_h_p, new_m_p, spline_paras, weight0, weight1)
    return distance_label


def ablated_method(pcd, label, NN=35, NN_TV=100, sigma_d=1.2, MaxIter=5, epsilon=0.58, var_epsilon=0.4):
    points = np.asarray(pcd.points[:]).astype(np.float64)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    num = points.shape[0]
    K = np.zeros((num, 3, 3), np.float64)
    for i in range(num):
        K[i] = np.eye(3).astype(np.float64)
    points2d = Projection_2D(points)
    print(points2d.shape)
    pcd2d = o3d.geometry.PointCloud()
    points2d = np.hstack((points2d, np.ones((points2d.shape[0], 1))))
    pcd2d.points = o3d.utility.Vector3dVector(points2d)
    pcd_tree2d = o3d.geometry.KDTreeFlann(pcd2d)
    Neighbor_Inf = Neigbor_Information(pcd_tree2d, NN, points2d)
    Neighbor_Inf_TV = Neigbor_Information(pcd_tree, NN_TV, points)
    K_f = Computation_Of_K_Converged(K, NN_TV, points, Neighbor_Inf_TV, sigma_d, MaxIter=MaxIter)
    tensor_diff = Tensor_Diff_K(K_f, Neighbor_Inf)
    deterministic_variations = np.mean(tensor_diff, axis=1)
    estimated_outlier_label = np.zeros_like(label)
    estimated_outlier_label[deterministic_variations >= epsilon] = 1
    inlier_indices = np.where(estimated_outlier_label == 0)[0]

    new_pcd = pcd.select_by_index(inlier_indices)
    new_label = label[inlier_indices]
    # Visualization_pcd(np.asarray(new_pcd.points[:]).astype(np.float64), new_label, scale=1)
    est_label_s = direct_reconstruction(new_pcd, var_epsilon)
    est_label = np.zeros_like(label)
    est_label[estimated_outlier_label == 1] = 2
    interim = est_label[inlier_indices]
    interim[est_label_s == 1] = 1
    est_label[inlier_indices] = interim
    return est_label, K_f


def ablation_study(seed):
    np.random.seed(seed)
    random.seed(seed)
    cur_pos = np.random.uniform(-0.5, 0.5, (2,))
    print(cur_pos)
    options = EM_options(0.0004, 0.01, 5, 4, 1.4, cur_pos, bg_std_depth=0.10, step=-0.35, spline_flag=False)
    print(options.bg_k, options.bg_std_depth)
    pcd, label, _, _ = depression_circle_v2(options, num_p=150)
    print(np.array(pcd.points).shape[0],np.array(pcd.points).shape[0])
    est_label_ablated, K_f = ablated_method(pcd, label)
    est_label, distance_label = EM_Directed_Graphical_Model(pcd, label, NN=35, n_knot=10, epoch_EM=16, C_f=1 / 4,
                                                            C_ns=1 / 48, visualization=0)
    confusion_mat = confusion_matrix(label, est_label)
    print("confusion mat:", confusion_mat)
    np.save('./parameter sensitivity/ablation_study_EM_DRG_' + str(seed), confusion_mat)
    confusion_mat_ablated = confusion_matrix(label, est_label_ablated)
    print("confusion mat ablated:", confusion_mat_ablated)
    np.save('./parameter sensitivity/ablation_study_ablated_' + str(seed), confusion_mat_ablated)


def ablation_study_analysis():
    final_confusion_matrix = np.zeros((3, 3))
    final_confusion_matrix_ablated = np.zeros((3, 3))
    for i in range(6):
        confusion_mat_em = np.load('./parameter sensitivity/ablation_study_EM_DRG_' + str(i) +'.npy')
        confusion_mat_ablated = np.load('./parameter sensitivity/ablation_study_ablated_' + str(i)+'.npy')

        final_confusion_matrix += confusion_mat_em
        final_confusion_matrix_ablated += confusion_mat_ablated

    print(final_confusion_matrix)
    FPR = (final_confusion_matrix[0, 1] + final_confusion_matrix[2, 1]) / (
            np.sum(final_confusion_matrix[0, :]) + np.sum(final_confusion_matrix[2, :]))
    FNR = (final_confusion_matrix[1, 0] + final_confusion_matrix[1, 2]) / np.sum(final_confusion_matrix[1, :])
    print("FPR EM_DRG:", FPR)
    print("FNR EM_DRG:", FNR)

    FPR = (final_confusion_matrix_ablated [0, 1] + final_confusion_matrix_ablated [2, 1]) / (
            np.sum(final_confusion_matrix_ablated [0, :]) + np.sum(final_confusion_matrix_ablated [2, :]))
    FNR = (final_confusion_matrix_ablated [1, 0] + final_confusion_matrix_ablated [1, 2]) / np.sum(final_confusion_matrix_ablated [1, :])
    print("FPR Ablated:", FPR)
    print("FNR Ablated:", FNR)


if __name__ == '__main__':
    # seed = 0
    # seed = 1
    # seed = 2
    # seed = 3
    # seed = 4
    # seed = 5

    # ablation_study(seed)
    ablation_study_analysis()
