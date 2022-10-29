from funcs_J_Nondestruct_2017 import moving_least_squares
import numpy as np
import open3d as o3d
from my_funcs import Initialization_h_m_MS


def window_plane_V1(pcd, label, thr=1.0):
    points = np.asarray(pcd.points[:]).astype(np.float64)
    h, m = Initialization_h_m_MS(points)
    MLS_pcd, inlier_label = moving_least_squares(pcd, 1.4, 30)
    cond_inlier = np.where(inlier_label == 1)[0]
    MLS_points = np.asarray(MLS_pcd.points[:]).astype(np.float64)
    dist = np.abs(np.dot(h.T, MLS_points.T).T - m)
    defect_cond = np.where(dist > thr)[0]
    Final_label = np.ones(inlier_label.shape, np.uint8) * 2
    Final_label[cond_inlier] = 0
    Final_label[cond_inlier[defect_cond]] = 1
    return Final_label
