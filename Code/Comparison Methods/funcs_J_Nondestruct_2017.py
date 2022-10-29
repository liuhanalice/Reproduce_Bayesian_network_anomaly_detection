import numpy as np
import open3d as o3d
from my_funcs import Neigbor_Information, Visualization_pcd


def Neigbor_Information_radius(pcd_tree, radius, points):
    Neigbor_Inf = {}
    for i in range(points.shape[0]):
        Neigbor_Inf[i] = Kdtree_Radius_Search(pcd_tree, i, points[i], radius)
    return Neigbor_Inf


def Kdtree_Radius_Search(tree, i, data, radius):
    neighbor = tree.search_radius_vector_3d(data, radius)
    int_vector_index = neighbor[1]
    int_vector_index_arr = np.asarray(int_vector_index)
    voter_index = list(int_vector_index_arr)
    return np.array(voter_index)


def count_outliers(neighbor, num):
    label = np.ones((num,), np.uint8)
    for i in range(num):
        cur_neighbir = neighbor[i]
        if cur_neighbir.shape[0] <= 8:
            label[i] = 0
    return label


def mls(points, inlier_label, neighbor):
    inlier_cond = np.where(inlier_label == 1)[0]
    num_inlier = inlier_cond.shape[0]
    new_points = np.zeros((num_inlier, 3), np.float64)
    normals = np.zeros_like(new_points)
    for i in range(num_inlier):
        cur_index = inlier_cond[i]
        new_points[i], normals[i] = estimation_projection(points[neighbor[cur_index]], points[cur_index])

    return new_points, normals


def weight_matrix(neighbor_pts, p):
    dist = np.linalg.norm(neighbor_pts - p.reshape((1, 3)), axis=1)
    return np.diag(np.exp(-dist ** 2 / (1.2) ** 2))


def estimation_projection(pts, p):
    # order is 2
    P = np.zeros((pts.shape[0], 6), np.float64)
    W = weight_matrix(pts, p)
    for i in range(P.shape[0]):
        P[i] = np.array([1, pts[i, 0], pts[i, 1], pts[i, 0] * pts[i, 1], pts[i, 0] ** 2, pts[i, 1] ** 2], np.float64)
    A = np.dot(P.T, np.dot(W, P))
    B = np.dot(P.T, W)
    y = pts[:, 2].reshape((pts.shape[0], 1))
    alpha = np.dot(np.linalg.inv(A), np.dot(B, y))
    phi_p = np.array([1, p[0], p[1], p[0] * p[1], p[0] ** 2, p[1] ** 2], np.float64)
    normal = np.ones((3,), np.float64)
    d_x = np.array([0, 1, 0, p[1], 2 * p[0], 0], np.float64)
    d_y = np.array([0, 0, 1, p[0], 0, 2 * p[1]], np.float64)
    normal[0] = -d_x @ alpha[:, 0]
    normal[1] = -d_y @ alpha[:, 0]
    normal /= np.linalg.norm(normal)
    return np.array([p[0], p[1], phi_p @ alpha[:, 0]], np.float64), normal


def moving_least_squares(pcd, radius, NN):
    points = np.asarray(pcd.points[:]).astype(np.float64)
    num = points.shape[0]
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    neighbor = Neigbor_Information(pcd_tree, NN, points)
    inlier_label = count_outliers(neighbor, num)
    new_points, normals = mls(points, inlier_label, neighbor)
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    new_pcd.colors = o3d.utility.Vector3dVector(np.zeros(new_points.shape, np.uint8))
    new_pcd.normals = o3d.utility.Vector3dVector(normals)
    # o3d.visualization.draw_geometries([new_pcd], point_show_normal=True)
    return new_pcd, inlier_label


def curvature_computation(cur_p, subpts):
    M = np.zeros((3, 3))
    for i in range(subpts.shape[0]):
        n = (subpts[i] - cur_p).reshape((3, 1))
        M += np.dot(n, n.T)
    M /= subpts.shape[0]
    vals = np.linalg.eigvalsh(M)
    return np.min(vals) / np.sum(vals)


def Curvature_estimation(points, neighbor):
    curvature = np.zeros(points.shape[0], )
    for i in range(points.shape[0]):
        subpts = points[neighbor[i]]
        curvature[i] = curvature_computation(points[i], subpts)
    return curvature


def Region_Growing(points, normals, curvatures, neighbor, ang_thr=np.pi / 16, curv_thr=0.01):
    L = np.ones((points.shape[0],))
    R = []
    while True:
        if np.sum(L) == 0:
            break

        Rc = []
        Sc = []
        argmin_index = np.argmin(curvatures)
        Sc.append(argmin_index)
        Rc.append(argmin_index)
        L[argmin_index] = 0
        curvatures[argmin_index] = np.inf
        while len(Sc):

            cur_seed = Sc[0]
            seed_neigh = neighbor[cur_seed]
            for i in range(seed_neigh.shape[0]):
                if L[seed_neigh[i]] == 1:
                    dot = normals[cur_seed] @ normals[seed_neigh[i]]
                    if dot >= 1:
                        dot = 1.
                    if dot <= -1:
                        dot = -1.
                    if np.arccos(np.abs(dot)) <= ang_thr:
                        Rc.append(seed_neigh[i])
                        L[seed_neigh[i]] = 0
                        if curvatures[seed_neigh[i]] <= curv_thr:
                            Sc.append(seed_neigh[i])
                        curvatures[seed_neigh[i]] = np.inf
            Sc.pop(0)
        R.append(Rc)
    return R


def Segmentation_Normal_Curvature(pcd, true_label, ang_thr=0.045, curv_thr=0.3):
    MLS_pcd, inlier_label = moving_least_squares(pcd, 1.0, 50)
    Final_label = np.ones(inlier_label.shape, np.uint8) * 2
    cond_inlier = np.where(inlier_label == 1)[0]
    normals = np.asarray(MLS_pcd.normals[:]).astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1).reshape((normals.shape[0], 1))
    pcd_tree = o3d.geometry.KDTreeFlann(MLS_pcd)
    points = np.asarray(MLS_pcd.points[:]).astype(np.float64)
    Neighbor_Inf = Neigbor_Information(pcd_tree, 50, points)
    curvatures = Curvature_estimation(points, Neighbor_Inf)
    Label_list = Region_Growing(points, normals, curvatures, Neighbor_Inf, ang_thr=ang_thr, curv_thr=curv_thr)
    num_label_list = np.zeros(len(Label_list, ))
    for i in range(len(Label_list)):
        num_label_list[i] = len(Label_list[i])
    normal_surface_index = np.argmax(num_label_list)
    cond = np.where(num_label_list >= 2)[0]
    for i in range(cond.shape[0]):
        cur_segment = Label_list[cond[i]]
        if cond[i] == normal_surface_index:
            Final_label[cond_inlier[np.array(cur_segment)]] = 0
        else:
            Final_label[cond_inlier[np.array(cur_segment)]] = 1

    # Visualization_pcd(np.asarray(pcd.points[:]), Final_label, scale=1)
    return Final_label
