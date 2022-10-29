
from my_funcs import *
import time
from scipy.spatial import ConvexHull


def Computation_Of_K_Converged(K, NN, points, Neighbor_Inf, sigma_d, max_iter):
    for i in range(max_iter):
        K = Computation_Of_K_New(K, NN, points, Neighbor_Inf, sigma_d)
    # print("K",K)
    return K


def Computation_Of_K_New(K, NN, points, neighbor, sigma_d):
    start_time = time.perf_counter()
    K_new = np.zeros(K.shape, np.float64)

    for i in range(points.shape[0]):
        cur_neighbor = neighbor[i]
        points_j = points[cur_neighbor]
        r = -points_j + points[i]
        c = np.exp(-np.square(np.linalg.norm(r, axis=1)) / sigma_d ** 2)
        r = r / np.linalg.norm(r, axis=1).reshape((r.shape[0], 1))
        r_vec = r.reshape((3 * NN, 1))
        R_all = np.dot(r_vec, r_vec.T)
        for j in range(NN):
            rrT = R_all[j * 3:j * 3 + 3, j * 3:j * 3 + 3]
            R = np.eye(3, dtype=np.float64) - 2 * rrT
            T_sym = 1 / 4 * np.dot(rrT, K[cur_neighbor[j]]) + 1 / 4 * np.dot(K[cur_neighbor[j]], rrT)
            S = c[j] * np.dot(R, np.dot(K[cur_neighbor[j]] - T_sym, R.T))
            K_new[i] += S
        eig_values = np.linalg.eigvalsh(K_new[i])
        scale = np.max(np.real(eig_values))
        K_new[i] = K_new[i] / scale
    end_time = time.perf_counter()
    print("Tensor Estimation Running Time (per epoch)", end_time - start_time)
    return K_new


def tensor(neighbor, cur_point, points, sigma_d):
    NN = neighbor.shape[0]
    points_j = points[neighbor]
    r = -points_j + cur_point
    c = np.exp(-np.square(np.linalg.norm(r, axis=1)) / sigma_d ** 2)
    r = r / np.linalg.norm(r, axis=1).reshape((r.shape[0], 1))
    r_vec = r.reshape((3 * NN, 1))
    R_all = np.dot(r_vec, r_vec.T)
    K = np.zeros((3, 3), np.float64)
    for j in range(NN):
        rrT = R_all[j * 3:j * 3 + 3, j * 3:j * 3 + 3]
        R = np.eye(3, dtype=np.float64) - 2 * rrT
        T_sym = 1 / 4 * np.dot(rrT, np.eye(3, dtype=np.float64)) + 1 / 4 * np.dot(np.eye(3, dtype=np.float64), rrT)
        S = c[j] * np.dot(R, np.dot(np.eye(3, dtype=np.float64) - T_sym, R.T))
        K += S
    return K


def single_indicator(weight, w_min, w_max):
    if weight > w_max:
        indicator = 1
    elif weight <= w_min:
        indicator = 0
    else:
        indicator = weight >= (w_min + w_max) / 2
    return indicator


def tensor_voting_neighbors(K_neighbors, p, pts, sigma):
    NN = K_neighbors.shape[0]
    r = -pts + p
    c = np.exp(-np.square(np.linalg.norm(r, axis=1)) / sigma ** 2)
    r = r / np.linalg.norm(r, axis=1).reshape((r.shape[0], 1))
    r_vec = r.reshape((3 * NN, 1))
    R_all = np.dot(r_vec, r_vec.T)
    K = np.zeros((3, 3), np.float64)
    for j in range(NN):
        rrT = R_all[j * 3:j * 3 + 3, j * 3:j * 3 + 3]
        R = np.eye(3, dtype=np.float64) - 2 * rrT
        T_sym = 1 / 4 * np.dot(rrT, K_neighbors[j]) + 1 / 4 * np.dot(K_neighbors[j], rrT)
        S = c[j] * np.dot(R, np.dot(K_neighbors[j] - T_sym, R.T))
        K += S
    return K


def feature_extraction_v3(pcd, label,w_min=0.55, w_max=0.85):
    sigma_d = 5.0
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points[:]).astype(np.float64)
    indicator = np.zeros((points.shape[0],), np.uint8)
    # Initialization- first tensor voting
    num = points.shape[0]
    Neighbor_Inf = Neigbor_Information(pcd_tree, 80, points)
    K = np.zeros((num, 3, 3), np.float64)
    for j in range(num):
        K[j] = np.eye(3).astype(np.float64)
    K_f = Computation_Of_K_Converged(K, 80, points, Neighbor_Inf, sigma_d, max_iter=5)

    Scale_list = [100, 130, 160, 190, 220]
    Sigma_list = [5.0, 6, 7, 8, 9]

    for i in range(points.shape[0]):
        weight_vec = np.inf * np.ones((len(Scale_list, )), np.float64)
        for scale in range(len(Scale_list)):
            neighbors = PointCloud_Kdtree_NN_Search(pcd_tree, i, points[i], Scale_list[scale])
            cur_K = tensor_voting_neighbors(K_f[neighbors], points[i], points[neighbors], sigma_d * Sigma_list[scale])
            vals = np.linalg.eigvalsh(cur_K)

            weight = (vals[0] + vals[1]) / vals[2]
            weight_vec[scale] = weight
            if (scale >= 1 and weight_vec[scale] >= 1.3 * weight_vec[scale - 1]) or scale == (len(Scale_list) - 1):
                indicator[i] = single_indicator(weight_vec[scale - 1], w_min, w_max)
                break
            if weight > w_max:
                indicator[i] = 1
                break
            elif weight <= w_min:
                indicator[i] = 0
                break
            else:
                continue
    defect_point_index = find_close_contour(pcd, indicator)
    Final_label = np.zeros(label.shape, np.uint8)
    Final_label[defect_point_index] = 1
    return Final_label


def Outlier_filter(pcd):
    _, index = pcd.remove_radius_outlier(nb_points=3, radius=1.5)
    return index


def interior_point_convexhull_judgement(homo_pts, equations):
    # dimension: equation m*3, homo_pts n*3
    s = np.dot(homo_pts, equations.T)
    index = np.zeros((homo_pts.shape[0],), np.uint8)
    for i in range(index.shape[0]):
        if np.where(s[i] <= 1e-4)[0].shape[0] == s.shape[1]:
            index[i] = 1
    return np.where(index == 1)[0]


def find_close_contour(pcd, indicator):
    all_pts = np.array(pcd.points[:])[:, :2]
    edge_pcd = pcd.select_by_index(np.where(indicator == 1)[0])
    filtered_edge_pcd = edge_pcd.select_by_index(Outlier_filter(edge_pcd))
    # o3d.visualization.draw_geometries([filtered_edge_pcd])
    # projection to an plane
    pts2d = np.array(filtered_edge_pcd.points[:])[:, :2]
    hull = ConvexHull(pts2d)
    equations = hull.equations
    homo_pts = np.hstack((all_pts, np.ones((all_pts.shape[0], 1))))
    interior_index = interior_point_convexhull_judgement(homo_pts, equations)
    # interior_pcd = pcd.select_by_index(interior_index)
    # o3d.visualization.draw_geometries([interior_pcd])
    return interior_index