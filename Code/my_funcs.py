import numpy as np
import open3d as o3d
import time
from numba import jit, float64, int32
import cvxpy as cp
import pwlf
import random
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats


def Computation_Of_K_Converged(K, NN, points, Neighbor_Inf, sigma_d=0.7, MaxIter=5):
    for i in range(MaxIter):
        old_K = K.copy()
        K = Computation_Of_K_New(K, NN, points, Neighbor_Inf, sigma_d)
        change = np.linalg.norm((old_K - K).reshape(K.shape[0], 9), axis=1)
        print("tensor change per iteration: ", np.mean(change))
    return K


def Computation_Of_K_New(K, NN, points, neighbor, sigma_d):
    start_time = time.perf_counter()
    K_new = K.copy()

    for i in range(points.shape[0]):
        cur_neighbor = neighbor[i]
        points_j = points[cur_neighbor]
        r = -points_j + points[i]

        c = np.exp(-np.square(np.linalg.norm(r, axis=1)) / sigma_d ** 2)
        norm = np.linalg.norm(r, axis=1).reshape((r.shape[0], 1))
        norm_inv = np.zeros_like(norm, np.float64)
        norm_inv[np.where(norm >= 1e-8)] = 1 / norm[np.where(norm >= 1e-8)]
        r = r * norm_inv
        r_vec = r.reshape((3 * NN, 1))
        R_all = np.dot(r_vec, r_vec.T)
        cur_K_updated = np.zeros((3, 3), np.float64)
        for j in range(NN):
            rrT = R_all[j * 3:j * 3 + 3, j * 3:j * 3 + 3]
            # R = np.eye(3, dtype=np.float64) - 2 * rrT
            # T_sym = 1 / 4 * np.dot(rrT, K_new[cur_neighbor[j]]) + 1 / 4 * np.dot(K_new[cur_neighbor[j]], rrT)
            # S = np.dot(R, np.dot(K_new[cur_neighbor[j]] - T_sym, R.T))
            S = numba_acceleration(c[j], rrT, K_new[cur_neighbor[j]])
            cur_K_updated += S

        eig_values = np.linalg.eigvalsh(cur_K_updated)
        scale = np.max(np.real(eig_values))
        K_new[i] = cur_K_updated / scale

    end_time = time.perf_counter()
    print("Tensor Estimation Running Time (per epoch)", end_time - start_time)
    return K_new


@jit()
def numba_acceleration(c, rrT, K_neighbor):
    R = np.eye(3, dtype=np.float64) - 2 * rrT
    T_sym = 1 / 4 * np.dot(rrT, K_neighbor) + 1 / 4 * np.dot(K_neighbor, rrT)
    S = np.dot(R, np.dot(K_neighbor - T_sym, R.T))
    return c * S


def Neigbor_Information(pcd_tree, NN, points):
    Neigbor_Inf = np.zeros((points.shape[0], NN), np.int32)
    for i in range(points.shape[0]):
        Neigbor_Inf[i] = PointCloud_Kdtree_NN_Search(pcd_tree, i, points[i], NN)
    return Neigbor_Inf


def Neighbor_Information_self(pcd_tree, NN, points):
    Neigbor_Inf = np.zeros((points.shape[0], NN + 1), np.int32)
    for i in range(points.shape[0]):
        neighbor = pcd_tree.search_knn_vector_3d(points[i], NN + 1)
        int_vector_index = neighbor[1]
        int_vector_index_arr = np.asarray(int_vector_index)
        Neigbor_Inf[i] = int_vector_index_arr
    return Neigbor_Inf


def PointCloud_Kdtree_NN_Search(pcd_tree, i, data, NN):
    neighbor = pcd_tree.search_knn_vector_3d(data, NN + 1)
    int_vector_index = neighbor[1]
    int_vector_index_arr = np.asarray(int_vector_index)
    voter_index = list(int_vector_index_arr)
    voter_index.remove(i)
    return np.array(voter_index)


def Initialization_h_m_MS(points):
    A = points
    vec_1 = np.ones((points.shape[0], 1), np.float64)
    ATA = np.dot(A.T, A)
    solu = np.dot(np.linalg.inv(ATA), np.dot(A.T, vec_1))
    if solu[2] <= 0:
        solu *= -1
        return solu / np.linalg.norm(solu), -1 / np.linalg.norm(solu)
    else:
        return solu / np.linalg.norm(solu), 1 / np.linalg.norm(solu)


def Projection_2D(points):
    center = np.mean(points, axis=0)
    cov = 1 / points.shape[0] * np.dot((points - center).T, points - center)
    vals, vecs = np.linalg.eigh(cov)
    point2d = np.dot(vecs[:, 1:].T, (points - center).T).T
    return point2d


def Visualization_pcd(pts, label, window_name='unspecified', scale=6):
    points = pts.copy()
    n = points.shape[0]
    pcd = o3d.geometry.PointCloud()
    points[:, 2] *= scale
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((n, 3), np.uint8)
    colors[label == 0] = np.array([[0, 0, 0]])
    colors[label == 1] = np.array([[0, 0, 255]])
    colors[label == 2] = np.array([[255, 0, 0]])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=window_name, point_show_normal=False)


def Read_txt_pcd(file_name):
    points = np.loadtxt(file_name)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((points.shape[0], 3), np.uint8)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


class EM_options:
    def __init__(self, bg_k, outliers_rate, defect_depth, defect_radius, trans, defect_pos, spline_paras=None,
                 spline_knot=None,
                 bg_size=20., bg_std_depth=0.15, bg_std_xy=0.02, outliers_std_depth=1.5, outliers_std_xy=0.02,
                 step=-0.45, spline_flag=True):
        self.bg_k = bg_k
        self.bg_size = bg_size
        self.bg_std_depth = bg_std_depth
        self.bg_std_xy = bg_std_xy
        self.outliers_rate = outliers_rate
        self.outliers_std_depth = outliers_std_depth
        self.outliers_std_xy = outliers_std_xy
        self.defect_depth = defect_depth
        self.defect_radius = defect_radius
        self.defect_trans = trans
        self.defect_pos = defect_pos
        self.step = step
        self.spline_flag = spline_flag
        self.spline_paras = spline_paras
        self.spline_knot = spline_knot


def Tensor_Diff_K(K, Neighbor_Inf):
    diff = np.zeros(Neighbor_Inf.shape, np.float64)
    for i in range(K.shape[0]):
        K_inv_minus_S = K[i] - K[Neighbor_Inf[i], :, :]
        diff[i] = np.linalg.norm(K_inv_minus_S.reshape((Neighbor_Inf.shape[1], 3 * 3)), axis=1)
    return diff


def Update_sigma_sq(weight0, points, h, m, paras, n_knot):
    R, T = Compute_R_T(h, m)
    n = points.shape[0]
    B_b, pts = Spline(points, R, T, n_knot)
    delta_m = np.dot(B_b, paras)
    xTh = np.dot(points, h)
    xTh_dm = (xTh - (m + delta_m))
    print(xTh_dm.shape)
    # plt.hist(xTh_dm,60)
    # plt.show()
    sigma_sq = np.sum((xTh_dm) * (xTh_dm) * weight0.reshape((n, 1)))
    pts_R = np.dot(R.T, points.T).T
    return max(sigma_sq / (np.sum(weight0)), 0.001), delta_m, B_b, R, pts_R


# plane equation nTx=b,nTx-b=0,n shape(3,)
def Compute_R_T(n, b):
    n = n[:, 0]
    norm = np.linalg.norm(n)
    n = n / norm
    b = b / norm
    R = np.zeros((3, 3), np.float64)
    R[:, 2] = n
    beta = np.pi - np.arccos(n[2])
    if abs(np.sin(beta)) <= 1e-8:
        alpha = 0
    else:
        cos_alpha = n[0] / np.sin(beta)
        sin_alpha = n[1] / np.sin(beta)
        alpha = np.arccos(cos_alpha)
        if np.abs(np.sin(alpha) - sin_alpha) >= 1e-8:
            alpha = 2 * np.pi - alpha
    # print("x3",np.array([[np.sin(beta)*np.cos(alpha)], [np.sin(beta)*np.sin(alpha)], [-np.cos(beta)]],np.float64))
    y1 = np.array([[-np.sin(alpha)], [np.cos(alpha)], [0]], np.float64)
    y2 = np.array([[np.cos(beta) * np.cos(alpha)], [np.cos(beta) * np.sin(alpha)], [np.sin(beta)]], np.float64)
    x3 = np.array([[np.sin(beta) * np.cos(alpha)], [np.sin(beta) * np.sin(alpha)], [-np.cos(beta)]], np.float64)
    gamma = 0.
    x1x2 = np.zeros((3, 2), np.float64)
    err = 10
    for i in range(1000):
        gamma += 2 * np.pi / 1000
        new_err, new_x1x2 = Axis_error(y1, y2, gamma)
        if new_err < err:
            err = new_err
            x1x2 = new_x1x2
    R[:, :2] = x1x2
    T = b * n
    return R, T


@jit(forceobj=True)
def Axis_error(y1, y2, gamma):
    rot_gamma_mat = np.array([[np.cos(gamma), -np.sin(gamma)], [np.sin(gamma), np.cos(gamma)]],
                             np.float64)
    y1y2 = np.hstack((y1, y2))
    x1x2 = np.dot(y1y2, rot_gamma_mat)

    return np.linalg.norm(x1x2[:, 0] - np.array([1., 0., 0.])) + np.linalg.norm(
        x1x2[:, 1] - np.array([0., 1., 0.])), x1x2


def Points_Range(points, R, t):
    pts = np.dot(R.T, points.T).T + (-np.dot(R.T, t))
    # print(pts)
    x_min = np.min(pts[:, 0])
    x_max = np.max(pts[:, 0])
    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])
    range_x = x_max - x_min
    range_y = y_max - y_min
    return x_min - 0.05 * range_x, x_max + 0.05 * range_x, y_min - 0.05 * range_y, y_max + 0.05 * range_y, pts


def BaseFunction(i, k, u, knot):
    Nik_u = 0
    if k == 1:
        if u >= knot[i] and u < knot[i + 1]:
            Nik_u = 1.0
        else:
            Nik_u = 0.0
    else:
        length1 = knot[i + k - 1] - knot[i]
        length2 = knot[i + k] - knot[i + 1]

        if not length1 and not length2:
            Nik_u = 0.0
        elif not length1:
            Nik_u = (knot[i + k] - u) / length2 * BaseFunction(i + 1, k - 1, u, knot)
        elif not length2:
            Nik_u = (u - knot[i]) / length1 * BaseFunction(i, k - 1, u, knot)
        else:
            Nik_u = (u - knot[i]) / length1 * BaseFunction(i, k - 1, u, knot) + \
                    (knot[i + k] - u) / length2 * BaseFunction(i + 1, k - 1, u, knot)
    return Nik_u


def Cubic_Knot_Generation(umin, umax, vmin, vmax, num_knot):
    knot_u = []
    for i in range(num_knot):
        if i <= 3:
            knot_u.append(umin)
        elif 3 < i < num_knot - 4:
            knot_u.append(umin + (umax - umin) / (num_knot - 4 - 3) * (i - 3))
        else:
            knot_u.append(umax)
    knot_v = []
    for i in range(num_knot):
        if i <= 3:
            knot_v.append(vmin)
        elif 3 < i < num_knot - 4:
            knot_v.append(vmin + (vmax - vmin) / (num_knot - 4 - 3) * (i - 3))
        else:
            knot_v.append(vmax)
    return knot_u, knot_v


@jit(float64[:, :](float64[:, :], int32, float64[:], float64[:], int32), forceobj=True)
# cubic B-spline
def BaseMatrix(Points, num, knot_u, knot_v, degree=3):
    num_BaseFunc = len(knot_u) - (degree + 1)
    k = num_BaseFunc ** 2
    B = np.zeros((num, k), np.float64)

    for i in range(num):
        # print(i)
        point = Points[i]
        for j in range(num_BaseFunc):
            for k in range(num_BaseFunc):
                B[i, j * num_BaseFunc + k] = BaseFunction(j, 4, point[0], knot_u) * BaseFunction(k, 4, point[1], knot_v)
    return B


def Spline(points, R, T, n_knot):
    start_time = time.perf_counter()
    umin, umax, vmin, vmax, pts = Points_Range(points, R, T)
    knotB_u, knotB_v = Cubic_Knot_Generation(umin, umax, vmin, vmax, num_knot=n_knot)
    B_b = BaseMatrix(pts, points.shape[0], knotB_u, knotB_v)
    end_time = time.perf_counter()
    print("Spline matrix", end_time - start_time)
    return B_b, pts


def Update_h_m_paras_Linear(epoch_i, points, weight, B, num_paras, num_H, visualization=None):
    start_time = time.perf_counter()
    A = np.zeros((points.shape[0], num_paras + 3), np.float64)

    for i in range(points.shape[0]):
        A[i, :2] = points[i, :2]
        A[i, 2] = -1
        A[i, 3:] = -B[i]
    WA = A.copy()
    for i in range(points.shape[0]):
        WA[i] = A[i] * max(weight[i], 1e-5)
    b = points[:, 2].reshape((points.shape[0], 1))
    ATWA = np.dot(A.T, WA)
    bTWA = np.dot(b.T, WA)
    SIDE_NUM = int(np.sqrt(num_paras))
    R = Differiential_matrix(SIDE_NUM, num_H)

    Rf = np.zeros((num_paras + 3, num_paras + 3))
    Rf[3:, 3:] = R
    Weight_A = A.copy()
    Weight_b = b.copy()
    for i in range(points.shape[0]):
        Weight_A[i] *= np.sqrt(weight[i])
        Weight_b[i] *= np.sqrt(weight[i])
    opt_thr = Gamma_Selection_Constraints(ATWA, bTWA, Rf, num_paras, Weight_A, Weight_b, A, b)
    opt_smooth = Gamma_Selection_Smooth_Constraints(opt_thr, ATWA, bTWA, Rf, num_paras, Weight_A, Weight_b, A, b,
                                                    max_esp=150.)
    # opt_smooth=300
    print("OPT_THR,OPT_SMOOTH", opt_thr, opt_smooth)
    x = cp.Variable(num_paras + 3)
    ATWA += 1e-6 * Rf
    obj = cp.Minimize(1 / Weight_A.shape[0] * (cp.quad_form(x, ATWA) + 2 * bTWA @ x))
    cvec = np.ones((num_paras,), np.float64)
    cmat = np.diag(cvec)
    constr = [cmat @ x[3:] <= opt_thr * np.ones((num_paras,), np.float64),
              -cmat @ x[3:] <= opt_thr * np.ones((num_paras,), np.float64)]
    constr.append(cp.quad_form(x, Rf) <= opt_smooth)
    prob = cp.Problem(obj, constr)
    try:
        prob.solve(solver=cp.CVXOPT)
        print("status:", prob.status)
    except:
        prob.solve(solver=cp.SCS)
    h = np.array([[x.value[0]], [x.value[1]], [1]], np.float64)
    m = x.value[2] / np.linalg.norm(h)
    paras = x.value[3:].reshape((num_paras, 1))
    paras /= np.linalg.norm(h)
    h = h / np.linalg.norm(h)

    if np.max(paras) - np.min(paras) <= 1e-5:
        mean_paras = np.mean(paras)
        paras -= mean_paras
        m += mean_paras
    end_time = time.perf_counter()
    print("Running time of updating h, m and spline paras", end_time - start_time)
    if epoch_i == 1 and visualization:
        Visulization_spline_surface(points, B, paras, 1, scale=visualization)
    return h, m, paras


def Visulization_spline_surface(points, B_b, paras, opt, scale):
    if opt == 0:
        for i in range(B_b.shape[1]):
            new_pts = points.copy()
            new_pts[:, 2] = B_b[:, i] * scale
            # print(np.max(B_b[:,i]))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(new_pts)
            colors = np.zeros((points.shape[0], 3), np.uint8)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd], point_show_normal=False)
    if opt == 1:
        B = np.dot(B_b, paras)
        print("stats B", np.max(B), np.min(B), np.std(B))
        new_pts = points.copy()
        new_pts[:, 2] = B[:, 0] * scale
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_pts)
        colors = np.zeros((points.shape[0], 3), np.uint8)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], point_show_normal=False)


def Differiential_matrix(SIDE_NUM, num_H):
    W = np.zeros((SIDE_NUM ** 2, SIDE_NUM ** 2), np.float64)
    for i in range(SIDE_NUM):
        for j in range(SIDE_NUM - 1):
            cur_index = SIDE_NUM * i + j
            next_index_j = cur_index + 1
            W[cur_index, next_index_j] = 1
            if i < SIDE_NUM - 1:
                next_index_i = SIDE_NUM * (i + 1) + j
                W[cur_index, next_index_i] = 1

    for i in range(SIDE_NUM ** 2):
        for j in range(SIDE_NUM ** 2):
            if W[i, j] == 1:
                W[j, i] = 1
    Out = np.ones((SIDE_NUM ** 2,))
    for i in range(SIDE_NUM):
        for j in range(SIDE_NUM):
            if (i <= num_H - 1 or i >= SIDE_NUM - num_H) and (j <= num_H - 1 or j >= SIDE_NUM - num_H):
                Out[i * SIDE_NUM + j] = 0
    zero_pos = np.where(Out == 0)[0]
    W[zero_pos] = 0
    W[:, zero_pos] = 0
    L = np.diag(np.sum(W, axis=1)) - W
    return L


def Gamma_Selection_Constraints(ATWA, bTWA, R, num_paras, Weight_A, Weight_b, A, b, max_esp=15.):
    # in the 1st case, smoothness is ignored.
    gamma = 0
    sum_num = 100
    order_list = [i for i in range(Weight_A.shape[0])]
    random.shuffle(order_list)
    final_list = np.array(order_list[:int(Weight_A.shape[0] * 0.7)])
    test_list = np.array(order_list[int(Weight_A.shape[0] * 0.7):])
    band_WA = Weight_A[final_list]
    band_ATWA = np.dot(A[final_list].T, band_WA)
    band_bTWA = np.dot(b[final_list].T, band_WA)
    eps_range = np.linspace(0.01, max_esp, sum_num)
    residual_vec = np.zeros((sum_num,), np.float64)
    for i in range(sum_num):
        cur_eps = eps_range[i]
        x = cp.Variable(num_paras + 3)
        band_ATWA = (band_ATWA + band_ATWA.T) / 2
        band_ATWA += 1e-6 * R
        obj = cp.Minimize(1 / Weight_A.shape[0] * (cp.quad_form(x, band_ATWA) + 2 * band_bTWA @ x))
        cvec = np.ones((num_paras,), np.float64)
        cmat = np.diag(cvec)
        constr = [cmat @ x[3:] <= cur_eps * np.ones((num_paras,), np.float64),
                  -cmat @ x[3:] <= cur_eps * np.ones((num_paras,), np.float64)]
        prob = cp.Problem(obj, constr)
        try:
            prob.solve(solver=cp.CVXOPT)
        except:
            prob.solve(solver=cp.SCS)
        if prob.status != 'optimal' and prob.status != 'optimal_inaccurate':
            print("SCS")
            prob.solve(solver=cp.SCS)
        residual_vec[i] = np.linalg.norm(
            np.dot(Weight_A[test_list], x.value.reshape((Weight_A.shape[1], 1))) + Weight_b[test_list])
    int_range = np.linspace(0, sum_num - 1, sum_num)
    my_pwlf = pwlf.PiecewiseLinFit(int_range, residual_vec)
    breaks = my_pwlf.fit(2)
    breaks_arr = np.array(breaks)
    break_point = np.array([[breaks_arr[1] + 0.1], [my_pwlf.predict(breaks_arr[1] + 0.1)[0]]])
    curve_pts = np.zeros((2, residual_vec.shape[0]))
    curve_pts[0, :] = int_range
    curve_pts[1, :] = residual_vec
    distance = np.linalg.norm(break_point - curve_pts, axis=0)
    pos = np.argmin(distance)
    pos += 40
    y_hat = my_pwlf.predict(int_range)
    # plt.plot(y_hat)
    # plt.scatter(int_range[pos], residual_vec[pos], color='red')
    # plt.plot(residual_vec)
    # plt.show()
    return eps_range[pos]


def Gamma_Selection_Smooth_Constraints(opt_thr, ATWA, bTWA, R, num_paras, Weight_A, Weight_b, A, b, max_esp=30.):
    sum_num = 150
    order_list = [i for i in range(Weight_A.shape[0])]
    random.shuffle(order_list)
    final_list = np.array(order_list[:int(Weight_A.shape[0] * 0.7)])
    test_list = np.array(order_list[int(Weight_A.shape[0] * 0.7):])
    band_WA = Weight_A[final_list]
    band_ATWA = np.dot(A[final_list].T, band_WA)

    band_bTWA = np.dot(b[final_list].T, band_WA)
    eps_range = np.linspace(0.01, max_esp, sum_num)
    residual_vec = np.zeros((sum_num,), np.float64)
    for i in range(sum_num):
        cur_eps = eps_range[i]
        x = cp.Variable(num_paras + 3)
        band_ATWA = (band_ATWA + band_ATWA.T) / 2
        band_ATWA += 1e-6 * R
        obj = cp.Minimize(1 / Weight_A.shape[0] * (cp.quad_form(x, band_ATWA) + 2 * band_bTWA @ x))
        cvec = np.ones((num_paras,), np.float64)
        cmat = np.diag(cvec)
        constr = [cmat @ x[3:] <= opt_thr * np.ones((num_paras,), np.float64),
                  -cmat @ x[3:] <= opt_thr * np.ones((num_paras,), np.float64)]
        constr.append(cp.quad_form(x, R) <= cur_eps)
        prob = cp.Problem(obj, constr)
        try:
            prob.solve(solver=cp.CVXOPT)
        except:
            print("Using SCS solver!")
            prob.solve(solver=cp.SCS)
        residual_vec[i] = np.linalg.norm(
            np.dot(Weight_A[test_list], x.value.reshape((Weight_A.shape[1], 1))) + Weight_b[test_list])
    int_range = np.linspace(0, sum_num - 1, sum_num)
    my_pwlf = pwlf.PiecewiseLinFit(int_range, residual_vec)
    breaks = my_pwlf.fit(2)
    breaks_arr = np.array(breaks)
    y_hat = my_pwlf.predict(int_range)
    break_point = np.array([[breaks_arr[1] + 0.1], [my_pwlf.predict(breaks_arr[1] + 0.1)[0]]])
    curve_pts = np.zeros((2, residual_vec.shape[0]))
    curve_pts[0, :] = int_range
    curve_pts[1, :] = residual_vec
    distance = np.linalg.norm(break_point - curve_pts, axis=0)
    pos = np.argmin(distance)
    pos += 15
    # plt.plot(y_hat)
    # plt.scatter(int_range[pos], residual_vec[pos], color='red')
    # plt.plot(residual_vec)
    # plt.show()
    return eps_range[pos]


def Label_Assignment(weight0, weight1):
    weight_mat = np.zeros((weight0.shape[0], 3), np.float64)
    weight_mat[:, 0] = weight0
    weight_mat[:, 1] = weight1
    weight_mat[:, 2] = 1. - weight0 - weight1
    return np.argmax(weight_mat, axis=1)


def Update_sigma_s(weight0, weight1, weight2, tensor_diff, miu, neighbor):
    fs = miu
    # update sigma_s_sq
    c_s = 0.
    d_s = 0.
    for i in range(tensor_diff.shape[0]):
        d_s += np.sum(
            (weight0[i] + weight1[i]) * (weight0[neighbor[i]] + weight1[neighbor[i]]) * np.square(fs - tensor_diff[i]))
        c_s += np.sum((weight0[i] + weight1[i]) * (weight0[neighbor[i]] + weight1[neighbor[i]]))
    return d_s / c_s


def Posterior_Prob(tensor_diff, sigma_sq, sigma_s_sq, neighbor, miu, points, h, m,
                   delta_m, alpha, max_iter, C_f, C_ns):
    start_time = time.perf_counter()
    num = tensor_diff.shape[0]
    weight0 = alpha[0] * np.ones((num,), np.float64)
    weight1 = alpha[1] * np.ones((num,), np.float64)
    weight2 = alpha[2] * np.ones((num,), np.float64)
    for i in range(max_iter):
        index = [i for i in range(num)]
        np.random.shuffle(index)
        for j in range(num):
            cur_j = index[j]

            weight0[cur_j], weight1[cur_j], weight2[cur_j] = site_posterior_prob(weight0[neighbor[cur_j]],
                                                                                 weight1[neighbor[cur_j]],
                                                                                 weight2[neighbor[cur_j]],
                                                                                 tensor_diff[cur_j], sigma_sq,
                                                                                 sigma_s_sq, C_ns,
                                                                                 miu, h, m, delta_m[cur_j],
                                                                                 points[cur_j], alpha, C_f)
    end_time = time.perf_counter()
    print("Posterior Probability Estimation Running Time", end_time - start_time)
    return weight0, weight1, weight2


@jit()
def site_posterior_prob(weight0, weight1, weight2, tensor_diff, sigma_sq, sigma_s_sq, C_ns, miu, h, m, delta_m,
                        point, alpha, C_f):
    sqrt2pi = np.sqrt(2 * np.pi)
    pxc = np.array([0, C_f, C_f], np.float64)
    pxc[0] = 1 / sqrt2pi / np.sqrt(sigma_sq) * np.exp(-(np.dot(h.T, point.reshape(3, 1)) - (m + delta_m)) ** 2 / (
            2 * sigma_sq))
    # p(c_i|{d_ij}^{N_i},c_{N_i})
    log_pn = np.ones((3,), np.float64)
    for i in range(weight0.shape[0]):
        # log_expect_prob_s = np.log(1 / sqrt2pi / np.sqrt(sigma_s_sq) * np.exp(
        #     -(miu - tensor_diff[i]) ** 2 / 2 / sigma_s_sq) * (
        #                                    weight0[i] + weight1[i]) + C_ns * (weight2[i]))
        log_expect_prob_s = np.log(1 / sqrt2pi / np.sqrt(sigma_s_sq) * np.exp(
            -(miu - tensor_diff[i]) ** 2 / 2 / sigma_s_sq)) * (
                                    weight0[i] + weight1[i]) + np.log(C_ns) * weight2[i]

        log_expect_prob_ns = np.log(C_ns)
        log_pn[0] += log_expect_prob_s
        log_pn[1] += log_expect_prob_s
        log_pn[2] += log_expect_prob_ns
    log_pn += np.log(alpha)
    pn = np.exp(log_pn - logsumexp(log_pn))
    prob = pxc * pn
    new_weight0 = prob[0] / np.sum(prob)
    new_weight1 = prob[1] / np.sum(prob)
    new_weight2 = prob[2] / np.sum(prob)
    return new_weight0, new_weight1, new_weight2


def Update_alpha(w1, w2, w3):
    vec = np.array([w1, w2, w3])
    return vec / np.sum(vec)


def Update_miu_diff(weight0, weight1, weight2, tensor_diff, neighbor):
    # wo can obtain optimal miu_s and miu_ns separably.
    prod = np.zeros(tensor_diff.shape, np.float64)
    for i in range(prod.shape[0]):
        prod[i] = (weight0[i] + weight1[i]) * (weight0[neighbor[i]] + weight1[neighbor[i]])
    cs2 = np.sum(prod)
    cs1 = np.sum(prod * tensor_diff)
    miu_s = cs1 / cs2
    return miu_s


def text_create(name):
    desktop_path = "./EM_DRG_result/"
    # 新创建的txt文件的存放路径
    full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')


def Eigen_Decomposition_PSort(tensor):
    eig_values, eig_vecs = np.linalg.eig(tensor)
    sorted_eig_values_index = np.argsort(-eig_values)
    sorted_eig_values = eig_values[sorted_eig_values_index]
    sorted_eig_vecs = eig_vecs[:, sorted_eig_values_index]
    return sorted_eig_values, sorted_eig_vecs


def Visualization_Final_Label_and_Normals(label, points, K_inv, type_list=1):
    n = points.shape[0]
    colors = np.zeros((n, 3), np.uint8)
    colors[label == 0] = np.array([[0, 0, 0]])
    colors[label == 1] = np.array([[0, 0, 255]])
    colors[label == 2] = np.array([[255, 0, 0]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    normals = np.zeros((n, 3), np.float64)
    for i in range(n):
        cur_tensor = K_inv[i]
        eig_values, eig_vecs = Eigen_Decomposition_PSort(cur_tensor)
        if label[i] in type_list:
            normals[i] = eig_vecs[:, 0]

    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
