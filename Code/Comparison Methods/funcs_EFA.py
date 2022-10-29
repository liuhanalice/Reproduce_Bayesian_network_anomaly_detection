from my_funcs import *
from Generate_Defects_v2 import depression_circle_v2


def init_normals(points, neighbor_info):
    normals = np.zeros_like(points)
    for i in range(points.shape[0]):
        neighbor_indices = neighbor_info[i]
        neighbor_pts = points[neighbor_indices]
        center = np.mean(neighbor_pts, axis=0)
        cov = 1 / neighbor_indices.shape[0] * np.dot((neighbor_pts - center).T, neighbor_pts - center)
        vals, vecs = np.linalg.eigh(cov)
        normals[i] = vecs[:, 0]
    return normals


def mls(points, neighbor):
    new_points = points.copy()
    normals = np.zeros_like(new_points)
    for i in range(new_points.shape[0]):
        new_points[i], normals[i] = estimation_projection(points[neighbor[i]], points[i])

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


def fpfh_detection(pcd, neighbor_size, neighbor_size2=50, visualization='False', tau=20):
    points = np.asarray(pcd.points[:]).astype(np.float64)
    print("Number of points:", points.shape[0])
    print("points:", points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    num = points.shape[0]
    neighbor_info = Neigbor_Information(pcd_tree, neighbor_size2, points)

    new_points, normals = mls(points, neighbor_info)
    pcd.points = o3d.utility.Vector3dVector(new_points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    if visualization:
        Visualization_pcd(new_points, np.zeros((num,)).astype(np.uint8), 'denoised_version',scale=1)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamKNN(neighbor_size))
    fpfh = pcd_fpfh.data.T
    print(fpfh.shape)
    label = np.zeros((num,), np.uint8)
    feature = np.zeros((num, 2))
    feature[:, 0] = np.abs(fpfh[:, 26] - fpfh[:, 27])
    feature[:, 1] = np.abs(fpfh[:, 28] - fpfh[:, 27])
    cond = np.where(np.min(feature, axis=1) < tau)
    # label[fpfh[:, 9] >= tau] = 1
    label[cond] = 1
    if visualization:
        Visualization_pcd(points, label, scale=1)
    return label