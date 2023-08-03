import numpy as np
from sklearn.neighbors import NearestNeighbors

def transformation_generate(rotation, translation):
    '''
    input:
        rotation 3*3
        translation 3*1
    output:
        transformation 4*4
    '''
    transformation = np.eye(4)
    
    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation
    
    return transformation

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def umeyama_alignment(x, y, with_scale = True):
    """
    input:
        x: 3*n
        y: 3*n
        with_scale: calculate scale or not
    output:
        r: rotation
        t: translation
        c: scale
    """

    if x.shape != y.shape:
        raise GeometryException("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)  

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0

    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c

def icp(source, target, max_iterations=100, tolerance=1e-6, init_scale = 1, init_transformation = None):
    """
    iterative get rigid transformation
    input:
        source: N*3
        target: N*3
        init_transformation = 4*4
    output:
        src: transformed point clouds
    """
    src = np.copy(source.T) # 3*N
    dst = np.copy(target.T) # 3*N
    
    if init_transformation is not None:
        r = init_transformation[:3, :3]
        t = init_transformation[:3, 3][:, np.newaxis]
        src = np.multiply(init_scale, np.dot(r, src)) + t
    prev_error = 0
    
    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src.T,dst.T)
        rotation, translation, scale = umeyama_alignment(src, dst[:,indices], with_scale=False)
        # TODO:write into homogeneous coordinates
        src = np.multiply(scale, rotation.dot(src)) + translation[:,np.newaxis]
        mean_error = np.mean(distances)
        
        if np.abs(prev_error-mean_error) < tolerance:
            # last iterative calculate the scale
            rotation, translation, scale = umeyama_alignment(src, dst[:,indices], with_scale=True)
            src = np.multiply(scale, rotation.dot(src)) + translation[:,np.newaxis]
            break
        prev_error = mean_error
    # calc source point clouds Transformation
    rotation, translation, scale = umeyama_alignment(source.T, src, with_scale=True)
    return rotation, translation, scale, src


"""
source_points: from RGBD
target_points: from canonical frame
"""

# init_trans: TODO can be improved to get better result
t_wlh = np.max(target_points,axis=0)-np.min(target_points,axis=0)
s_wlh = np.max(source_points,axis=0)-np.min(source_points,axis=0)
init_scale = np.mean(t_wlh/s_wlh)

# iterative closest points: get rigid transformation result
rotation, translation, scale, Tsrc = icp(
    source_points, target_points, init_scale = init_scale, init_transformation=np.eye(4)
)