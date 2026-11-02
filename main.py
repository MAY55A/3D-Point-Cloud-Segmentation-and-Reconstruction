import numpy as np
import matplotlib.pyplot as plt
import math


def read_pcd(file_path):
    data = []
    with open(file_path, "r") as f:
        # Ignore first lines before data
        while True:
            line = f.readline()
            if line.startswith("DATA"):
                break

        while True:
            line = f.readline()
            if not line:
                break
            data.append(line.strip().split()[:3]) # take only first 3 cols (x, y, z)

    data = np.array(data, dtype=float)
    return data

def write_ply(file_path, planes, sphere_points):
    with open(file_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex ")
        f.write(f"{len(sphere_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        for plane in planes:
            f.write("element vertex ")
            f.write(f"{len(plane)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for point in sphere_points:
            f.write(f"{point[0]} {point[1]} {point[2]} 255 255 0\n")

        for i in range(len(planes)):
            plane_rg = np.random.choice(range(256), size=2)
            for point in planes[i]:
                f.write(f"{point[0]} {point[1]} {point[2]} {plane_rg[0]} {plane_rg[1]} 50\n") # 50 for blue to make sure it's not yellow


def fit_sphere(points):
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    A = np.ones((len(points), 4))
    A[:, 0] = X*2
    A[:, 1] = Y*2
    A[:, 2] = Z*2
    B = X**2 + Y**2 + Z**2
    x, _, rank, _ = np.linalg.lstsq(A, B, rcond=None)
    if rank < 4:
        return None
    d = x[0]**2 + x[1]**2 + x[2]**2 + x[3]
    if d <= 0:
        return None
    r = np.sqrt(d)
    return x[0], x[1], x[2], r

def distance_points_sphere(params, points):
    center = params[:3]
    radius = params[3]
    distances = np.linalg.norm(points - center, axis=1)
    return np.abs(distances - radius)

def ransac_sphere(points, iterations=2000, threshold=1):
    best_inliers = []
    best_params = None

    for _ in range(iterations):
        sample = points[np.random.choice(len(points), 4, replace=False)]
        params = fit_sphere(sample)
        if params is None:
            continue
        inliers = points[distance_points_sphere(params, points) < threshold]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_params = params

    best_params = fit_sphere(best_inliers)
    best_inliers_mask = distance_points_sphere(best_params, points) < threshold
    return best_inliers_mask

def distance_points_plane(params, points):
    a, b, c, d = params
    return np.abs((a*points[:, 0]+b*points[:, 1]+c*points[:, 2]+d)/np.sqrt(a**2+b**2+c**2))

def fit_plane(points):
    centroid = np.mean(points, axis=0)
    _, s, vh = np.linalg.svd(points - centroid, full_matrices=False)
    if s[1] < 1e-12: # points are almost collinear
        return None
    n = vh[2]
    d = -np.dot(n, centroid)
    return n[0], n[1], n[2], d

def ransac_plane(points, threshold, iterations):
    best_inliers = []
    best_params = None
    N = points.shape[0]

    for _ in range(iterations):
        sample = points[np.random.choice(N, 3, replace=False)]
        params = fit_plane(sample)
        if params is None:
            continue
        inliers = points[distance_points_plane(params, points) < threshold]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_params = params

    best_params = fit_plane(best_inliers)
    best_inliers_mask = distance_points_plane(best_params, points) < threshold
    return best_inliers_mask

# Run RANSAC to find planes with a minimum number of inliers and remove the inliers
def find_planes(points, min_inliers, threshold=20, iterations=2000):
    planes = []
    while True:
        inliers_mask = ransac_plane(points, threshold, iterations)
        n_inliers = inliers_mask.sum()
        if n_inliers < min_inliers:
            break
        print(f"Found plane with {n_inliers} inliers")

        planes.append(points[inliers_mask])
        points = points[~inliers_mask]
    return planes, points

planes = []
points = read_pcd("example2.pcd")
# decrease threshold -> more planes (less condensed)
# too big threshold -> non planes are detected (part of sphere)
# increase min_inliers -> less planes

# Find big planes, then find sphere, then find small planes
new_planes, points = find_planes(points, min_inliers=30000, threshold=30, iterations=2000)
planes.extend(new_planes)
print(f"Found {len(planes)} planes before sphere detection")

inliers_mask = ransac_sphere(points, threshold=150, iterations=3000)
inliers = points[inliers_mask]
points = points[~inliers_mask]
print(f"Found sphere with {len(inliers)} inliers")

new_planes, points = find_planes(points, min_inliers=5000, threshold=10, iterations=1000)
planes.extend(new_planes)
print(f"Found {len(new_planes)} planes after sphere detection")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='gray', s=1, label='Points')
for i, plane in enumerate(planes):
    ax.scatter(plane[:, 0], plane[:, 1], plane[:, 2], s=1, label=f'Plane {i+1}')

ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], color='black', s=1, label='Sphere')
ax.legend()
plt.show()