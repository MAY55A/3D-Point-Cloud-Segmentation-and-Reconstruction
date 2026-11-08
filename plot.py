import matplotlib.pyplot as plt
import numpy as np

def display_sample(points, max_points=20000):
    """Return a deterministic subset for interactive rendering."""
    if len(points) <= max_points:
        return points
    step = int(np.ceil(len(points) / max_points))
    return points[::step]

def plot_sphere(params, ax, label, resolution=40):
    center = params[:3]
    radius = params[3]
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)

    ax.plot_surface(x, y, z, alpha=0.5, label=label)

def plot_planes_sphere(points, planes, params, inliers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect((1, 1, 1))
    scatter_options = {
        "s": 1,
        "depthshade": False,
        "edgecolors": "none",
        "linewidths": 0,
    }
    display_points = display_sample(points)

    # plot unclassified points
    ax.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
            color='gray', label='Points', **scatter_options)

    # plot planes
    for i, plane in enumerate(planes):
        display_plane = display_sample(plane)
        ax.scatter(display_plane[:, 0], display_plane[:, 1], display_plane[:, 2],
                label=f'Plane {i+1}', **scatter_options)

    # plot sphere inliers
    display_inliers = display_sample(inliers)
    ax.scatter(display_inliers[:, 0], display_inliers[:, 1], display_inliers[:, 2],
            color='black', label='Sphere', **scatter_options)

    # plot sphere from params
    #plot_sphere(params, ax, label="Best Sphere")

    ax.legend()
    plt.show()