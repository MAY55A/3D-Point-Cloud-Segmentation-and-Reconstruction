# 3D Point Cloud Segmentation and Reconstruction

This project uses RANSAC to detect planes and one sphere in 3D point clouds. The detected objects are saved as a colored ASCII PLY point cloud.

## Requirements

- Python 3.x
- NumPy
- Matplotlib (used by the optional visualization script)

Install the dependencies from the project directory:

```bash
python -m pip install -r requirements.txt
```

## Structure

```text
.
├── main.py                 # Point-cloud processing and RANSAC algorithms
├── plot.py                 # Optional Matplotlib visualization helpers
├── requirements.txt        # Python dependencies
├── data/                   # Input .pcd point clouds
├── output/                 # Generated .ply point clouds
└── snapshots/              # Example result images
```

## Usage

Run `main.py` with an input PCD path and an output PLY path:

```bash
python main.py data/Frame15439.pcd output/Frame15439.ply
```

The program prints the number of inliers found for each plane and the sphere.

## Processing pipeline

- Read point coordinates from PCD input files. 
- Find large planes with RANSAC and remove their inlier points.
- Find one sphere in the remaining points and remove its inliers.
- Find smaller planes in the points that are still unclassified.
- Save the sphere and planes as colored points in the output PLY file.

The thresholds and minimum inlier counts are defined in `main.py` and may need
to be changed for different point clouds.

## Limitations

- It detects only one sphere per input scan.
- Detection values are hardcoded in `main.py`.
- The values are tuned for the included scans.
- Only ASCII PCD files with `x`, `y`, and `z` as the first fields are supported.
- The output contains colored points, not a mesh or solid reconstruction.
- Noise and occlusion can cause missed or incorrect detections.

## Output format

The generated files use ASCII PLY format and store colored vertices:

- Sphere inliers are written in yellow.
- Each detected plane receives a randomly generated color.
- No polygon faces are generated, the output is a colored point cloud rather
  than a surface mesh.

The optional `plot.py` module provides `plot_planes_sphere()` for interactive
3D inspection of classified points. It is not called by default, the
visualization call in `main.py` can be enabled after uncommenting some lines.
