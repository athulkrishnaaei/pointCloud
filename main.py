import open3d as o3d
import numpy as np


def calculate_residual(plane_model, inliers):
    # Extracting coefficients
    A, B, C, D = plane_model

    # Calculate residual
    residual = 0.0
    num_inliers = len(inliers)

    if num_inliers > 0:
        for index in inliers:
            if 0 <= index < len(points):
                point = points[index]
                x, y, z = point
                distance_to_plane = abs(A * x + B * y + C * z + D) / np.sqrt(A**2 + B**2 + C**2)
                residual += distance_to_plane
            else:
                print("Invalid index:", index)
        residual = np.sqrt(residual)

    return residual


def evaluate_plane(coefficients, inliers):
    result = {"successful_fit": False, "residual_less_than_2cm": False, "slope_less_than_4_degrees": False}

    A, B, C, D = coefficients

    # Calculating slope
    slope = -A / C

    # Calculating residual
    result["successful_fit"] = True
    second_residual = calculate_residual(coefficients, inliers)
    result["residual_less_than_2cm"] = abs(second_residual) < 0.2  # 2cm threshold
    result["slope_less_than_4_degrees"] = slope < 4.0  # 4 degrees threshold

    print("Slope is  " + str(slope))  # for debugging
    print("residual is " + str(second_residual))  # for debugging

    return result


# Load point cloud
pcd = o3d.io.read_point_cloud("cloud.pcd")

downpcd = pcd.voxel_down_sample(voxel_size=0.1)

resolution = 0.02  # 2x2cm resolution
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(downpcd, voxel_size=resolution)  # applying voxel filtering

voxel_points = {}

# Iterate through points to assign them to appropriate voxel
for i in range(len(pcd.points)):
    voxel_center = voxel_grid.get_voxel(pcd.points[i])
    if tuple(voxel_center) not in voxel_points:
        voxel_points[tuple(voxel_center)] = []
    voxel_points[tuple(voxel_center)].append(i)

geometries = []  # List to hold colored planes

# Iterate through voxels
for voxel_center, indices in voxel_points.items():
    # Extract points within the voxel
    points = np.asarray(pcd.points)[indices]

    # Checking if enough points for plane fitting
    if len(points) < 3:
        continue

    # Convert points to PointCloud object
    voxel_pcd = o3d.geometry.PointCloud()
    voxel_pcd.points = o3d.utility.Vector3dVector(points)

    # Perform plane fitting
    plane_model, inliers = voxel_pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=100)
    if len(inliers) < 3:  # Skip if not enough inliers for plane fitting
        continue

    # Evaluate the plane
    evaluation_result = evaluate_plane(plane_model, inliers)

    # Color coding planes based on evaluation results
    if evaluation_result["successful_fit"]:
        if evaluation_result["residual_less_than_2cm"] and evaluation_result["slope_less_than_4_degrees"]:
            print("Assigning green")
            plane_color = [0, 1, 0]  # Green
        elif evaluation_result["residual_less_than_2cm"]:
            plane_color = [1, 1, 0]  # Yellow
            print("Assigning yellow")
        elif evaluation_result["slope_less_than_4_degrees"]:
            print("Assigning orange")
            plane_color = [1, 0.5, 0]  # Orange
        else:
            print("Assigning red")
            plane_color = [1, 0, 0]  # Red
    else:
        print("Assigning red")
        plane_color = [1, 0, 0]  # Red

    plane = o3d.geometry.PointCloud()
    plane.points = o3d.utility.Vector3dVector(points[inliers])
    plane.paint_uniform_color(plane_color)
    geometries.append(plane)

# Saving final output
output_path = "colored_planes.pcd"
final_output = o3d.geometry.PointCloud()
for plane in geometries:
    final_output += plane
o3d.io.write_point_cloud(output_path, final_output)