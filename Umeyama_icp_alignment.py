import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_pose_data(file_path):
    """
    Reads pose data from a text file and returns it as a numpy array.
    Each line in the file represents a pose in the format: timestamp x y z qx qy qz qw
    """
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            pose = [float(val) for val in data[1:]]  # Ignoring timestamp
            poses.append(pose)
    return np.array(poses)

def rigid_transform_3D(A, B):
    """
    Computes the rigid transformation from A to B in 3D space.
    A and B are N x 3 matrices. This function does not require A and B to be of the same length.
    """
    min_len = min(len(A), len(B))

    A = A[:min_len]
    B = B[:min_len]

    N = A.shape[0]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t

# The rest of the script remains the same, except the assert line is removed.
# This function now handles source and reference data of different lengths.


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src.
    """
    indices = np.zeros(src.shape[0], dtype=int)
    for i, point in enumerate(src):
        distances = np.sqrt(np.sum((dst - point) ** 2, axis=1))
        indices[i] = np.argmin(distances)
    return indices

def icp(A, B, init_pose=None, max_iterations=100, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best fit to align two point clouds.
    """
    src = np.copy(A)
    dst = np.copy(B)

    if init_pose is not None:
        R, t = init_pose
        src = (R @ src.T).T + t

    prev_error = 0
    
    for i in range(max_iterations):
        
        # Find the nearest neighbors between the current source and destination points
        indices = nearest_neighbor(src, dst)
        # Compute the transformation between the current source and nearest destination points
        R, t = rigid_transform_3D(src, dst[indices, :])

        # Update the current source
        src = (R @ src.T).T + t

        # Check error
        mean_error = np.mean(np.linalg.norm(src - dst[indices, :], axis=1))
        print("ICP #",i, "mean_error : ", mean_error)
        if np.abs(prev_error - mean_error) < tolerance:
            print("prev_error - mean_error is lower than tolerance 0.001, END")
            break
        prev_error = mean_error

    # Calculate final transformation

    return R, t, src

# Parse command line arguments
parser = argparse.ArgumentParser(description='Rigid transformation between two sets of poses.')
parser.add_argument('--src', type=str, help='Path to the source pose file')
parser.add_argument('--ref', type=str, help='Path to the reference pose file')

args = parser.parse_args()

# Read pose data
source_poses = read_pose_data(args.src)
reference_poses = read_pose_data(args.ref)

# Extract position data
source_positions = source_poses[:, :3]
reference_positions = reference_poses[:, :3]

# Apply ICP
final_R, final_t,  aligned_source_positions = icp(source_positions, reference_positions)
print("Rotation:", final_R)
print("Translation:", final_t)

# 원본 소스 파일 읽기
with open(args.src, 'r') as file:
    lines = file.readlines()

# aligned_source_positions와 함께 새로운 내용으로 파일 쓰기
output_filename = 'modified.txt'
with open(output_filename, 'w') as file:
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if i < len(aligned_source_positions):
            # x y z 좌표만 교체
            new_line = f"{parts[0]} {aligned_source_positions[i][0]:.6f} {aligned_source_positions[i][1]:.6f} {aligned_source_positions[i][2]:.6f} {' '.join(parts[4:])}\n"
            file.write(new_line)
        else:
            # aligned_source_positions보다 더 많은 줄이 있을 경우, 원본 내용 유지
            file.write(line)

print(f"Modified source positions with original timestamps saved to {output_filename}")

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(source_positions[:, 0], source_positions[:, 1], source_positions[:, 2], c='r', marker='o', label='Source')
ax.scatter(reference_positions[:, 0], reference_positions[:, 1], reference_positions[:, 2], c='g', marker='^', label='Reference')
ax.scatter(aligned_source_positions[:, 0], aligned_source_positions[:, 1], aligned_source_positions[:, 2], c='b', marker='*', label='Aligned Source')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(-40, 40)
ax.legend()
plt.show()