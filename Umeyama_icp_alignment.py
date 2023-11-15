import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def quaternion_to_matrix(quat):
    """
    Converts a quaternion to a rotation matrix.
    """
    return R.from_quat(quat).as_matrix()

def matrix_to_quaternion(matrix):
    """
    Converts a rotation matrix to a quaternion.
    """
    return R.from_matrix(matrix).as_quat()

def update_quaternions(original_quaternions, rotation_matrix):
    """
    Updates the quaternions based on the rotation matrix from ICP.
    """
    updated_quats = []
    for quat in original_quaternions:
        original_matrix = quaternion_to_matrix(quat)
        updated_matrix = rotation_matrix @ original_matrix
        updated_quat = matrix_to_quaternion(updated_matrix)
        updated_quats.append(updated_quat)
    return np.array(updated_quats)

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


def icp(A, B, max_iterations=100, tolerance=0.001):
    src = np.copy(A)
    dst = np.copy(B)

    # 초기 변환 행렬 설정
    final_R = np.eye(3)
    final_t = np.zeros((3,))

    prev_error = 0

    for i in range(max_iterations):
        # 현재 소스와 대상 간 가장 가까운 이웃 찾기
        indices = nearest_neighbor(src, dst)
        # 현재 소스와 가장 가까운 대상 포인트 간 변환 계산
        R, t = rigid_transform_3D(src, dst[indices, :])

        # 현재 소스 업데이트
        src = (R @ src.T).T + t

        # 누적 변환 행렬 업데이트
        final_t = R @ final_t + t
        final_R = R @ final_R

        # 오류 체크
        mean_error = np.mean(np.linalg.norm(src - dst[indices, :], axis=1))
        print("ICP #",i, "mean_error : ", mean_error)
        if np.abs(prev_error - mean_error) < tolerance:
            print("prev_error - mean_error is lower than tolerance 0.001, END")
            break
        prev_error = mean_error

    return final_R, final_t, (final_R @ A.T).T + final_t


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

# Extract quaternion data
source_quaternions = source_poses[:, 3:]
updated_source_quaternions = update_quaternions(source_quaternions, final_R)


# 원본 소스 파일 읽기
with open(args.src, 'r') as file:
    lines = file.readlines()

# aligned_source_positions와 함께 새로운 내용으로 파일 쓰기
output_filename = 'modified_' + args.src
# Writing the new file with updated positions and quaternions
with open(output_filename, 'w') as file:
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if i < len(aligned_source_positions):
            new_line = f"{parts[0]} {aligned_source_positions[i][0]:.6f} {aligned_source_positions[i][1]:.6f} {aligned_source_positions[i][2]:.6f} {updated_source_quaternions[i][0]:.6f} {updated_source_quaternions[i][1]:.6f} {updated_source_quaternions[i][2]:.6f} {updated_source_quaternions[i][3]:.6f}\n"
            file.write(new_line)
        else:
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