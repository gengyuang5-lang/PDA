import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# Simulate true target position
true_position = np.array([10, 10])

# Function to convert polar coordinates (distance, angle) to Cartesian (x, y)
def polar_to_cartesian(distance, angle):
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)
    return np.array([x, y])

# Function to compute Jacobian for polar to Cartesian transformation
# Used to properly transform noise covariance from polar to Cartesian space
def polar_to_cartesian_jacobian(distance, angle):
    """
    Jacobian matrix for transformation from (r, theta) to (x, y)
    J = [[cos(theta), -r*sin(theta)],
         [sin(theta),  r*cos(theta)]]
    """
    J = np.array([[np.cos(angle), -distance * np.sin(angle)],
                  [np.sin(angle),  distance * np.cos(angle)]])
    return J

# Function to simulate radar measurement (distance and angle with noise)
def generate_radar_measurement(true_position, distance_noise=0.3, angle_noise=0.05):
    distance = np.linalg.norm(true_position) + np.random.normal(0, distance_noise)  # Add Gaussian noise to distance
    angle = np.arctan2(true_position[1], true_position[0]) + np.random.normal(0, angle_noise)  # Add noise to angle
    return np.array([distance, angle])

# Function to simulate camera measurement (x, y position with noise)
def generate_camera_measurement(true_position, noise_level=0.1):
    x = true_position[0] + np.random.normal(0, noise_level)
    y = true_position[1] + np.random.normal(0, noise_level)
    return np.array([x, y])

# ========== PDA算法相关函数 ==========
# Function to generate multiple candidate measurements (including clutter)
def generate_candidate_measurements(true_position, num_clutter=3, measurement_range=20, noise_level=0.1):
    """
    生成候选测量值，包括真实测量和杂波
    true_position: 真实目标位置
    num_clutter: 杂波数量
    measurement_range: 测量范围
    noise_level: 测量噪声水平
    """
    measurements = []
    
    # 真实测量值（带噪声）
    true_measurement = generate_camera_measurement(true_position, noise_level)
    measurements.append(true_measurement)
    
    # 生成杂波（随机位置）
    for _ in range(num_clutter):
        clutter = np.array([
            np.random.uniform(-measurement_range, measurement_range),
            np.random.uniform(-measurement_range, measurement_range)
        ])
        measurements.append(clutter)
    
    return np.array(measurements)

# Function for gating (validation gate)
def gating(measurements, x_pred, P_pred, H, R, gate_threshold=9.21):
    """
    门控：筛选出在门控内的候选测量值
    gate_threshold: 门控阈值（卡方分布，对应95%置信度约为9.21）
    """
    validated_measurements = []
    innovations = []
    innovation_covariances = []
    
    S = H @ P_pred @ H.T + R  # 新息协方差矩阵
    
    for z in measurements:
        y = z - H @ x_pred  # 新息
        d_squared = y.T @ np.linalg.inv(S) @ y  # 马氏距离的平方
        
        if d_squared <= gate_threshold:
            validated_measurements.append(z)
            innovations.append(y)
            innovation_covariances.append(S)
    
    return np.array(validated_measurements), innovations, innovation_covariances

# Function to compute association probabilities
def compute_association_probabilities(validated_measurements, x_pred, P_pred, H, R, 
                                     pd=0.9, lambda_clutter=0.1):
    """
    计算关联概率
    pd: 检测概率
    lambda_clutter: 杂波密度（单位体积内的杂波数）
    """
    if len(validated_measurements) == 0:
        return [], 0.0
    
    S = H @ P_pred @ H.T + R
    S_inv = np.linalg.inv(S)
    det_S = np.linalg.det(S)
    
    # 计算每个测量值的似然函数值
    likelihoods = []
    for z in validated_measurements:
        y = z - H @ x_pred
        # 高斯似然函数
        likelihood = np.exp(-0.5 * y.T @ S_inv @ y) / np.sqrt((2 * np.pi) ** len(y) * det_S)
        likelihoods.append(likelihood)
    
    # 计算关联概率
    # P(θ_i | Z) = likelihood_i / (sum(likelihoods) + lambda_clutter * (1 - pd) / pd)
    sum_likelihoods = sum(likelihoods)
    denominator = sum_likelihoods + lambda_clutter * (1 - pd) / pd
    
    association_probs = [likelihood / denominator for likelihood in likelihoods]
    
    # 无关联概率（所有测量值都是杂波的概率）
    prob_no_association = (lambda_clutter * (1 - pd) / pd) / denominator
    
    return association_probs, prob_no_association

# Function for PDA update
def pda_update(x_pred, P_pred, validated_measurements, association_probs, prob_no_association, H, R):
    """
    PDA更新步骤
    """
    if len(validated_measurements) == 0:
        # 没有有效测量值，只使用预测值
        return x_pred, P_pred
    
    # 计算加权新息
    weighted_innovation = np.zeros(len(x_pred))
    for i, z in enumerate(validated_measurements):
        y = z - H @ x_pred
        weighted_innovation += association_probs[i] * y
    
    # 计算加权协方差
    S = H @ P_pred @ H.T + R
    S_inv = np.linalg.inv(S)
    
    weighted_covariance = np.zeros_like(S)
    for i, z in enumerate(validated_measurements):
        y = z - H @ x_pred
        weighted_covariance += association_probs[i] * np.outer(y, y)
    
    # 计算卡尔曼增益
    K = P_pred @ H.T @ S_inv
    
    # PDA状态更新
    x_pda = x_pred + K @ weighted_innovation
    
    # PDA协方差更新
    P_c = P_pred - K @ S @ K.T
    P_pda = prob_no_association * P_pred + (1 - prob_no_association) * P_c + K @ weighted_covariance @ K.T
    
    return x_pda, P_pda

# Generate simulated radar and camera measurements
z_radar_polar = generate_radar_measurement(true_position)
z_camera = generate_camera_measurement(true_position)

# Convert radar measurement from polar to Cartesian coordinates
z_radar = polar_to_cartesian(z_radar_polar[0], z_radar_polar[1])

# Initial state estimate - use first measurement as initial guess (more realistic)
# Or use a weighted average of measurements
x = (z_radar + z_camera) / 2.0  # Better initial estimate using both sensors
P = np.eye(2) * 5.0  # Larger initial covariance to reflect uncertainty

# Measurement noise covariance in polar coordinates for radar
R_radar_polar = np.array([[0.3**2, 0],      # Distance noise variance
                          [0, 0.05**2]])     # Angle noise variance

# Transform radar noise covariance from polar to Cartesian using Jacobian
J = polar_to_cartesian_jacobian(z_radar_polar[0], z_radar_polar[1])
R_radar = J @ R_radar_polar @ J.T  # Properly transformed noise covariance

# Camera measurement noise covariance
R_camera = np.eye(2) * 0.1**2  # Camera measurement noise (variance)

# State transition matrix (Identity for simplicity in this case)
F = np.eye(2)

# Process noise covariance (small uncertainty in the model)
Q = np.eye(2) * 0.01  # Reduced process noise for better tracking

# Kalman filter functions
def kalman_predict(x, P, F, Q):
    x_pred = np.dot(F, x)
    P_pred = np.dot(np.dot(F, P), F.T) + Q
    return x_pred, P_pred

def kalman_update(x_pred, P_pred, z, H, R):
    y = z - np.dot(H, x_pred)  # Innovation
    S = np.dot(np.dot(H, P_pred), H.T) + R  # Innovation covariance
    K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))  # Kalman gain
    x_new = x_pred + np.dot(K, y)  # State update
    P_new = P_pred - np.dot(np.dot(K, H), P_pred)  # Covariance update
    return x_new, P_new

# Measurement models for radar and camera (both measure x, y directly after conversion)
H_radar = np.eye(2)  # Radar measurement matrix (after polar to Cartesian conversion)
H_camera = np.eye(2)  # Camera measurement matrix (directly measures x, y)

# Predict step
x_pred, P_pred = kalman_predict(x, P, F, Q)

# Sequential update: First update with radar, then with camera (sensor fusion)
x_after_radar, P_after_radar = kalman_update(x_pred, P_pred, z_radar, H_radar, R_radar)
x_fused, P_fused = kalman_update(x_after_radar, P_after_radar, z_camera, H_camera, R_camera)

# Also compute individual estimates for comparison
x_new_radar, P_new_radar = kalman_update(x_pred, P_pred, z_radar, H_radar, R_radar)
x_new_camera, P_new_camera = kalman_update(x_pred, P_pred, z_camera, H_camera, R_camera)

# ========== PDA算法执行 ==========
print("\n" + "=" * 60)
print("PDA算法执行 (PDA Algorithm Execution)")
print("=" * 60)

# 生成候选测量值（包括真实测量和杂波）
candidate_measurements = generate_candidate_measurements(true_position, num_clutter=4, 
                                                          measurement_range=20, noise_level=0.1)
print(f"候选测量值数量 (Number of Candidate Measurements): {len(candidate_measurements)}")
print(f"候选测量值 (Candidate Measurements):")
for i, z in enumerate(candidate_measurements):
    print(f"  测量 {i+1}: [{z[0]:.4f}, {z[1]:.4f}]")

# 使用相机测量模型进行PDA（因为相机直接测量x,y坐标）
H_pda = np.eye(2)
R_pda = R_camera

# 门控步骤
validated_measurements, innovations, innovation_covariances = gating(
    candidate_measurements, x_pred, P_pred, H_pda, R_pda, gate_threshold=9.21
)

print(f"\n门控后有效测量值数量 (Validated Measurements): {len(validated_measurements)}")
if len(validated_measurements) > 0:
    print(f"有效测量值 (Validated Measurements):")
    for i, z in enumerate(validated_measurements):
        print(f"  有效测量 {i+1}: [{z[0]:.4f}, {z[1]:.4f}]")

# 计算关联概率
association_probs, prob_no_association = compute_association_probabilities(
    validated_measurements, x_pred, P_pred, H_pda, R_pda, pd=0.9, lambda_clutter=0.1
)

print(f"\n关联概率 (Association Probabilities):")
if len(association_probs) > 0:
    for i, prob in enumerate(association_probs):
        print(f"  测量 {i+1} 的关联概率: {prob:.4f} ({prob*100:.2f}%)")
print(f"  无关联概率 (No Association Probability): {prob_no_association:.4f} ({prob_no_association*100:.2f}%)")

# PDA更新
x_pda, P_pda = pda_update(x_pred, P_pred, validated_measurements, 
                          association_probs, prob_no_association, H_pda, R_pda)

print(f"\nPDA估计结果 (PDA Estimate): [{x_pda[0]:.4f}, {x_pda[1]:.4f}]")
print("=" * 60)

# Displaying the results
print("=" * 60)
print("目标跟踪结果 (Target Tracking Results)")
print("=" * 60)
print(f"真实位置 (True Position): {true_position}")
print(f"雷达测量 (极坐标) (Radar Measurement - Polar): {z_radar_polar}")
print(f"雷达测量 (笛卡尔坐标) (Radar Measurement - Cartesian): {z_radar}")
print(f"相机测量 (Camera Measurement): {z_camera}")
print("-" * 60)
print(f"仅雷达估计 (Radar Only Estimate): {x_new_radar}")
print(f"仅相机估计 (Camera Only Estimate): {x_new_camera}")
print(f"融合估计 (Fused Estimate): {x_fused}")
print(f"PDA估计 (PDA Estimate): {x_pda}")
print("=" * 60)

# Calculate errors
error_radar = np.linalg.norm(x_new_radar - true_position)
error_camera = np.linalg.norm(x_new_camera - true_position)
error_fused = np.linalg.norm(x_fused - true_position)
error_pda = np.linalg.norm(x_pda - true_position)

print(f"\n位置误差 (Position Errors):")
print(f"雷达误差 (Radar Error): {error_radar:.4f}")
print(f"相机误差 (Camera Error): {error_camera:.4f}")
print(f"融合误差 (Fused Error): {error_fused:.4f}")
print(f"PDA误差 (PDA Error): {error_pda:.4f}")

# 显示改进说明
print("\n" + "=" * 60)
print("优化说明 (Optimization Notes):")
print("=" * 60)
print("[*] 使用测量值初始化状态，而非从原点开始")
print("[*] 正确转换雷达极坐标噪声协方差矩阵（雅可比变换）")
print("[*] 降低雷达角度噪声水平（0.05 vs 0.5）")
print("[*] 增大初始协方差矩阵以反映真实不确定性")
print("[*] 使用方差而非标准差定义噪声协方差矩阵")
print("=" * 60)

# Plotting the results
plt.figure(figsize=(12, 10))

# True position of the target
plt.plot(true_position[0], true_position[1], 'go', markersize=12, label='真实位置 (True Position)', zorder=5)

# Measurements
plt.plot(z_radar[0], z_radar[1], 'rs', markersize=8, label='雷达测量 (Radar Measurement)', alpha=0.7)
plt.plot(z_camera[0], z_camera[1], 'bs', markersize=8, label='相机测量 (Camera Measurement)', alpha=0.7)

# PDA候选测量值
if len(candidate_measurements) > 0:
    plt.plot(candidate_measurements[:, 0], candidate_measurements[:, 1], 'kx', 
             markersize=6, label='PDA候选测量值 (PDA Candidates)', alpha=0.5)

# PDA有效测量值（门控内）
if len(validated_measurements) > 0:
    plt.plot(validated_measurements[:, 0], validated_measurements[:, 1], 'yo', 
             markersize=10, label='PDA有效测量值 (PDA Validated)', alpha=0.8, zorder=3)

# Estimated positions
plt.plot(x_new_radar[0], x_new_radar[1], 'r^', markersize=10, label='仅雷达估计 (Radar Only)', alpha=0.8)
plt.plot(x_new_camera[0], x_new_camera[1], 'b^', markersize=10, label='仅相机估计 (Camera Only)', alpha=0.8)
plt.plot(x_fused[0], x_fused[1], 'm*', markersize=15, label='融合估计 (Fused Estimate)', zorder=4)
plt.plot(x_pda[0], x_pda[1], 'cD', markersize=12, label='PDA估计 (PDA Estimate)', zorder=4)

# Draw error circles (uncertainty ellipses simplified as circles)
circle_radar = plt.Circle((x_new_radar[0], x_new_radar[1]), np.sqrt(P_new_radar[0, 0] + P_new_radar[1, 1]), 
                         color='r', fill=False, linestyle='--', alpha=0.3, label='雷达不确定性 (Radar Uncertainty)')
circle_camera = plt.Circle((x_new_camera[0], x_new_camera[1]), np.sqrt(P_new_camera[0, 0] + P_new_camera[1, 1]), 
                          color='b', fill=False, linestyle='--', alpha=0.3, label='相机不确定性 (Camera Uncertainty)')
circle_fused = plt.Circle((x_fused[0], x_fused[1]), np.sqrt(P_fused[0, 0] + P_fused[1, 1]), 
                         color='m', fill=False, linestyle='--', alpha=0.5, linewidth=2, label='融合不确定性 (Fused Uncertainty)')
circle_pda = plt.Circle((x_pda[0], x_pda[1]), np.sqrt(P_pda[0, 0] + P_pda[1, 1]), 
                       color='c', fill=False, linestyle=':', alpha=0.6, linewidth=2, label='PDA不确定性 (PDA Uncertainty)')
plt.gca().add_patch(circle_radar)
plt.gca().add_patch(circle_camera)
plt.gca().add_patch(circle_fused)
plt.gca().add_patch(circle_pda)

# Draw validation gate (ellipse)
if len(validated_measurements) > 0:
    S = H_pda @ P_pred @ H_pda.T + R_pda
    eigenvals, eigenvecs = np.linalg.eigh(S)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width = 2 * np.sqrt(9.21) * np.sqrt(eigenvals[0])  # gate_threshold = 9.21
    height = 2 * np.sqrt(9.21) * np.sqrt(eigenvals[1])
    gate_ellipse = Ellipse((x_pred[0], x_pred[1]), width, height, angle=angle,
                          fill=False, linestyle='-.', color='orange', alpha=0.4, linewidth=1.5,
                          label='门控区域 (Validation Gate)')
    plt.gca().add_patch(gate_ellipse)

# Labels and legend
plt.title('目标跟踪：雷达与相机融合 + PDA算法\nTarget Tracking with Radar/Camera Fusion + PDA Algorithm', fontsize=12)
plt.xlabel('X 位置 (X Position)', fontsize=11)
plt.ylabel('Y 位置 (Y Position)', fontsize=11)
plt.legend(loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Show the plot
plt.tight_layout()
plt.show()
