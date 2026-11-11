# lesson_02_transforms.py
# 运行方式：python3 lesson_02_transforms.py

import numpy as np


def rotation_matrix_2d(theta_deg):
    """生成 2D 旋转矩阵"""
    theta = np.radians(theta_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])


def homogeneous_transform_2d(theta_deg, tx, ty):
    """生成 2D 齐次变换矩阵"""
    theta = np.radians(theta_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta), tx],
        [np.sin(theta),  np.cos(theta), ty],
        [0,              0,              1]
    ])


def fk_via_matrix(L1, L2, theta1_deg, theta2_deg):
    """用矩阵连乘计算两连杆机械臂正运动学"""
    theta1 = np.radians(theta1_deg)
    theta2 = np.radians(theta2_deg)

    T_b_1 = np.array([
        [np.cos(theta1), -np.sin(theta1), L1 * np.cos(theta1)],
        [np.sin(theta1),  np.cos(theta1), L1 * np.sin(theta1)],
        [0,               0,              1]
    ])

    T_1_e = np.array([
        [np.cos(theta2), -np.sin(theta2), L2 * np.cos(theta2)],
        [np.sin(theta2),  np.cos(theta2), L2 * np.sin(theta2)],
        [0,               0,              1]
    ])

    T_b_e = T_b_1 @ T_1_e
    return T_b_e[0, 2], T_b_e[1, 2]


if __name__ == "__main__":
    print("=== 旋转矩阵演示 ===")
    R = rotation_matrix_2d(30)
    P = np.array([1, 0])
    print(f"点 (1,0) 旋转 30° 后: {(R @ P).round(4)}")

    print("\n=== 齐次变换矩阵演示 ===")
    T = homogeneous_transform_2d(45, 1, 0.5)
    P_h = np.array([2, 0, 1])
    result = T @ P_h
    print(f"点 (2,0) 旋转 45° 再平移 (1,0.5) 后: ({result[0]:.4f}, {result[1]:.4f})")

    print("\n=== 矩阵连乘 FK 演示 ===")
    configs = [
        (1.0, 0.8, 0,  0,   "两杆伸直"),
        (1.0, 0.8, 0,  90,  "第二杆向上弯 90°"),
        (1.0, 0.8, 30, 45,  "θ1=30°, θ2=45°"),
        (1.0, 0.8, 90, -90, "θ1=90°, θ2=-90°"),
    ]
    for L1, L2, t1, t2, desc in configs:
        x, y = fk_via_matrix(L1, L2, t1, t2)
        print(f"  {desc}: 末端 = ({x:.4f}, {y:.4f})")
