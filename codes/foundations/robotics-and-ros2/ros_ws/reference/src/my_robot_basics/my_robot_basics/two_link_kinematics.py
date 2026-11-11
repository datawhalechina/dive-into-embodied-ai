# two_link_kinematics.py

import math


def forward_kinematics(theta1_deg, theta2_deg, L1=1.0, L2=1.0):
    """
    两连杆平面机械臂正运动学

    参数：
        theta1_deg: 关节1角度（度），相对 base 坐标系
        theta2_deg: 关节2角度（度），相对连杆1坐标系
        L1: 连杆1长度（米），默认 1.0
        L2: 连杆2长度（米），默认 1.0

    返回：
        (x, y): 末端位置（米）
    """
    theta1 = math.radians(theta1_deg)
    theta2 = math.radians(theta2_deg)

    x = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
    y = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)

    return x, y


def inverse_kinematics(x, y, L1=1.0, L2=1.0):
    """
    两连杆平面机械臂逆运动学

    参数：
        x, y: 目标末端位置（米）
        L1: 连杆1长度（米），默认 1.0
        L2: 连杆2长度（米），默认 1.0

    返回：
        dict，包含两组解：
        {
            "elbow_down": (theta1_deg, theta2_deg),  # 肘下解，theta2 > 0
            "elbow_up":   (theta1_deg, theta2_deg),  # 肘上解，theta2 < 0
        }
        如果目标不可达，抛出 ValueError

    注意：
        返回角度单位为度
    """
    if not all(math.isfinite(v) for v in (x, y, L1, L2)) or L1 <= 0 or L2 <= 0:
        raise ValueError('Coordinates must be finite and link lengths positive')
    r_sq = x**2 + y**2
    r = math.sqrt(r_sq)

    # 可达性检查
    r_max = L1 + L2
    r_min = abs(L1 - L2)
    if r > r_max and not math.isclose(r, r_max, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"目标点 ({x:.3f}, {y:.3f}) 超出工作空间："
            f"距离 {r:.3f} m > 最大可达 {r_max:.3f} m"
        )
    if r < r_min and not math.isclose(r, r_min, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"目标点 ({x:.3f}, {y:.3f}) 在死区内："
            f"距离 {r:.3f} m < 最小可达 {r_min:.3f} m"
        )

    # 用余弦定理求 θ2
    cos_theta2 = (r_sq - L1**2 - L2**2) / (2 * L1 * L2)
    # 防止浮点误差导致 acos 输入略微超出 [-1, 1]
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))

    theta2_down = math.acos(cos_theta2)   # 肘下解：θ2 > 0
    theta2_up   = -theta2_down            # 肘上解：θ2 < 0

    def solve_theta1(theta2):
        k1 = L1 + L2 * math.cos(theta2)
        k2 = L2 * math.sin(theta2)
        return math.atan2(y, x) - math.atan2(k2, k1)

    theta1_down = solve_theta1(theta2_down)
    theta1_up   = solve_theta1(theta2_up)

    return {
        "elbow_down": (math.degrees(theta1_down), math.degrees(theta2_down)),
        "elbow_up":   (math.degrees(theta1_up),   math.degrees(theta2_up)),
    }
