import numpy as np
from scipy.spatial.transform import Rotation as Rs
from . import FK_learn
# 当前各关节角度
q = np.array([
    1.23,
    2.1,
    0.3,
    -0.3,
    0.5,
    -1.7])

# 控制周期
dt = 0.01
# 每个关节角速度限制
q_dot_max = np.array([
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5
])
# 末端速度限制
v_max = 0.1       # m/s
w_max = 0.5       # rad/s
# 速度增益
K = np.diag([
    2.0, 2.0, 2.0,
    1.0, 1.0, 1.0
])
Tb1 = FK_learn.Cartesian2transform(
    [0.0, 0.0, 0.3],
    [0.0, 0.0, 0.0]
)

T12 = FK_learn.Cartesian2transform(
    [0.3, 0.0, 0.0],
    [np.pi/2, 0.0, 0.0]
)

T23 = FK_learn.Cartesian2transform(
    [0.3, 0.0, 0.0],
    [0.0, np.pi/2, 0.0]
)

T34 = FK_learn.Cartesian2transform(
    [0.2, 0.0, 0.0],
    [np.pi/2, 0.0, 0.0]
)

T45 = FK_learn.Cartesian2transform(
    [0.15, 0.0, 0.0],
    [0.0, np.pi/2, 0.0]
)

T56 = FK_learn.Cartesian2transform(
    [0.1, 0.0, 0.0],
    [0.0, 0.0, 0.0]
)



# # 当前的T，而不是初始0位的
# T_joints = FK_learn.Tbn(q,Tb1,T12,T23,T34,T45,T56)
# T_ee = T_joints[-1]

# 雅可比矩阵的作用是将关节速度转换成末端的线速度和角速度
# 这里用的是6轴机械臂做例子，假设第 i 个关节绕 zi−1为q
# 雅可比矩阵的上半部分线速度第i列Jvi = zi-1（pe-pi-1）
# 下半部分角速度 = zi
# pe末端执行器当前在 base 下的位置
# pi第i个关节原点当前在base下的位置
# zi：第i个关节旋转轴在base坐标系下的单位方向向量
# zi * q_dot[i] 才是该关节产生的角速度向量
# 一般是已知q和Tb1-T56,还有q_dot(关节速度)
def geometric_jacobian(
    T_joints,
    T_ee,
    axes_base,
):
    n = len(T_joints)
    J = np.zeros((6, n))

    p_ee = T_ee[:3, 3]

    for i in range(n):
        p_i = T_joints[i][:3, 3]
        axis_i = axes_base[i]

        J[:3, i] = np.cross(
            axis_i,
            p_ee - p_i,
        )

        J[3:, i] = axis_i

    return J

def q_dot2eevw(J,q_dot):
    vw = J@q_dot
    v = vw[:3]
    w = vw[3:]
    return v,w




# 差分验证
def numerical_position_jacobian(q, Tb1, T12, T23, T34, T45, T56):

    epsilon = 1e-6

    T0 = FK_learn.Tbn(
        q, Tb1, T12, T23, T34, T45, T56
    )

    p0 = T0[-1][:3, 3]

    J_num = np.zeros((3, 6))

    for i in range(6):

        q_new = q.copy()
        q_new[i] += epsilon

        T_new = FK_learn.Tbn(
            q_new,
            Tb1, T12, T23,
            T34, T45, T56
        )

        p_new = T_new[-1][:3, 3]

        J_num[:, i] = (p_new - p0) / epsilon

    return J_num
def numerical_angular_jacobian(
    q,
    Tb1, T12, T23, T34, T45, T56
):
    epsilon = 1e-6

    # 当前姿态
    T0 = FK_learn.Tbn(
        q,
        Tb1, T12, T23,
        T34, T45, T56
    )

    R0 = T0[-1][:3, :3]

    Jw_num = np.zeros((3, 6))

    for i in range(6):

        # 只让第 i 个关节增加 epsilon
        q_new = q.copy()
        q_new[i] += epsilon

        T_new = FK_learn.Tbn(
            q_new,
            Tb1, T12, T23,
            T34, T45, T56
        )

        R_new = T_new[-1][:3, :3]

        # 在 base frame 下的相对旋转
        R_delta = R_new @ R0.T

        # 微小旋转矩阵 -> rotation vector
        rotvec = Rs.from_matrix(
            R_delta
        ).as_rotvec()

        # 数值角速度 Jacobian
        Jw_num[:, i] = rotvec / epsilon

    return Jw_num
# 阻尼伪逆：当 \(J\) 接近不可逆时，避免关节速度突然变得非常大。
# 代价是末端速度不一定完全等于目标速度，但系统会更加稳定
def damped_pseudoinverse(J, damping=0.05):
    n = J.shape[1]
    return J.T @ np.linalg.inv(
        J @ J.T + damping**2 * np.eye(J.shape[0])
    )

# rpy的error处理
# 因为欧拉角存在角度跳变和万向节锁
# 因此需要先用旋转矩阵计算旋转，再转换成rotation vector->角速度误差
def pose_error(T_current, pos_target,rpy_target):
    T_target = FK_learn.Cartesian2transform(pos_target,rpy_target)
    p_current = T_current[:3, 3]
    p_target = T_target[:3,3]

    # 位置误差
    e_position = p_target - p_current

    R_current = T_current[:3, :3]
    R_target = T_target[:3,:3]

    # 当前姿态旋转到目标姿态
    R_error = R_target @ R_current.T

    # 旋转矩阵转旋转向量变成rpy
    e_rotation = Rs.from_matrix(R_error).as_rotvec()

    return np.concatenate([e_position, e_rotation])
#速度IK：延某个方向运动一个周期产生的q
def v_inverse_kinematics(
    q_init,
    transforms_relative,
    axes_relative,
    T6ee,
    x_dot,
    dt=0.01,
    q_dot_max=0.5,
):
    q = np.asarray(
        q_init,
        dtype=float,
    ).copy()

    T_joints, axes_base = FK_learn.Tbn(
        q,
        transforms_relative,
        axes_relative,
    )

    # joint6 还不是 attachment_site
    T_ee = T_joints[-1] @ T6ee

    J = geometric_jacobian(
        T_joints,
        T_ee,
        axes_base,
    )

    J_pinv = damped_pseudoinverse(J)

    q_dot = J_pinv @ x_dot

    q_dot = np.clip(
        q_dot,
        -q_dot_max,
        q_dot_max,
    )

    q_next = q + q_dot * dt

    return q_next
# 位置Ik：q_init:当前关节值，pos_target , rpy_target ：目标pose，dt：控制周期
def inverse_kinematics(
    q_init,
    pos_target,
    rpy_target,
    transforms_relative,
    axes_relative,
    T6ee,
    K = np.diag([
        2.0, 2.0, 2.0,
        1.0, 1.0, 1.0
    ]),
    max_steps=1000,
    pos_tolerance=0.001,
    rot_tolerance=0.01,
    dt=0.01,
    q_dot_max=0.5,
):
    q = np.asarray(
        q_init,
        dtype=float,
    ).copy()

    for step in range(max_steps):
        # 新版 FK：同时返回关节位姿和世界/base坐标系下的轴
        T_joints, axes_base = FK_learn.Tbn(
            q,
            transforms_relative,
            axes_relative,
        )

        # 第六关节原点 -> attachment_site
        T_current = T_joints[-1] @ T6ee

        # 六维误差：[位置误差, 姿态误差]
        e = pose_error(
            T_current,
            pos_target,
            rpy_target,
        )

        position_error = np.linalg.norm(e[:3])
        rotation_error = np.linalg.norm(e[3:])

        if (
            position_error < pos_tolerance
            and rotation_error < rot_tolerance
        ):
            return q, True, step

        J = geometric_jacobian(
            T_joints,
            T_current,
            axes_base,
        )

        J_pinv = damped_pseudoinverse(J)

        q_dot = J_pinv @ (K @ e)

        q_dot = np.clip(
            q_dot,
            -q_dot_max,
            q_dot_max,
        )

        q = q + q_dot * dt

    return q, False, max_steps

# if __name__ == "__main__":

#     # 第一部分：当前各关节速度对末端速度产生的影响
#     q_dot = np.array([
#         0.2,
#         0.3,
#         -0.1,
#         0.5,
#         0.2,
#         -0.3
#     ])
#     J = geometric_jacobian(T_joints, T_ee)

#     J_num = numerical_position_jacobian(
#         q,
#         Tb1, T12, T23,
#         T34, T45, T56
#     )
#     Jw_num = numerical_angular_jacobian(
#     q,
#     Tb1, T12, T23,
#     T34, T45, T56
# )
    
#     print("Geometric:")
#     print(J[:3, :])

#     print("Finite difference:")
#     print(J_num)
#     print("error:")
#     print(J[:3, :] - J_num)

#     print("Finite difference Jw:")
#     print(Jw_num)
#     print("Jw error:")
#     print(J[3:, :] - Jw_num)

#     v, w = q_dot2eevw(J, q_dot)

#     print("v =", v)
#     print("w =", w)
#     print("rank =", np.linalg.matrix_rank(J))
#     print("condition number =", np.linalg.cond(J))
#     # 如果想在瞬时意义上独立控制末端的 6 个自由度：理想情况下需要：rank = 6
#     # 现在只有 5，意味着末端的 6 个速度自由度里面，有一个方向无法独立产生。
#     # 是因为J[:,0]=J[:,1]也就是 joint 1 和 joint 2 对末端产生完全一样的瞬时运动效果。
#     # 因为雅可比矩阵几乎不可逆

#     # DifferentialIK就是反过来，从末端的运动差->关节的角速度，但是奇异姿态导致rank(J)=5，不能简单求逆
#     # 改成J+，叫 Moore-Penrose pseudoinverse
#     # 比如人为指定末端速度
#     x_dot = np.array([
#         0.05,   # vx m/s
#         0.00,   # vy
#         0.00,   # vz
#         0.00,   # wx rad/s
#         0.00,   # wy
#         0.00    # wz
#     ])
#     # 得到下一次的末端q值
#     q_next = v_inverse_kinematics(q_init = q,x_dot = x_dot)
   

#     # 或者位置Ik
#     # 当前末端pos
#     T_joints = FK_learn.Tbn(q,Tb1,T12,T23,T34,T45,T56)
#     T_ee = T_joints[-1]
#     p_current,r_current = FK_learn.transfotm2Cartesian(T_ee)
#     # 目标末端pos,rpy
#     p_target = np.array([
#         p_current[0] + 0.05,
#         p_current[1],
#         p_current[2]
#     ])
#     r_target = np.array([
#             r_current[0] + 0.5,
#             r_current[1],
#             r_current[2]
#         ])
#     q_solution, success, steps = inverse_kinematics(
#         q_init=q,
#         pos_target=p_target,
#         rpy_target = r_target
#     )
#     T_joints_solution  = FK_learn.Tbn(q_solution,Tb1,T12,T23,T34,T45,T56)
#     T_ee_solution = T_joints_solution[-1]
#     p_solution,r_solution=FK_learn.transfotm2Cartesian(T_ee_solution)
#     print(f"pos_target:{p_target},rpy_target = {r_target},sulution:{q_solution},pos_current:{p_solution},rpy_current = {r_solution},success:{success}")
        
        

