"""离线几何限高：只保护夹爪与指定水平面，不是全机械臂碰撞规划器。"""
from itertools import product

import mujoco
import numpy as np


class TableClearance:
    """以 base Z 为高度，使用全开合包络，避免闭合/张开时改变安全下限。"""

    def __init__(self, model, ee_name, gripper_joint_names, surface_geom_names,
                 margin=0.010, base_body_name="base"):
        if model is None:
            raise ValueError("安全限高需要已加载的 MuJoCo model，不能在 robot.connect() 前创建")
        self.margin = float(margin)
        if not np.isfinite(self.margin) or self.margin < 0:
            raise ValueError("桌面安全间隙必须是非负有限数，单位米")
        if not surface_geom_names:
            raise ValueError("自动抓取必须指定至少一个保护平面")

        def lookup(kind, name):
            idx = mujoco.mj_name2id(model, kind, name)
            if idx < 0:
                raise ValueError(f"安全限高找不到模型元素：{name}")
            return idx

        def ancestors(body):
            while body:
                yield body
                body = int(model.body_parentid[body])

        def require_fixed(body):
            if any(model.body_jntnum[b] for b in ancestors(body)):
                raise ValueError("安全限高要求 base 和保护平面固定在 world 中")

        base = lookup(mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        site = lookup(mujoco.mjtObj.mjOBJ_SITE, ee_name)
        require_fixed(base)
        # 独立 data：绝不改动控制循环正在使用的 qpos/data。
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        R_world_base = data.xmat[base].reshape(3, 3).copy()
        if not np.allclose(R_world_base[:, 2], [0, 0, 1], atol=1e-8):
            raise ValueError("安全限高要求 base Z 朝世界上方")
        p_world_base = data.xpos[base].copy()
        signs = np.asarray(list(product((-1., 1.), repeat=3)))

        surfaces = []
        for name in surface_geom_names:
            geom = lookup(mujoco.mjtObj.mjOBJ_GEOM, name)
            require_fixed(int(model.geom_bodyid[geom]))
            R = data.geom_xmat[geom].reshape(3, 3)
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX or not np.isclose(abs(R[2, 2]), 1, atol=1e-8):
                raise ValueError(f"保护平面必须是水平 box：{name}")
            corners = (signs * model.geom_size[geom]) @ R.T + data.geom_xpos[geom]
            surfaces.append(float(np.max(((corners - p_world_base) @ R_world_base)[:, 2])))
        # 使用最高表面的无限水平面；即使在平台之外也保守限高。
        self.surface_z_base = max(surfaces)

        joints = [lookup(mujoco.mjtObj.mjOBJ_JOINT, n) for n in gripper_joint_names]
        if not joints or len(set(joints)) != len(joints):
            raise ValueError("必须指定不重复的夹爪 slide 关节")
        for j in joints:
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE or not model.jnt_limited[j]:
                raise ValueError("夹爪包络要求具有限位的 slide 关节")
        geoms = [lookup(mujoco.mjtObj.mjOBJ_GEOM, n)
                 for n in ("gripper_palm", "left_pad", "right_pad")]
        site_body = int(model.site_bodyid[site])
        for geom in geoms:
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError("夹爪包络只支持 box 掌部与夹指")
            body = int(model.geom_bodyid[geom])
            while body != site_body:
                if body == 0:
                    raise ValueError("夹爪几何与末端 site 不在同一运动链")
                first = int(model.body_jntadr[body])
                if any(j not in joints for j in range(first, first + int(model.body_jntnum[body]))):
                    raise ValueError("夹爪包络存在未配置的活动关节")
                body = int(model.body_parentid[body])

        points = []
        for endpoints in product((0, 1), repeat=len(joints)):
            for joint, endpoint in zip(joints, endpoints):
                data.qpos[model.jnt_qposadr[joint]] = model.jnt_range[joint, endpoint]
            mujoco.mj_forward(model, data)
            R_world_ee = data.site_xmat[site].reshape(3, 3)
            for geom in geoms:
                R_world_geom = data.geom_xmat[geom].reshape(3, 3)
                corners = (signs * model.geom_size[geom]) @ R_world_geom.T + data.geom_xpos[geom]
                points.extend((corners - data.site_xpos[site]) @ R_world_ee)
        self.points_ee = np.asarray(points)
        if not np.all(np.isfinite(self.points_ee)):
            raise ValueError("夹爪包络包含非有限坐标")

    @staticmethod
    def _rotation(value):
        R = np.asarray(value, dtype=float)
        if (R.shape != (3, 3) or not np.all(np.isfinite(R))
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(R), 1, atol=1e-6)):
            raise ValueError("限高计算需要有效的 3×3 旋转矩阵")
        return R

    def minimum_ee_z(self, R_base_ee):
        R = self._rotation(R_base_ee)
        lowest_relative_z = float(np.min((self.points_ee @ R.T)[:, 2]))
        return self.surface_z_base + self.margin - lowest_relative_z

    def minimum_tcp_z(self, T_base_ee, T_ee_tcp):
        for T in (T_base_ee, T_ee_tcp):
            if np.shape(T) != (4, 4) or not np.all(np.isfinite(T)) or not np.allclose(T[3], [0, 0, 0, 1]):
                raise ValueError("限高计算需要有效的 4×4 位姿矩阵")
        R = self._rotation(T_base_ee[:3, :3])
        self._rotation(T_ee_tcp[:3, :3])
        return self.minimum_ee_z(R) + float((R @ T_ee_tcp[:3, 3])[2])

    def require_safe_ee(self, T_base_ee):
        floor = self.minimum_ee_z(T_base_ee[:3, :3])
        if not np.isfinite(T_base_ee[2, 3]) or T_base_ee[2, 3] < floor - 0.01:
            raise RuntimeError("夹爪已低于安全高度，停止自动下发；请人工检查并复位到安全位置")
