"""在录制前检查 MuJoCo 场景中的抓放目标是否可达。

这个脚本只做预检，不连接真实设备，也不会修改 XML。它使用和
``MyArmAuto``/``KeyboardIKActionStep`` 相同的 FK、IK 和 joint limit。

示例：

    conda activate lerobot
    python /home/mia/data/smolvla_work/projects/user/check_reachability.py
    python /home/mia/data/smolvla_work/projects/user/check_reachability.py \
        --task-id 1 --pick-offset-base 0 0 -0.055 \
        --place-offset-base 0 0 0.015

``pick-offset-base`` 和 ``place-offset-base`` 是从 AprilTag 中心到 TCP
目标的偏移，坐标系是 base，单位是米；应与 ``MyArmAutoConfig`` 中的值一致。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import mujoco
import numpy as np


# 允许直接用 ``python projects/user/check_reachability.py`` 运行；正常的
# lerobot 环境已经通过 editable install 提供这两个包，因此不会重复加载。
PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "my_plugin"
KINEMATICS_SRC = PLUGIN_ROOT / "mia_kinematics" / "src"
if KINEMATICS_SRC.is_dir():
    sys.path.insert(0, str(KINEMATICS_SRC))

from mia_kinematics import FK_learn, IK_learn  # noqa: E402
from lerobot_robot_myarm.myarm import (  # noqa: E402
    get_T_joint_to_site,
    get_joint_initial_info,
)
from lerobot_robot_myarm.task_tags import (  # noqa: E402
    default_tasklist,
    resolve_task_tags,
)


DEFAULT_XML = Path("/home/mia/data/smolvla_work/projects/models/world.xml")

# AprilTag 与 world.xml 中 tag geom 的对应关系。Tag 几何体的位置就是视觉
# 检测得到的 tag 中心；抓取 TCP 的实际高度由 --pick-offset-base 修正。
TAG_GEOM_NAMES = {
    0: "cube_small_tag",
    1: "cube_large_tag",
    2: "plate_small_tag",
    3: "plate_large_tag",
    4: "cup_small_tag",
    5: "cup_large_tag",
}

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ACTUATOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)


@dataclass(frozen=True)
class TargetPath:
    name: str
    points: tuple[np.ndarray, ...]
    labels: tuple[str, ...]


@dataclass
class IKChecker:
    model: mujoco.MjModel
    data: mujoco.MjData
    transforms_relative: np.ndarray
    axes_relative: np.ndarray
    T6ee: np.ndarray
    qpos_addresses: np.ndarray
    ctrl_min: np.ndarray
    ctrl_max: np.ndarray
    home_q: np.ndarray
    home_rpy: np.ndarray
    base_body_id: int
    robot_body_ids: set[int]
    obstacle_geom_ids: set[int]

    def fk(self, q: np.ndarray) -> np.ndarray:
        T_joints, _ = FK_learn.Tbn(
            q,
            self.transforms_relative,
            self.axes_relative,
        )
        return T_joints[-1] @ self.T6ee

    def solve(self, q_init: np.ndarray, position: np.ndarray) -> tuple[np.ndarray, bool, float, float, int]:
        """以固定的 home 末端姿态求一个目标位置的 IK。"""
        q, success, iterations = IK_learn.inverse_kinematics(
            q_init=q_init,
            pos_target=position,
            rpy_target=self.home_rpy,
            transforms_relative=self.transforms_relative,
            axes_relative=self.axes_relative,
            T6ee=self.T6ee,
            pos_tolerance=1e-4,
            rot_tolerance=1e-3,
        )
        q = np.asarray(q, dtype=float)
        if q.shape != self.home_q.shape or not np.all(np.isfinite(q)):
            return q, False, float("inf"), float("inf"), int(iterations)

        T = self.fk(q)
        position_error = float(np.linalg.norm(T[:3, 3] - position))
        rotation_error = float(
            np.linalg.norm(IK_learn.pose_error(T, position, self.home_rpy)[3:])
        )
        within_limits = np.all(q >= self.ctrl_min) and np.all(q <= self.ctrl_max)
        ok = bool(success and within_limits and position_error <= 1e-3 and rotation_error <= 1e-2)
        return q, ok, position_error, rotation_error, int(iterations)

    def set_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_addresses] = q
        # 保持夹爪张开，避免预检中的夹爪自碰撞影响结果。
        if self.data.qpos.size >= 8:
            self.data.qpos[6:8] = 0.04
        mujoco.mj_forward(self.model, self.data)

    def obstacle_contacts(self, q: np.ndarray) -> list[str]:
        """返回机械臂自碰撞或与桌面/地面/平台的碰撞；物体接触不作为预检失败。

        在抓取和放置终点，夹爪与物体的接触是预期行为，因此这里仅检查
        robot-vs-static-obstacle；机械臂自碰撞也会报告。物体接触被忽略。
        """
        self.set_q(q)
        mujoco.mj_collision(self.model, self.data)
        names: list[str] = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            robot1, robot2 = g1 in self.robot_geom_ids, g2 in self.robot_geom_ids
            if robot1 and robot2:
                n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom#{g1}"
                n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom#{g2}"
                names.append(f"self({n1},{n2})")
                continue
            if not (robot1 or robot2):
                continue
            obstacle_geom = g2 if robot1 else g1
            if obstacle_geom not in self.obstacle_geom_ids:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, obstacle_geom)
            names.append(name or f"geom#{obstacle_geom}")
        return sorted(set(names))

    @property
    def robot_geom_ids(self) -> set[int]:
        return {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in self.robot_body_ids
            and self.model.geom_contype[geom_id] != 0
        }


def _body_is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    current = body_id
    # MuJoCo 的 world body 是 0，它的 parent 仍然是 0，不能只用
    # ``current >= 0`` 作为终止条件。
    while True:
        if current == ancestor_id:
            return True
        if current == 0:
            return False
        current = int(model.body_parentid[current])
    return False


def make_checker(xml_path: Path) -> IKChecker:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise ValueError("XML 中不存在名为 home 的 keyframe")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES],
        dtype=int,
    )
    actuator_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES],
        dtype=int,
    )
    if np.any(joint_ids < 0) or np.any(actuator_ids < 0):
        raise ValueError("XML 中缺少 MyArm 配置的关节或 actuator")
    qpos_addresses = model.jnt_qposadr[joint_ids].copy()
    _, transforms_relative, axes_relative = get_joint_initial_info(model, list(JOINT_NAMES))
    T6ee = get_T_joint_to_site(model, JOINT_NAMES[-1], "attachment_site")
    home_q = data.qpos[qpos_addresses].copy()
    T_home = FK_learn.compose_transform(
        FK_learn.Tbn(home_q, transforms_relative, axes_relative)[0][-1],
        T6ee,
    )
    _, home_rpy = FK_learn.transfotm2Cartesian(T_home)

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_body_id < 0:
        raise ValueError("XML 中不存在 base body")
    robot_body_ids = {
        body_id
        for body_id in range(model.nbody)
        if _body_is_descendant(model, body_id, base_body_id)
    }

    # 只把静态支撑物作为 obstacle；物体与盘子接触在抓放任务中是预期的。
    static_names = {"ground", "white_table", "work_platform", "robot_pedestal"}
    obstacle_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) in static_names
    }
    return IKChecker(
        model=model,
        data=data,
        transforms_relative=np.asarray(transforms_relative),
        axes_relative=np.asarray(axes_relative),
        T6ee=np.asarray(T6ee),
        qpos_addresses=qpos_addresses,
        ctrl_min=model.actuator_ctrlrange[actuator_ids, 0].copy(),
        ctrl_max=model.actuator_ctrlrange[actuator_ids, 1].copy(),
        home_q=home_q,
        home_rpy=np.asarray(home_rpy),
        base_body_id=base_body_id,
        robot_body_ids=robot_body_ids,
        obstacle_geom_ids=obstacle_geom_ids,
    )


def tag_position_in_base(checker: IKChecker, tag_id: int) -> np.ndarray:
    geom_name = TAG_GEOM_NAMES[tag_id]
    geom_id = mujoco.mj_name2id(checker.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        raise ValueError(f"XML 中找不到 Tag geom：{geom_name}")

    base_pos = checker.data.xpos[checker.base_body_id]
    base_rot = checker.data.xmat[checker.base_body_id].reshape(3, 3)
    p_world_tag = checker.data.geom_xpos[geom_id]
    return base_rot.T @ (p_world_tag - base_pos)


def interpolate(a: np.ndarray, b: np.ndarray, step: float) -> list[np.ndarray]:
    distance = float(np.linalg.norm(b - a))
    count = max(1, int(np.ceil(distance / step)))
    return [a + (b - a) * (i / count) for i in range(1, count + 1)]


def make_path(home_position: np.ndarray, pick: np.ndarray, place: np.ndarray, clearance: float, step: float) -> TargetPath:
    travel_z = max(float(pick[2]), float(place[2])) + clearance
    above_pick = pick.copy()
    above_place = place.copy()
    above_pick[2] = above_place[2] = travel_z
    waypoints = [
        ("home_to_above_pick", home_position, above_pick),
        ("above_pick_to_pick", above_pick, pick),
        ("pick_to_above_pick", pick, above_pick),
        ("above_pick_to_above_place", above_pick, above_place),
        ("above_place_to_place", above_place, place),
        ("place_to_above_place", place, above_place),
    ]
    points: list[np.ndarray] = []
    labels: list[str] = []
    for label, start, end in waypoints:
        segment = interpolate(start, end, step)
        points.extend(segment)
        labels.extend([label] * len(segment))
    return TargetPath(name=" -> ".join(labels), points=tuple(points), labels=tuple(labels))


def check_task(
    checker: IKChecker,
    task_id: int,
    task_text: str,
    pick_offset: np.ndarray,
    place_offset: np.ndarray,
    clearance: float,
    path_step: float,
    max_joint_step: float,
    limit_margin: float,
    check_collision: bool,
) -> bool:
    tags = resolve_task_tags(task_id, default_tasklist())
    object_tag = tag_position_in_base(checker, tags.object_tag_id)
    plate_tag = tag_position_in_base(checker, tags.plate_tag_id)
    pick = object_tag + pick_offset
    place = plate_tag + place_offset
    home_position = checker.fk(checker.home_q)[:3, 3]
    path = make_path(home_position, pick, place, clearance, path_step)

    q = checker.home_q.copy()
    passed = True
    first_failure = ""
    for index, position in enumerate(path.points, start=1):
        q_next, ik_ok, pos_error, rot_error, iterations = checker.solve(q, position)
        near_limit = np.any(q_next < checker.ctrl_min + limit_margin) or np.any(q_next > checker.ctrl_max - limit_margin)
        joint_jump = float(np.max(np.abs(q_next - q))) if q_next.shape == q.shape else float("inf")
        collision_names = checker.obstacle_contacts(q_next) if check_collision and ik_ok else []
        point_ok = ik_ok and not near_limit and joint_jump <= max_joint_step and not collision_names
        if not point_ok and passed:
            failure_bits = []
            if not ik_ok:
                failure_bits.append(f"IK 失败(pos={pos_error:.4g}, rot={rot_error:.4g}, iter={iterations})")
            if near_limit:
                failure_bits.append("接近关节限位")
            if joint_jump > max_joint_step:
                failure_bits.append(f"关节跳变 {joint_jump:.4g} rad")
            if collision_names:
                failure_bits.append("碰撞:" + ",".join(collision_names))
            segment = path.labels[index - 1]
            first_failure = f"{segment} 的第 {index} 个路径点：" + "；".join(failure_bits)
            passed = False
            # 后续路径点的 IK 必须以上一个有效关节解为种子；一旦当前点
            # 已经失败，继续计算只会浪费时间，也不能证明后续路径可行。
            break
        if ik_ok:
            q = q_next

    print(
        f"[{ 'PASS' if passed else 'FAIL' }] task {task_id:2d}: {task_text} "
        f"| object_tag={tags.object_tag_id}, plate_tag={tags.plate_tag_id}"
    )
    print(f"       pick={np.round(pick, 4)}, place={np.round(place, 4)}, points={len(path.points)}")
    if first_failure:
        print(f"       {first_failure}")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--task-id", type=int, default=0, help="1～16；0 表示检查全部任务")
    parser.add_argument("--pick-offset-base", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--place-offset-base", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--clearance", type=float, default=0.10)
    parser.add_argument("--path-step", type=float, default=0.005, help="笛卡尔路径采样间隔，单位米")
    parser.add_argument("--max-joint-step", type=float, default=0.10, help="相邻采样点允许的最大关节变化，单位 rad")
    parser.add_argument("--limit-margin", type=float, default=0.05, help="距离关节限位的最小安全余量，单位 rad")
    parser.add_argument("--check-collision", action="store_true", help="同时检查机械臂与桌面/地面/平台的碰撞")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.xml_path.is_file():
        raise FileNotFoundError(f"找不到 XML：{args.xml_path}")
    if args.task_id < 0 or args.task_id > 16:
        raise ValueError("--task-id 必须是 0～16")
    if args.clearance <= 0 or args.path_step <= 0 or args.max_joint_step <= 0 or args.limit_margin < 0:
        raise ValueError("clearance/path-step/max-joint-step 必须为正，limit-margin 不能为负")

    checker = make_checker(args.xml_path.expanduser().resolve())
    tasklist = default_tasklist()
    task_ids = [args.task_id] if args.task_id else list(range(1, len(tasklist) + 1))
    pick_offset = np.asarray(args.pick_offset_base, dtype=float)
    place_offset = np.asarray(args.place_offset_base, dtype=float)
    # 16 条语言任务中，颜色/大小别名经常指向同一个物体；相同的
    # (object_tag, plate_tag) 只做一次昂贵的路径 IK，随后复用结果。
    cache: dict[tuple[int, int], bool] = {}
    results: list[bool] = []
    for task_id in task_ids:
        tags = resolve_task_tags(task_id, tasklist)
        key = (tags.object_tag_id, tags.plate_tag_id)
        if key in cache:
            result = cache[key]
            status = "PASS" if result else "FAIL"
            print(
                f"[{status}] task {task_id:2d}: {tasklist[task_id - 1]} "
                f"| object_tag={key[0]}, plate_tag={key[1]}（复用相同物理目标的预检结果）"
            )
        else:
            result = check_task(
                checker,
                task_id,
                tasklist[task_id - 1],
                pick_offset,
                place_offset,
                args.clearance,
                args.path_step,
                args.max_joint_step,
                args.limit_margin,
                args.check_collision,
            )
            cache[key] = result
        results.append(result)
    passed = sum(results)
    print(f"\n结果：{passed}/{len(results)} 个任务通过。")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
