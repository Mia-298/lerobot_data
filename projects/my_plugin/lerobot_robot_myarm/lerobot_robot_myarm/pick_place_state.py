"""逐帧抓放状态机。"""
from enum import Enum, auto

import numpy as np


class PickPlaceState(Enum):
    WAIT_TARGET = auto()
    MOVE_ABOVE_PICK = auto()
    MOVE_TO_PICK = auto()
    GRASP = auto()
    LIFT = auto()
    MOVE_ABOVE_PLACE = auto()
    MOVE_TO_PLACE = auto()
    RELEASE = auto()
    RETREAT = auto()
    DONE = auto()
    FAILED = auto()


class PickPlaceStateMachine:
    def __init__(self, clearance=0.10, tolerance=0.005, max_step=0.002):
        if not all(np.isfinite(v) and v > 0 for v in (clearance, tolerance, max_step)):
            raise ValueError("高度、到位容差和单步位移必须为正有限数")
        self.clearance = clearance
        self.tolerance = tolerance
        self.max_step = max_step
        self.reset()

    def reset(self):
        self.state = PickPlaceState.WAIT_TARGET
        self.targets = {}
        self.gripper_target = None
        self.failure_reason = None

    def fail(self, reason):
        self.failure_reason = reason
        self.state = PickPlaceState.FAILED

    @staticmethod
    def _position(value):
        position = np.asarray(value, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("TCP 位置必须为三个有限数值，单位米")
        return position.copy()

    def set_targets(self, pick_tcp, place_tcp,obj_height):
        # 相对base的pos
        pick = self._position(pick_tcp)
        place = self._position(place_tcp)+ [0,0,obj_height]
        # 搬运保持同一安全高度；默认假定机器人坐标系 Z 轴朝上。
        travel_z = max(pick[2], place[2]) + self.clearance
        above_pick = pick.copy()
        above_place = place.copy()
        above_pick[2] = above_place[2] = travel_z
        self.reset()
        self.targets = {
            PickPlaceState.MOVE_ABOVE_PICK: above_pick,
            PickPlaceState.MOVE_TO_PICK: pick,
            PickPlaceState.LIFT: above_pick,
            PickPlaceState.MOVE_ABOVE_PLACE: above_place,
            PickPlaceState.MOVE_TO_PLACE: place,
            PickPlaceState.RETREAT: above_place,
        }
        self.gripper_target = 1.0  # MyArm：1 张开，0 闭合
        self.state = PickPlaceState.MOVE_ABOVE_PICK

    def step(self, tcp_position, *, gripper_open=False, grasp_confirmed=False, release_confirmed=False):
        """每次只产生一个周期的动作；抓取成功必须由外部反馈确认。"""
        position = self._position(tcp_position)
        delta = np.zeros(3)
        state = self.state
        if state in self.targets:
            error = self.targets[state] - position
            distance = float(np.linalg.norm(error))
            next_states = {
                PickPlaceState.MOVE_ABOVE_PICK: PickPlaceState.MOVE_TO_PICK,
                PickPlaceState.MOVE_TO_PICK: PickPlaceState.GRASP,
                PickPlaceState.LIFT: PickPlaceState.MOVE_ABOVE_PLACE,
                PickPlaceState.MOVE_ABOVE_PLACE: PickPlaceState.MOVE_TO_PLACE,
                PickPlaceState.MOVE_TO_PLACE: PickPlaceState.RELEASE,
                PickPlaceState.RETREAT: PickPlaceState.DONE,
            }
            if distance <= self.tolerance:
                # 下降抓取前确认夹爪已张开。
                if state != PickPlaceState.MOVE_ABOVE_PICK or gripper_open:
                    self.state = next_states[state]
            else:
                delta = error * min(1.0, self.max_step / distance)
        elif state == PickPlaceState.GRASP:
            if grasp_confirmed:
                self.state = PickPlaceState.LIFT
        elif state == PickPlaceState.RELEASE:
            if release_confirmed:
                self.state = PickPlaceState.RETREAT

        if self.state == PickPlaceState.GRASP:
            self.gripper_target = 0.0
        elif self.state == PickPlaceState.RELEASE:
            self.gripper_target = 1.0

        # 现有 IK processor 将输入乘以 0.002，转换成米。
        action = dict(zip(("delta_x", "delta_y", "delta_z"), (delta / 0.002).tolist()))
        if self.gripper_target is not None:
            action["gripper.pos"] = self.gripper_target
        return action
