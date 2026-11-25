"""Base-Z grasp yaw and persistent measured AprilTag zero references (no robot I/O)."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

ROTATION_TARGET_KEY = "ee_rotation_target_base"
CUP_TAG_IDS = frozenset((4, 5))


def rotation_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6)):
        raise ValueError("姿态必须是有限的 3×3 SO(3) 旋转矩阵")
    return matrix.copy()


def rotation_error(current, target):
    return float(Rotation.from_matrix(rotation_matrix(target) @ rotation_matrix(current).T).magnitude())


def step_rotation(current, target, max_step_rad):
    """Bounded shortest SO(3) step, left-multiplied in base coordinates."""
    if not np.isfinite(max_step_rad) or max_step_rad <= 0:
        raise ValueError("单周期旋转步长必须为正有限数")
    current, target = rotation_matrix(current), rotation_matrix(target)
    vector = Rotation.from_matrix(target @ current.T).as_rotvec()
    angle = float(np.linalg.norm(vector))
    if angle <= max_step_rad:
        return target
    return Rotation.from_rotvec(vector * (max_step_rad / angle)).as_matrix() @ current


class AprilTagRotationReference:
    """First observed poses define zero; persisted poses are never re-zeroed per episode.

    Capture in the unrotated scene before enabling object-yaw randomization. Missing
    tags acquire their own reference when first visible. Invalid files fail closed.
    """
    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.ee_reference = None
        self.tags = {}
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text())
                if (payload["version"] != 1 or payload["frame"] != "robot_base"
                        or not isinstance(payload["tags_base"], dict)):
                    raise ValueError("不支持的基准格式")
                self.ee_reference = rotation_matrix(payload["ee_base"])
                self.tags = {str(int(key)): rotation_matrix(value)
                             for key, value in payload["tags_base"].items()}
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"AprilTag 旋转基准文件无效：{self.path}；请检查或在零旋转场景重新标定") from exc

    def capture_missing(self, tag_rotations, ee_rotation):
        # Validate the complete sample before changing anything.
        ee = rotation_matrix(ee_rotation)
        samples = {str(int(key)): rotation_matrix(value) for key, value in tag_rotations.items()}
        if not samples:
            return
        tags = {**self.tags, **{key: value for key, value in samples.items() if key not in self.tags}}
        if self.ee_reference is not None and len(tags) == len(self.tags):
            return
        reference = ee if self.ee_reference is None else self.ee_reference
        payload = {"version": 1, "frame": "robot_base", "ee_base": reference.tolist(),
                   "tags_base": {key: value.tolist() for key, value in tags.items()}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as stream:
                temporary = stream.name
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        self.ee_reference, self.tags = reference, tags

    def grasp_rotation(self, tag_id, current_rotation, *, cup_yaw_offset_deg=90.0, follow_tag_yaw=True):
        if self.ee_reference is None or str(tag_id) not in self.tags:
            raise ValueError(f"Tag {tag_id} 尚无旋转基准")
        if not np.isfinite(cup_yaw_offset_deg):
            raise ValueError("杯子旋转角必须为有限数")
        delta = rotation_matrix(current_rotation) @ self.tags[str(tag_id)].T
        yaw = float(np.arctan2(delta[1, 0], delta[0, 0])) if follow_tag_yaw else 0.0
        offset = np.deg2rad(cup_yaw_offset_deg) if tag_id in CUP_TAG_IDS else 0.0
        target = Rotation.from_euler("z", yaw + offset).as_matrix() @ self.ee_reference
        return target, yaw
