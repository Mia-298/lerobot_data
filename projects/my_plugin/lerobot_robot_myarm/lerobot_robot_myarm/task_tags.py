"""当前 world.xml 的任务到 AprilTag 映射；颜色变化后需同步更新。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskTags:
    object_tag_id: int
    plate_tag_id: int


# 同一物体可用大小或颜色称呼；顺序决定默认菜单的 1～16 编号。
# 与 generate_apriltag_textures.py 的 ID、world.xml 的颜色对应。
OBJECT_TAGS = {
    "large cup": 5,
    "green cup": 5,
    "small cup": 4,
    "yellow cup": 4,
    "large block": 1,
    "blue block": 1,
    "small block": 0,
    "red block": 0,
}
PLATE_TAGS = {"large": 3, "small": 2}

TASK_TAGS_BY_TEXT = {
    f"Pick up the {obj} and place it on the {size} plate.": TaskTags(tag_id, plate_id)
    for obj, tag_id in OBJECT_TAGS.items()
    for size, plate_id in PLATE_TAGS.items()
}


def default_tasklist() -> list[str]:
    """4 个物体 × 2 种称呼 × 2 个盘子，共 16 条语言指令。"""
    return list(TASK_TAGS_BY_TEXT)


def resolve_task_tags(task_id: int, tasklist: list[str]) -> TaskTags:
    """菜单编号从 1 开始；按文本查表，避免重排菜单后抓错物体。"""
    if type(task_id) is not int or not 1 <= task_id <= len(tasklist):
        raise ValueError(f"无效任务编号：{task_id}，应为 1～{len(tasklist)}")
    text = tasklist[task_id - 1]
    if text not in TASK_TAGS_BY_TEXT:
        raise ValueError(
            f"任务 {task_id} 没有配置 AprilTag 映射：{text}。"
            "请使用任务表中的指令，或在 task_tags.py 中补充对应映射。"
        )
    return TASK_TAGS_BY_TEXT[text]
