# MyArm 自动抓取：杯子姿态与 AprilTag 旋转基准

## 行为与坐标系

自动遥操作类型为 `my_arm_auto` 时，杯子（Tag **4、5**）默认在物体原有朝向上，沿 **robot base 的 Z 轴正向旋转 90°**；其他物体没有这项额外偏置。目标姿态使用左乘，不是绕工具自身 Z 轴旋转，也不是简单修改某个关节角：

```python
R_delta = R_base_tag_current @ R_base_tag_reference.T
yaw = atan2(R_delta[1, 0], R_delta[0, 0])
R_base_ee_goal = Rz(yaw + cup_offset) @ R_base_ee_reference
```

例如：物体相对基准为 0° 时，杯子目标偏转 90°；将杯子与其 Tag 一起绕 base Z 转 30° 后，目标为 120°。当前只跟随物体 **yaw**，不跟随 roll/pitch。

机械臂先到达抓取点正上方，再分步转腕；实测末端姿态满足误差阈值后才允许下降。即使本周期 XYZ 位移全为零，姿态目标也会进入自定义 IK，转换为关节动作。转动时补偿标定的 TCP 偏移，不将 EE 原点误当作 TCP。

这不是完整碰撞规划：桌面限高和离散关节插值检查并不能证明避开杯把、其他物体或所有连杆碰撞。

## 配置

配置定义在本包的 `config_myarm_auto.py`：

```python
cup_yaw_offset_deg = 90.0          # 0 取消杯子额外偏转；-90 改为反向
follow_tag_yaw = True             # False 时忽略相对 Tag yaw，但保留杯子偏置
rotation_step_rad = 0.02          # 每次处理动作的最大姿态步长，不是 rad/s
orientation_tolerance_rad = 0.03  # 实测末端姿态误差阈值，约 1.72°
```

录制入口必须把自动配置交给自定义 processor；项目的 `record_my_arm.py` 已接入：

```python
processor = make_myarm_processor(
    cfg.robot,
    auto_config=cfg.teleop if cfg.teleop.type == "my_arm_auto" else None,
)
_lerobot_record.record(cfg, teleop_action_processor=processor)
```

不需要修改官方 LeRobot 的录制程序。

## 保存零旋转基准

默认文件为项目中的 `projects/models/apriltag_rotation_reference.json`，可通过 `apriltag_rotation_reference_path` 指定其他路径。

- 文件保存完整的 base 坐标系下 EE 与每个 Tag 的 3×3 旋转矩阵，而不是强行把实测矩阵设为单位阵。
- 已有基准不会每个 episode 重写；使用绝对目标，因此不会出现第一个 episode 90°、第二个 180° 的累加。
- 没有基准文件时，首次成功检测目标会保存当时的 EE 姿态和可见 Tag 姿态；缺失 Tag 在以后首次可见时补齐，各自定义为零。
- **新增或重建基准时，必须先将物体、Tag 和末端置于期望的基准姿态，并关闭物体旋转随机化。** 等所有需要的 Tag 都有基准后，再开启物体旋转随机化。
- 需要重新标定时，先备份旧文件，再配置一个新的基准路径。不要在物体已经旋转的场景中直接重置基准。
- 基准文件格式或旋转矩阵无效时会报错停止，而不是静默使用错误朝向。

目标检测日志输出相对 yaw、基准文件路径、当前/基准 Tag 矩阵以及目标 EE 矩阵，便于确认配置生效。

## 离线验证

在项目根目录运行（使用安装了项目依赖的 Python 环境）：

```bash
python -m unittest discover \
  -s projects/my_plugin/lerobot_robot_myarm/tests \
  -p 'test_grasp_orientation.py' -v
```

新增 8 项测试覆盖 base-Z 左乘、杯子偏置叠加、基准持久化与跨 episode 不累加、旋转步长、零 XYZ 经过 IK、TCP 偏移补偿，以及实测姿态到位前禁止下降。其中包含实际模型/XML 与 FK/IK 的合成反馈验证，不连接机器人、不运行录制或实时仿真循环。

当前完整插件回归为 43 项中 39 项通过；原有抓取安全测试仍有 4 项失败，未由本次姿态改动消除。尚未验证实时仿真中的杯把避碰效果，也不应据离线测试宣称实机安全。
