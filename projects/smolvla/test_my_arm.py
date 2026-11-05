
import time
from lerobot_robot_myarm import MyArm, MyArmConfig,MyArmKeyboardConfig,MyArmKeyboard,KeyboardIKActionStep
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
robot = MyArm(MyArmConfig(id="mujoco_arm"))

# 录制器可能在 connect() 之前读取这些声明。
print("相机数量：", len(robot.cameras))
print("观测格式：", robot.observation_features)
print("动作格式：", robot.action_features)
config = MyArmKeyboardConfig(id="keyboard")
teleop = MyArmKeyboard(config)
try:
    robot.connect()

    obs = robot.get_observation()

    assert set(obs) == set(robot.observation_features)

    for name in robot.config.joint_names:
        print(name, obs[f"{name}.pos"])

    print("夹爪实际开合量：", obs["gripper.pos"])

    for name in robot.camera_names:
        image = obs[name]
        print(name, image.shape, image.dtype)

        assert image.shape == robot.observation_features[name]
        assert str(image.dtype) == "uint8"

    # 用当前实际位置作为目标，先测试保持姿态。
    action = {
        key: obs[key]
        for key in robot.action_features
    }

    sent_action = robot.send_action(action)
    print("下发动作：", sent_action)
    teleop.connect()
    time.sleep(0.1)

    print("键盘是否连接：", teleop.is_connected)

    if not teleop.is_connected:
        raise RuntimeError("键盘监听没有启动，请检查终端中的提示")

    print("接下来的 10 秒内，尝试按住、松开 w、a、s、d")
    previous_action = None
    start = time.monotonic()
    teleop_action_processor = RobotProcessorPipeline(
                steps=[
                    KeyboardIKActionStep(
                        joint_names=robot.config.joint_names,
                        transforms_relative=robot.transforms_relative,
                        axes_relative=robot.axes_relative,
                        T6ee=robot.T6ee,
                        q_min=robot.ctrl_min,       # 补上
                        q_max=robot.ctrl_max,       # 补上
                    )
                ],
                to_transition=robot_action_observation_to_transition,
                to_output=transition_to_robot_action,
            )

    while teleop.is_connected and time.monotonic() - start < 10:
        action = teleop.get_action()

        # 只在按键状态变化时打印，避免刷屏
        if action != previous_action:
            print("当前按键：", action)
            previous_action = action
        obs = robot.get_observation()

        # 这一行才是真正执行流水线
        processed_action = teleop_action_processor((action, obs))
        sent_action = robot.send_action(processed_action)

        time.sleep(0.05)
    robot.reset_simulation()
    obs = robot.get_observation()
    print(obs["shoulder_pan_joint.pos"])

except KeyboardInterrupt:
    print("测试被中断")
finally:
    robot.disconnect()
    if teleop.is_connected:
        teleop.disconnect()