import argparse

from lerobot_robot_aubo import MyArm, MyArmConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--power-on",
        action="store_true",
        help="显式给机械臂上电并松刹车；默认不会上电",
    )
    parser.add_argument(
        "--send-current-position",
        action="store_true",
        help="显式向机械臂下发当前关节位置；默认测试不会发送运动命令",
    )
    args = parser.parse_args()

    robot = MyArm(MyArmConfig(id="aubo_arm"))
    print("相机数量：", len(robot.cameras))
    print("观测格式：", robot.observation_features)
    print("动作格式：", robot.action_features)

    try:
        robot.connect()
        print("AUBO 和相机连接成功")
        print("机械臂状态：", robot.robot_mode)


        robot.power_on_and_start()
        print("机械臂上电完成，当前状态：", robot.robot_mode)

        observation = robot.get_observation()
        assert set(observation) == set(robot.observation_features)

        for name,index in robot.config.joint_names:
            print(name, observation[f"{name}.pos"])
        reset_pose = [float(value) for value in robot.robot_interface.getRobotState().getJointPositions()]

        # robot.robot_interface.getMotionControl().setHomePosition(current_q)
        reset_pose
        for name in robot.camera_names:
            image = observation[name]
            print(name, image.shape, image.dtype)
            assert image.shape == robot.observation_features[name]
            assert image.dtype.name == "uint8"

        if args.send_current_position:
            action = {
                key: observation[key]
                for key in robot.action_features
            }
            sent_action = robot.send_action(action)
            print("已下发当前关节位置：", sent_action)
        else:
            print("只读测试完成，未向机械臂发送运动命令")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
