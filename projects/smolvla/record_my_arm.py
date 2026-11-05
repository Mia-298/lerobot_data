from lerobot.configs import parser
from lerobot.scripts.lerobot_record import RecordConfig, record

# 导入这个包时，也会执行其中的配置类注册
from lerobot_robot_myarm import MyArmConfig, make_myarm_processor


@parser.wrap()
def main(cfg: RecordConfig):
    if not isinstance(cfg.robot, MyArmConfig):
        raise ValueError("这个启动脚本需要使用 --robot.type=my_arm")
    # processor是流水线，之后调用processor((act, obs))才会调用call，然后才调用ik_step.action(action),在
    processor = make_myarm_processor(cfg.robot)

    # 把自定义流水线交给官方录制函数
    record(
        cfg,
        teleop_action_processor=processor,
    )


if __name__ == "__main__":
    main()