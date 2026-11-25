"""先注册本地环境，再调用官方训练入口；参数与 lerobot-train 相同。"""
import myarm_eval_config  # noqa: F401

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main
    main()
