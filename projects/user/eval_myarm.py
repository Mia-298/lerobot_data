"""先注册本地环境，再调用官方评估入口；参数与 lerobot-eval 相同。"""
import myarm_eval_config  # noqa: F401

if __name__ == "__main__":
    from lerobot.scripts.lerobot_eval import main
    main()
