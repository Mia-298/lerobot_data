import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

model_dir = "/home/mia/data/smolvla_work/models/smolvla_base"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 加载配置和已经下载好的处理器
config = SmolVLAConfig.from_pretrained(model_dir)
config.device = device
# preprocess用于处理输入数据，列表里有很多字典
# postprocesspostprocess用于输出结果
preprocess, postprocess = make_pre_post_processors(
    config,
    model_dir,
    preprocessor_overrides={
        "device_processor": {"device": device}
    },
)
# 2. 构造一帧测试输入
frame = {
    "observation.state": torch.zeros(6, dtype=torch.float32),
    "observation.images.camera1": torch.zeros(3, 256, 256),
    "observation.images.camera2": torch.zeros(3, 256, 256),
    "observation.images.camera3": torch.zeros(3, 256, 256),
    "task": "Pick up the cube.",
}

# 3. 测试预处理
batch = preprocess(frame)

# 4. 用一个假动作测试后处理
# 注意：这个动作不是模型预测的，也不会发送给机器人。
dummy_action = torch.zeros(1, 6, device=device)
processed_action = postprocess(dummy_action)