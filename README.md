# LeRobot 自定义机器人与数据采集项目

## 目录说明

- projects/lerobot_robot_myarm：仿真机器人插件。
- projects/lerobot_robot_aubo：AUBO 机器人插件。
- projects/mia_kinematics：运动学 Python 包。
- projects/models：仿真 XML 和网格资源。
- projects/smolvla：采集、测试、模型相关程序。
- environment.yml：Conda 环境依赖清单。
- lerobot-revision.txt：官方 LeRobot 提交版本。
- smolvla-revision.txt：SmolVLA 模型版本。
- scripts/bootstrap.sh：恢复环境及源码。
- scripts/download_models.sh：下载模型权重。

官方 LeRobot 源码和模型权重不提交到本仓库。

## 新电脑恢复

前提：
- 安装 Git、Miniconda，并确保终端可以使用 conda。
- 配置 GitHub SSH 访问。
- 优先使用与原电脑相近的 Ubuntu x86_64 环境。
- 使用 GPU 时，单独准备兼容的 NVIDIA 驱动。

```bash
mkdir -p "$HOME/data"

git clone git@github.com:Mia-298/lerobot_data.git \
  "$HOME/data/smolvla_work"

bash "$HOME/data/smolvla_work/scripts/bootstrap.sh"

bash "$HOME/data/smolvla_work/scripts/download_models.sh"

conda activate lerobot
```

模型下载脚本默认使用 `https://hf-mirror.com`，也可以临时指定其他端点：

```bash
HF_ENDPOINT=https://huggingface.co \
  bash "$HOME/data/smolvla_work/scripts/download_models.sh"
```

恢复脚本不会自动连接或控制机器人。

## 日常同步

开始工作前：

```bash
git -C "$HOME/data/smolvla_work" status
git -C "$HOME/data/smolvla_work" pull --ff-only
```

修改后，检查差异，再提交并推送。

## 已存在环境的依赖更新

bootstrap.sh 会保留已有 lerobot 环境，不会自动更新其中的依赖。
当 environment.yml 变化时，执行：

```bash
conda env update -n lerobot \
  -f "$HOME/data/smolvla_work/environment.yml"

bash "$HOME/data/smolvla_work/scripts/bootstrap.sh"
```

## 不在 Git 同步范围内的内容

模型权重、数据集、训练输出、个人环境变量及机器相关配置需要单独管理。
