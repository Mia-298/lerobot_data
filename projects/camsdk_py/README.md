# camsdk-py

Python bindings for `/opt/CamSDK/CamSDK_CPP`.

Build and install into the LeRobot environment:

```bash
conda activate lerobot
cd /home/mia/data/smolvla_work/projects/camsdk_py
python -m pip install -e . --no-build-isolation
```

Test camera discovery and capture:

```bash
python tests/test_camera.py
```

The camera announces itself through UDP port 50001. WSL must be allowed to
receive that port through the Windows Hyper-V firewall.

