import pyaubo_sdk
from camsdk_py import Camera
ee_name: str = "attachment_site"
robot_ip = "192.168.193.130"  # 服务器 IP 地址
robot_port = 30004  # 端口号

robot_speed = 0.75
robot_name = "rob1"
robot_password = "aubo123"


robot_rpc_client = pyaubo_sdk.RpcClient()
robot_rpc_client.setRequestTimeout(1000)  # 接口调用: 设置 RPC 超时
robot_rpc_client.connect(robot_ip, robot_port)  # 接口调用: 连接 RPC 服务
if robot_rpc_client.hasConnected():
    print("RPC客户端连接成功!")
    robot_rpc_client.login(robot_name, robot_password)  # 接口调用: 登录
    robot_name = robot_rpc_client.getRobotNames()[0]

    robot_interface = robot_rpc_client.getRobotInterface(robot_name)
    robot_interface.getMotionControl().setSpeedFraction(robot_speed)
    _is_connected = True
    q_current = (
            robot_interface
            .getRobotState()
            .getJointPositions()
        )
    print(f"当前关节位置: {q_current}")
    camera = Camera()

   

    # 打开指定相机
    try:
        info = camera.open(
            "10.10.10.18",
            discover_timeout_ms=2000,
        )
    except Exception as e:
        print(f"相机打开失败：{e}")
        raise

    # 采集一帧
    # frame = camera.capture()

    # rgb = frame["rgb"]
    # depth = frame["depth"]

    # print(rgb.shape, rgb.dtype)
    # print(depth.shape, depth.dtype)

    camera.close()
    print("相机已关闭")
else:
    print("RPC客户端连接失败!")