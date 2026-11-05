import sys; import pyaubo_sdk
print("SDK 中包含的公开名称：")
for name in dir(pyaubo_sdk):
    if not name.startswith("_"):
        print(name)
help(pyaubo_sdk.robot_interface.getMotionControl() \
        .moveJoint)