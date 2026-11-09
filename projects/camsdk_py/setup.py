import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


SDK_ROOT = Path(os.environ.get("CAMSDK_ROOT", "/opt/CamSDK/CamSDK_CPP"))
SDK_INCLUDE = SDK_ROOT / "include"
SDK_LIBRARY = SDK_ROOT / "lib" / "linux_x86_64"

if not (SDK_INCLUDE / "Camera.h").is_file():
    raise RuntimeError(f"Camera.h not found under CAMSDK_ROOT={SDK_ROOT}")
if not (SDK_LIBRARY / "libCameraPro.so").is_file():
    raise RuntimeError(f"libCameraPro.so not found under CAMSDK_ROOT={SDK_ROOT}")


setup(
    packages=["camsdk_py"],
    ext_modules=[
        Pybind11Extension(
            "camsdk_py._camsdk",
            ["src/bindings.cpp"],
            include_dirs=[str(SDK_INCLUDE)],
            library_dirs=[str(SDK_LIBRARY)],
            libraries=["CameraPro"],
            runtime_library_dirs=[str(SDK_LIBRARY)],
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

