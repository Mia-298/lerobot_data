import numpy as np

from camsdk_py import Camera


def main() -> None:
    camera = Camera()
    print("discovered:", camera.discover(timeout_ms=2000))

    try:
        info = camera.open("10.10.10.18", discover_timeout_ms=2000)
        print("opened:", info)

        frame = camera.capture()
        rgb = frame["rgb"]
        depth = frame["depth"]

        if rgb is not None:
            assert rgb.dtype == np.uint8
            assert rgb.ndim == 3 and rgb.shape[2] == 3
            print("rgb:", rgb.shape, rgb.dtype)
        if depth is not None:
            assert depth.dtype == np.float32
            assert depth.ndim == 2
            print("depth:", depth.shape, depth.dtype)
    finally:
        camera.close()


if __name__ == "__main__":
    main()

