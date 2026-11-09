"""Generate the six AprilTag textures referenced by world.xml."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "textures"
OUT.mkdir(exist_ok=True)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
tags = {
    "tag_cube_small": 0,
    "tag_cube_large": 1,
    "tag_plate_small": 2,
    "tag_plate_large": 3,
    "tag_cup_small": 4,
    "tag_cup_large": 5,
}

for name, tag_id in tags.items():
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 192)
    canvas = np.full((256, 256), 255, dtype=np.uint8)
    canvas[32:224, 32:224] = marker
    path = OUT / f"{name}.png"
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write {path}")
    print(f"wrote {path}")
