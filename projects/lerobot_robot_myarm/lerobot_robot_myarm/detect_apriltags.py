"""Detect all AprilTags in a MuJoCo renderer image.

``MyArm._renderer.render().copy()`` returns an RGB ``uint8`` array with
shape ``(height, width, 3)``.  The public function in this file accepts that
array directly and uses the AprilTag 36h11 family, which is also used by the
textures in ``projects/models/textures``.

Example
-------
    from detect_apriltags import detect_apriltags

    image = robot._renderer.render().copy()
    detections = detect_apriltags(image)
    for detection in detections:
        print(detection.tag_id, detection.center, detection.corners)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11

# world.xml 中六个 Tag 的实际边长，单位米；用于把 0.5 cm 换算成像素。
TAG_SIZE_M = {
    0: 0.030,  # cube_small
    1: 0.050,  # cube_large
    2: 0.035,  # plate_small
    3: 0.045,  # plate_large
    4: 0.030,  # cup_small
    5: 0.040,  # cup_large
}


@dataclass(frozen=True)
class AprilTagDetection:
    """One detected AprilTag in pixel coordinates."""

    tag_id: int
    corners: np.ndarray  # shape (4, 2), clockwise from detector output
    center: np.ndarray  # shape (2,), x/y pixel coordinates
    side_lengths: np.ndarray  # four edge lengths in pixels


def _make_detector() -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
    parameters = cv2.aruco.DetectorParameters()

    # The rendered tags are high-contrast, but these settings make the
    # detector less sensitive to small anti-aliasing and lighting changes.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.01

    return cv2.aruco.ArucoDetector(dictionary, parameters)


_DETECTOR = _make_detector()


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image 必须是 numpy.ndarray，收到 {type(image)!r}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image 必须是 H×W×3，收到 shape={image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image 必须是 uint8，收到 dtype={image.dtype}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image 不能为空")
    return np.ascontiguousarray(image)


def detect_apriltags(image: np.ndarray) -> list[AprilTagDetection]:
    """Detect every AprilTag in an RGB MuJoCo renderer image.

    Parameters
    ----------
    image:
        The RGB ``uint8`` image returned by
        ``MyArm._renderer.render().copy()`` or ``MyArm.get_images()``.

    Returns
    -------
    list[AprilTagDetection]
        Detections sorted by ``tag_id``.  An empty list means no tag was
        detected; this is not treated as an error.
    """

    image = _validate_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners, ids, _rejected = _DETECTOR.detectMarkers(gray)

    if ids is None:
        return []

    detections: list[AprilTagDetection] = []
    for marker_corners, marker_id in zip(corners, ids.reshape(-1), strict=True):
        points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        side_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        detections.append(
            AprilTagDetection(
                tag_id=int(marker_id),
                corners=points,
                center=points.mean(axis=0),
                side_lengths=side_lengths,
            )
        )

    detections.sort(key=lambda item: item.tag_id)
    return detections


def draw_detections(
    image: np.ndarray,
    detections: list[AprilTagDetection] | None = None,
) -> np.ndarray:
    """Return an RGB copy annotated with tag IDs and corner polygons."""

    image = _validate_image(image)
    if detections is None:
        detections = detect_apriltags(image)

    annotated = image.copy()
    for detection in detections:
        points = np.round(detection.corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        center = tuple(np.round(detection.center).astype(int))
        cv2.circle(annotated, center, 4, (255, 0, 0), -1)
        cv2.putText(
            annotated,
            f"id={detection.tag_id}",
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def mask_apriltags(
    image: np.ndarray,
    detections: list[AprilTagDetection] | None = None,
    *,
    margin_m: float = 0.005,
    tag_sizes_m: dict[int, float] | None = None,
) -> np.ndarray:
    """Replace each Tag region with the median color just outside it.

    ``margin_m`` is a physical margin around the Tag used to sample the
    replacement color.  It is converted to pixels from the detected Tag edge
    length and its known physical edge length.  The Tag itself is filled; the
    outside 0.5 cm ring is only used as the color source.

    Parameters
    ----------
    image:
        RGB ``uint8`` image from MuJoCo.
    detections:
        Results from :func:`detect_apriltags`; detected again when omitted.
    margin_m:
        Sampling-ring width in meters.  The default is 0.005 m = 0.5 cm.
    tag_sizes_m:
        Optional mapping ``tag_id -> physical Tag edge length`` in meters.
        Defaults to the sizes used by ``world.xml``.
    """

    image = _validate_image(image)
    if margin_m <= 0:
        raise ValueError("margin_m 必须大于 0")
    if detections is None:
        detections = detect_apriltags(image)
    if not detections:
        return image.copy()

    sizes = TAG_SIZE_M if tag_sizes_m is None else tag_sizes_m
    output = image.copy()
    height, width = image.shape[:2]

    for detection in detections:
        tag_size_m = sizes.get(detection.tag_id)
        if tag_size_m is None or tag_size_m <= 0:
            raise ValueError(
                f"没有 tag_id={detection.tag_id} 的物理尺寸，请通过 tag_sizes_m 提供"
            )

        inner = np.round(detection.corners).astype(np.int32)
        mean_side_px = float(np.mean(detection.side_lengths))
        margin_px = mean_side_px * margin_m / tag_size_m
        scale = 1.0 + 2.0 * margin_px / max(mean_side_px, 1e-6)

        center = detection.center.astype(np.float32)
        outer = np.round(center + (detection.corners - center) * scale).astype(np.int32)
        inner = np.clip(inner, [0, 0], [width - 1, height - 1])
        outer = np.clip(outer, [0, 0], [width - 1, height - 1])

        inner_mask = np.zeros((height, width), dtype=np.uint8)
        outer_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(inner_mask, inner, 255)
        cv2.fillConvexPoly(outer_mask, outer, 255)

        # 只使用 Tag 外扩 0.5 cm 环带采样，避免把 Tag 黑白像素混入颜色。
        ring_mask = cv2.subtract(outer_mask, inner_mask)
        ring_pixels = output[ring_mask > 0]
        if ring_pixels.size == 0:
            continue

        fill_color = np.median(ring_pixels, axis=0).astype(np.uint8)
        output[inner_mask > 0] = fill_color

    return output


def _run_robot_demo(
    xml_path: Path,
    camera_name: str,
    output_path: Path | None,
    masked_output_path: Path | None,
) -> None:
    """Render one MyArm frame and detect tags; useful for a quick smoke test."""

    from lerobot_robot_myarm import MyArm, MyArmConfig

    config = MyArmConfig(xml_path=str(xml_path), camera_names=[camera_name])
    robot = MyArm(config)
    try:
        robot.connect()
        image = robot.get_images(camera_name)
        detections = detect_apriltags(image)
        print(f"image shape={image.shape}, dtype={image.dtype}")
        print(f"detected {len(detections)} AprilTag(s)")
        for detection in detections:
            print(
                f"id={detection.tag_id} "
                f"center=({detection.center[0]:.1f}, {detection.center[1]:.1f}) "
                f"corners={np.round(detection.corners, 1).tolist()}"
            )

        if output_path is not None:
            annotated = draw_detections(image, detections)
            # cv2.imwrite expects BGR, while MuJoCo returns RGB.
            ok = cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError(f"无法写入标注图像：{output_path}")
            print(f"saved {output_path}")

        if masked_output_path is not None:
            masked = mask_apriltags(image, detections)
            ok = cv2.imwrite(
                str(masked_output_path),
                cv2.cvtColor(masked, cv2.COLOR_RGB2BGR),
            )
            if not ok:
                raise RuntimeError(f"无法写入去 Tag 图像：{masked_output_path}")
            print(f"saved {masked_output_path}")
    finally:
        robot.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect all AprilTags in a MyArm MuJoCo frame")
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "models" / "world.xml",
        help="MuJoCo XML path; defaults to the complete world.xml scene",
    )
    parser.add_argument("--camera", default="front", choices=["front", "wrist"])
    parser.add_argument("--output", type=Path, help="optional annotated RGB frame output path")
    parser.add_argument(
        "--masked-output",
        type=Path,
        help="optional output path with Tag regions replaced by surrounding color",
    )
    args = parser.parse_args()
    _run_robot_demo(args.xml, args.camera, args.output, args.masked_output)


if __name__ == "__main__":
    main()
