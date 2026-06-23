#!/usr/bin/env python3
"""Lightweight book-spine YOLO segmentation inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


APP_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SpineCrop:
    index: int
    confidence: float
    area: int
    bbox_xyxy: tuple[int, int, int, int]
    polygon_xy: tuple[tuple[float, float], ...]
    masked_path: Path
    rectified_path: Path


def default_weights_path() -> Path:
    return APP_DIR / "weights" / "best.pt"


def iter_images(source: Path) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() in IMAGE_SUFFIXES:
            yield source
        return

    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def safe_name(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in path.stem)


def polygon_to_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.round(polygon).astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def expand_box(
    box_xyxy: np.ndarray,
    width: int,
    height: int,
    pad: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = np.round(box_xyxy).astype(int).tolist()
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def transparent_masked_crop(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    alpha = mask[y1:y2, x1:x2]
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba


def rectify_from_masked_crop(masked_bgra: np.ndarray) -> np.ndarray:
    alpha = masked_bgra[:, :, 3]
    points = cv2.findNonZero(alpha)
    if points is None or len(points) < 4:
        return masked_bgra

    rect = cv2.minAreaRect(points)
    angle = rect[2]
    width, height = rect[1]
    if width <= 1 or height <= 1:
        return masked_bgra
    if width < height:
        angle += 90

    h, w = masked_bgra.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        masked_bgra,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    rotated_alpha = rotated[:, :, 3]
    points = cv2.findNonZero(rotated_alpha)
    if points is None:
        return rotated

    x, y, bw, bh = cv2.boundingRect(points)
    rectified = rotated[y : y + bh, x : x + bw]
    if rectified.shape[1] > rectified.shape[0]:
        rectified = cv2.rotate(rectified, cv2.ROTATE_90_CLOCKWISE)
    return rectified


def draw_overlay(image: np.ndarray, polygons: list[np.ndarray], crops: list[SpineCrop]) -> np.ndarray:
    overlay = image.copy()
    alpha_layer = image.copy()

    for polygon, crop in zip(polygons, crops, strict=False):
        points = polygon.astype(np.int32)
        color = (
            int(40 + (crop.index * 47) % 180),
            int(210 - (crop.index * 31) % 130),
            int(70 + (crop.index * 73) % 180),
        )
        cv2.fillPoly(alpha_layer, [points], color)
        cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=2)
        x1, y1, _, _ = crop.bbox_xyxy
        cv2.putText(
            overlay,
            f"{crop.index}:{crop.confidence:.2f}",
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return cv2.addWeighted(alpha_layer, 0.32, overlay, 0.68, 0)


def segment_image(
    model,
    image_path: Path,
    output_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    pad: int,
    retina_masks: bool,
    max_det: int,
    *,
    device: str | None = None,
    save_crops: bool = True,
) -> tuple[Path, list[SpineCrop]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]
    predict_kwargs = {
        "source": str(image_path),
        "task": "segment",
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "retina_masks": retina_masks,
        "max_det": max_det,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device
    result = model.predict(**predict_kwargs)[0]

    stem = safe_name(image_path)
    image_out_dir = output_dir / stem
    image_out_dir.mkdir(parents=True, exist_ok=True)
    masked_dir = image_out_dir / "masked"
    rectified_dir = image_out_dir / "rectified"
    if save_crops:
        masked_dir.mkdir(parents=True, exist_ok=True)
        rectified_dir.mkdir(parents=True, exist_ok=True)

    crops: list[SpineCrop] = []
    kept_polygons: list[np.ndarray] = []
    if result.masks is not None and result.boxes is not None:
        polygons = result.masks.xy
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        order = sorted(range(len(polygons)), key=lambda idx: (int(boxes[idx][0]), int(boxes[idx][1])))

        for out_index, pred_index in enumerate(order, start=1):
            polygon = np.asarray(polygons[pred_index], dtype=np.float32)
            if polygon.size < 6:
                continue

            bbox = expand_box(boxes[pred_index], width, height, pad)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            if save_crops:
                mask = polygon_to_mask((height, width), polygon)
                area = int(cv2.countNonZero(mask))
            else:
                area = int(abs(cv2.contourArea(polygon)))
            if area <= 0:
                continue

            if save_crops:
                masked = transparent_masked_crop(image, mask, bbox)
                rectified = rectify_from_masked_crop(masked)
                masked_path = masked_dir / f"{stem}_spine_{out_index:03d}_masked.png"
                rectified_path = rectified_dir / f"{stem}_spine_{out_index:03d}_rectified.png"
                cv2.imwrite(str(masked_path), masked)
                cv2.imwrite(str(rectified_path), rectified)
            else:
                masked_path = Path()
                rectified_path = Path()

            crops.append(
                SpineCrop(
                    index=out_index,
                    confidence=float(confidences[pred_index]),
                    area=area,
                    bbox_xyxy=bbox,
                    polygon_xy=tuple((float(x), float(y)) for x, y in polygon.reshape(-1, 2)),
                    masked_path=masked_path,
                    rectified_path=rectified_path,
                )
            )
            kept_polygons.append(polygon)

    overlay = draw_overlay(image, kept_polygons, crops) if crops else image
    overlay_path = image_out_dir / f"{stem}_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    return overlay_path, crops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run book-spine segmentation inference.")
    parser.add_argument("--source", required=True, type=Path, help="Input image file or directory.")
    parser.add_argument("--weights", default=default_weights_path(), type=Path, help="YOLO segmentation weights.")
    parser.add_argument("--output", default=APP_DIR / "runs", type=Path, help="Output directory.")
    parser.add_argument("--imgsz", default=1024, type=int, help="Inference image size.")
    parser.add_argument("--conf", default=0.60, type=float, help="Confidence threshold.")
    parser.add_argument("--iou", default=0.6, type=float, help="NMS IoU threshold.")
    parser.add_argument("--pad", default=8, type=int, help="Crop padding in pixels.")
    parser.add_argument("--max-det", default=30, type=int, help="Maximum detections per image.")
    parser.add_argument("--device", default=None, help="Inference device, for example cpu, cuda:0.")
    parser.add_argument("--no-retina-masks", action="store_true", help="Disable high-resolution masks.")
    parser.add_argument("--no-save-crops", action="store_true", help="Only save the overlay preview image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(f"Weights file not found: {args.weights}")

    images = list(iter_images(args.source))
    if not images:
        raise SystemExit(f"No supported images found in {args.source}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    args.output.mkdir(parents=True, exist_ok=True)

    total = 0
    for image_path in images:
        overlay_path, crops = segment_image(
            model=model,
            image_path=image_path,
            output_dir=args.output,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            pad=args.pad,
            retina_masks=not args.no_retina_masks,
            max_det=args.max_det,
            device=args.device,
            save_crops=not args.no_save_crops,
        )
        total += len(crops)
        print(f"{image_path}: {len(crops)} spines -> {overlay_path}")

    print(f"Done. Segmented {total} book spines from {len(images)} image(s).")


if __name__ == "__main__":
    main()
