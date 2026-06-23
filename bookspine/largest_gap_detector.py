#!/usr/bin/env python3
"""Detect the largest bookshelf gap without a template image.

The method is designed for reliability over speed:
1. Find near-vertical book/shelf boundaries with Canny + Hough.
2. Cluster those boundaries by x position.
3. Score adjacent boundary pairs as possible gaps.
4. Pick the widest low-texture internal gap and draw its centerline.

The returned centerline is not forced to be perfectly vertical. Its endpoints
are the midpoint between the two selected boundary segments at the top/bottom
of their common vertical span.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Boundary:
    x_top: float
    x_bottom: float
    y_top: float
    y_bottom: float
    support: int
    strength: float

    def x_at(self, y: float) -> float:
        if abs(self.y_bottom - self.y_top) < 1e-6:
            return 0.5 * (self.x_top + self.x_bottom)
        t = (y - self.y_top) / (self.y_bottom - self.y_top)
        return self.x_top + t * (self.x_bottom - self.x_top)


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def default_roi(img: np.ndarray, margin_x=0.06, top=0.10, bottom=0.94):
    h, w = img.shape[:2]
    return (
        int(w * margin_x),
        int(h * top),
        int(w * (1.0 - margin_x)),
        int(h * bottom),
    )


def cluster_lines(lines: np.ndarray, y_offset: int, x_offset: int) -> list[Boundary]:
    raw = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [float(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 80:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle < 78 or abs(dx) > 35:
            continue
        if y1 <= y2:
            xt, yt, xb, yb = x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset
        else:
            xt, yt, xb, yb = x2 + x_offset, y2 + y_offset, x1 + x_offset, y1 + y_offset
        raw.append((0.5 * (xt + xb), xt, yt, xb, yb, length))

    raw.sort(key=lambda v: v[0])
    clusters: list[list[tuple[float, float, float, float, float, float]]] = []
    for item in raw:
        if not clusters or item[0] - clusters[-1][-1][0] > 10:
            clusters.append([item])
        else:
            clusters[-1].append(item)

    boundaries: list[Boundary] = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        weights = np.array([c[5] for c in cluster], dtype=np.float32)
        weights /= max(float(weights.sum()), 1e-6)
        x_top = float(sum(w * c[1] for w, c in zip(weights, cluster)))
        y_top = float(min(c[2] for c in cluster))
        x_bottom = float(sum(w * c[3] for w, c in zip(weights, cluster)))
        y_bottom = float(max(c[4] for c in cluster))
        boundaries.append(Boundary(x_top, x_bottom, y_top, y_bottom, len(cluster), float(sum(c[5] for c in cluster))))

    boundaries.sort(key=lambda b: 0.5 * (b.x_top + b.x_bottom))
    return boundaries


def find_boundaries(img: np.ndarray, roi_box) -> list[Boundary]:
    x0, y0, x1, y1 = roi_box
    roi = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 110)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=42, minLineLength=95, maxLineGap=16)
    if lines is None:
        return []
    return cluster_lines(lines, y0, x0)


def edge_density(gray: np.ndarray, x_left: int, x_right: int, y_top: int, y_bottom: int) -> float:
    x_left = max(0, min(gray.shape[1] - 1, x_left))
    x_right = max(0, min(gray.shape[1], x_right))
    y_top = max(0, min(gray.shape[0] - 1, y_top))
    y_bottom = max(0, min(gray.shape[0], y_bottom))
    if x_right <= x_left + 1 or y_bottom <= y_top + 1:
        return 1.0
    patch = gray[y_top:y_bottom, x_left:x_right]
    edges = cv2.Canny(patch, 35, 110)
    return float(np.count_nonzero(edges) / edges.size)


def color_texture(img: np.ndarray, x_left: int, x_right: int, y_top: int, y_bottom: int) -> tuple[float, float]:
    x_left = max(0, min(img.shape[1] - 1, x_left))
    x_right = max(0, min(img.shape[1], x_right))
    y_top = max(0, min(img.shape[0] - 1, y_top))
    y_bottom = max(0, min(img.shape[0], y_bottom))
    if x_right <= x_left + 1 or y_bottom <= y_top + 1:
        return 255.0, 1.0
    patch = img[y_top:y_bottom, x_left:x_right]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    sat = float(np.median(hsv[:, :, 1]))
    texture = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))) / 255.0)
    return sat, texture


def choose_largest_gap(img: np.ndarray, boundaries: list[Boundary], roi_box):
    if len(boundaries) < 2:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    candidates = []
    x0, y0, x1, y1 = roi_box

    for left, right in zip(boundaries, boundaries[1:]):
        y_top = max(left.y_top, right.y_top, y0)
        y_bottom = min(left.y_bottom, right.y_bottom, y1)
        overlap = y_bottom - y_top
        if overlap < 120:
            continue

        if left.support < 2 or right.support < 2:
            continue

        xl_top = left.x_at(y_top)
        xr_top = right.x_at(y_top)
        xl_bottom = left.x_at(y_bottom)
        xr_bottom = right.x_at(y_bottom)
        width_top = xr_top - xl_top
        width_bottom = xr_bottom - xl_bottom
        width = 0.5 * (width_top + width_bottom)

        # A gripper-entry gap should be wider than a hairline seam but much
        # narrower than a whole book spine.
        if width < 10 or width > 70:
            continue

        x_min = int(max(min(xl_top, xl_bottom) + 2, x0))
        x_max = int(min(max(xr_top, xr_bottom) - 2, x1))
        density = edge_density(gray, x_min, x_max, int(y_top), int(y_bottom))
        sat, texture = color_texture(img, x_min, x_max, int(y_top), int(y_bottom))

        if density > 0.22 or texture > 0.42:
            continue

        # Prefer wide gaps with low internal texture and solid boundary support.
        support_bonus = np.log1p(left.support + right.support)
        color_penalty = 1.0 - min(max(sat - 40.0, 0.0) / 220.0, 0.45)
        score = width * (1.0 - min(density * 3.2, 0.8)) * color_penalty * (1.0 + 0.12 * support_bonus)
        candidates.append(
            {
                "score": float(score),
                "width": float(width),
                "edge_density": density,
                "median_saturation": sat,
                "texture": texture,
                "left": left,
                "right": right,
                "y_top": float(y_top),
                "y_bottom": float(y_bottom),
                "center_top": [float(0.5 * (xl_top + xr_top)), float(y_top)],
                "center_bottom": [float(0.5 * (xl_bottom + xr_bottom)), float(y_bottom)],
            }
        )

    if not candidates:
        return None
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[0], candidates


def draw_result(img: np.ndarray, roi_box, boundaries: list[Boundary], chosen, all_candidates) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = roi_box
    cv2.rectangle(out, (x0, y0), (x1, y1), (80, 255, 80), 1)

    for b in boundaries:
        cv2.line(out, (int(b.x_top), int(b.y_top)), (int(b.x_bottom), int(b.y_bottom)), (180, 180, 180), 1, cv2.LINE_AA)

    if chosen:
        c = chosen[0]
        left: Boundary = c["left"]
        right: Boundary = c["right"]
        y_top = c["y_top"]
        y_bottom = c["y_bottom"]
        lt = (int(left.x_at(y_top)), int(y_top))
        lb = (int(left.x_at(y_bottom)), int(y_bottom))
        rt = (int(right.x_at(y_top)), int(y_top))
        rb = (int(right.x_at(y_bottom)), int(y_bottom))
        ct = tuple(int(v) for v in c["center_top"])
        cb = tuple(int(v) for v in c["center_bottom"])

        cv2.line(out, lt, lb, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.line(out, rt, rb, (0, 165, 255), 3, cv2.LINE_AA)
        cv2.line(out, ct, cb, (0, 0, 255), 3, cv2.LINE_AA)
        label = f"max gap {c['width']:.1f}px"
        cv2.putText(out, label, (max(5, ct[0] - 55), max(22, ct[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return out


def boundary_payload(b: Boundary):
    return {
        "x_top": b.x_top,
        "x_bottom": b.x_bottom,
        "y_top": b.y_top,
        "y_bottom": b.y_bottom,
        "support": b.support,
        "strength": b.strength,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--roi-margin-x", type=float, default=0.06)
    ap.add_argument("--roi-top", type=float, default=0.10)
    ap.add_argument("--roi-bottom", type=float, default=0.94)
    args = ap.parse_args()

    img = read_image(args.input)
    roi_box = default_roi(img, args.roi_margin_x, args.roi_top, args.roi_bottom)
    boundaries = find_boundaries(img, roi_box)
    chosen = choose_largest_gap(img, boundaries, roi_box)
    annotated = draw_result(img, roi_box, boundaries, chosen, chosen[1] if chosen else [])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)

    payload = {
        "ok": chosen is not None,
        "input": args.input,
        "roi": {"x0": roi_box[0], "y0": roi_box[1], "x1": roi_box[2], "y1": roi_box[3]},
        "boundary_count": len(boundaries),
    }
    if chosen:
        best = chosen[0]
        payload["max_gap"] = {
            "score": best["score"],
            "width_px": best["width"],
            "edge_density": best["edge_density"],
            "centerline": {"p1": best["center_top"], "p2": best["center_bottom"]},
            "left_boundary": boundary_payload(best["left"]),
            "right_boundary": boundary_payload(best["right"]),
        }
        payload["candidates"] = [
            {
                "score": c["score"],
                "width_px": c["width"],
                "edge_density": c["edge_density"],
                "median_saturation": c["median_saturation"],
                "texture": c["texture"],
                "centerline": {"p1": c["center_top"], "p2": c["center_bottom"]},
                "left_boundary": boundary_payload(c["left"]),
                "right_boundary": boundary_payload(c["right"]),
            }
            for c in chosen[1][:10]
        ]

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
