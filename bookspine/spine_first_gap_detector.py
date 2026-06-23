#!/usr/bin/env python3
"""Detect the largest gap by detecting book spines first.

Pipeline:
1. Detect candidate book-spine vertical boundaries.
2. Build candidate spine strips between neighboring boundaries.
3. Keep only strips that look like real book spines.
4. Measure gaps only between adjacent detected spines.
5. Return the largest gap centerline.

This intentionally avoids choosing a whole book width or a shelf-side blank
area as a "gap", because a valid gap must be bounded by two detected spines.
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
    source: str

    @property
    def x_mid(self) -> float:
        return 0.5 * (self.x_top + self.x_bottom)

    def x_at(self, y: float) -> float:
        if abs(self.y_bottom - self.y_top) < 1e-6:
            return self.x_mid
        t = (y - self.y_top) / (self.y_bottom - self.y_top)
        return self.x_top + t * (self.x_bottom - self.x_top)


@dataclass
class Spine:
    left: Boundary
    right: Boundary
    score: float
    width_px: float
    y_top: float
    y_bottom: float
    bookness: float
    edge_density: float


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def roi_from_args(img: np.ndarray, margin_x: float, top: float, bottom: float):
    h, w = img.shape[:2]
    return int(w * margin_x), int(h * top), int(w * (1.0 - margin_x)), int(h * bottom)


def normalize_signal(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    lo, hi = np.percentile(v, [5, 95])
    if hi <= lo + 1e-6:
        return np.zeros_like(v, dtype=np.float32)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def column_bookness(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    sx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
    edges = cv2.Canny(gray, 35, 110).mean(axis=0) / 255.0
    sat = np.median(hsv[:, :, 1], axis=0) / 255.0

    # Estimate shelf/background color from the least saturated pixels in ROI.
    sat_img = hsv[:, :, 1]
    bg_mask = sat_img <= np.percentile(sat_img, 30)
    if np.count_nonzero(bg_mask) < 100:
        bg_lab = np.median(lab.reshape(-1, 3), axis=0)
    else:
        bg_lab = np.median(lab[bg_mask], axis=0)
    dist = np.linalg.norm(lab - bg_lab, axis=2).mean(axis=0)

    score = (
        0.36 * normalize_signal(sx)
        + 0.28 * normalize_signal(edges)
        + 0.20 * normalize_signal(sat)
        + 0.16 * normalize_signal(dist)
    )
    return cv2.GaussianBlur(score.reshape(1, -1), (1, 17), 0).ravel()


def cluster_x_positions(items, max_gap: float):
    items = sorted(items, key=lambda v: v[0])
    clusters = []
    for item in items:
        if not clusters or item[0] - clusters[-1][-1][0] > max_gap:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return clusters


def hough_boundaries(roi_bgr: np.ndarray, x_offset: int, y_offset: int) -> list[Boundary]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 110)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=36, minLineLength=70, maxLineGap=18)
    if lines is None:
        return []

    raw = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < 70:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle < 76 or abs(dx) > 42:
            continue
        if y1 <= y2:
            xt, yt, xb, yb = x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset
        else:
            xt, yt, xb, yb = x2 + x_offset, y2 + y_offset, x1 + x_offset, y1 + y_offset
        raw.append((0.5 * (xt + xb), float(xt), float(yt), float(xb), float(yb), length))

    boundaries = []
    for cluster in cluster_x_positions(raw, max_gap=8):
        weights = np.array([c[5] for c in cluster], dtype=np.float32)
        weights /= max(float(weights.sum()), 1e-6)
        boundaries.append(
            Boundary(
                x_top=float(sum(w * c[1] for w, c in zip(weights, cluster))),
                x_bottom=float(sum(w * c[3] for w, c in zip(weights, cluster))),
                y_top=float(min(c[2] for c in cluster)),
                y_bottom=float(max(c[4] for c in cluster)),
                support=len(cluster),
                source="hough",
            )
        )
    return boundaries


def projection_boundaries(roi_bgr: np.ndarray, x_offset: int, y_offset: int) -> list[Boundary]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    sx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
    sx = cv2.GaussianBlur(normalize_signal(sx).reshape(1, -1), (1, 11), 0).ravel()
    threshold = max(0.30, float(np.percentile(sx, 72)))

    peaks = []
    for x in range(2, len(sx) - 2):
        if sx[x] > threshold and sx[x] >= sx[x - 1] and sx[x] >= sx[x + 1]:
            peaks.append((float(x + x_offset), float(sx[x])))

    selected = []
    for x, score in sorted(peaks, key=lambda v: v[1], reverse=True):
        if all(abs(x - sx0) >= 7 for sx0, _ in selected):
            selected.append((x, score))

    h = roi_bgr.shape[0]
    return [
        Boundary(x, x, float(y_offset), float(y_offset + h - 1), 1, "projection")
        for x, _ in sorted(selected, key=lambda v: v[0])
    ]


def color_profile_boundaries(roi_bgr: np.ndarray, x_offset: int, y_offset: int) -> list[Boundary]:
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    profile = np.median(lab, axis=0)
    smooth = np.vstack(
        [cv2.GaussianBlur(profile[:, i].reshape(1, -1), (1, 15), 0).ravel() for i in range(3)]
    ).T
    grad = np.linalg.norm(np.gradient(smooth, axis=0), axis=1)
    threshold = max(float(np.percentile(grad, 75)), 4.0)

    peaks = []
    for x in range(2, len(grad) - 2):
        if grad[x] > threshold and grad[x] >= grad[x - 1] and grad[x] >= grad[x + 1]:
            peaks.append((float(x + x_offset), float(grad[x])))

    selected = []
    for x, score in sorted(peaks, key=lambda v: v[1], reverse=True):
        if all(abs(x - sx0) >= 8 for sx0, _ in selected):
            selected.append((x, score))

    h = roi_bgr.shape[0]
    return [
        Boundary(x, x, float(y_offset), float(y_offset + h - 1), 2, "color_profile")
        for x, _ in sorted(selected, key=lambda v: v[0])
    ]


def merge_boundaries(boundaries: list[Boundary]) -> list[Boundary]:
    merged = []
    for cluster in cluster_x_positions([(b.x_mid, b) for b in boundaries], max_gap=7):
        bs = [item[1] for item in cluster]
        weights = np.array(
            [
                max(1, b.support)
                * (1.8 if b.source == "hough" else 1.35 if b.source == "color_profile" else 1.0)
                for b in bs
            ],
            dtype=np.float32,
        )
        weights /= max(float(weights.sum()), 1e-6)
        merged.append(
            Boundary(
                x_top=float(sum(w * b.x_top for w, b in zip(weights, bs))),
                x_bottom=float(sum(w * b.x_bottom for w, b in zip(weights, bs))),
                y_top=float(min(b.y_top for b in bs)),
                y_bottom=float(max(b.y_bottom for b in bs)),
                support=int(sum(b.support for b in bs)),
                source="+".join(sorted(set(b.source for b in bs))),
            )
        )
    merged.sort(key=lambda b: b.x_mid)
    return merged


def patch_metrics(img: np.ndarray, left: Boundary, right: Boundary, y_top: float, y_bottom: float) -> tuple[float, float, float]:
    xs = [
        left.x_at(y_top),
        left.x_at(y_bottom),
        right.x_at(y_top),
        right.x_at(y_bottom),
    ]
    x0 = max(0, int(min(xs)))
    x1 = min(img.shape[1], int(max(xs)))
    y0 = max(0, int(y_top))
    y1 = min(img.shape[0], int(y_bottom))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return 0.0, 1.0, 0.0
    patch = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 35, 110)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    texture = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))) / 255.0)
    saturation = float(np.median(hsv[:, :, 1]) / 255.0)
    return edge_density, texture, saturation


def build_spines(
    img: np.ndarray,
    boundaries: list[Boundary],
    bookness: np.ndarray,
    roi_box,
    min_width: float,
    max_width: float,
) -> list[Spine]:
    x0, y0, x1, y1 = roi_box
    spines: list[Spine] = []
    bookness_floor = max(0.22, float(np.percentile(bookness, 48)))

    for left, right in zip(boundaries, boundaries[1:]):
        y_top = max(left.y_top, right.y_top, y0)
        y_bottom = min(left.y_bottom, right.y_bottom, y1)
        if y_bottom - y_top < 120:
            continue
        width_top = right.x_at(y_top) - left.x_at(y_top)
        width_bottom = right.x_at(y_bottom) - left.x_at(y_bottom)
        width = 0.5 * (width_top + width_bottom)
        if width < min_width or width > max_width:
            continue

        col0 = max(0, int(min(left.x_at(y_top), left.x_at(y_bottom)) - x0))
        col1 = min(len(bookness), int(max(right.x_at(y_top), right.x_at(y_bottom)) - x0))
        if col1 <= col0 + 1:
            continue
        interval_bookness = float(np.mean(bookness[col0:col1]))
        edge_density, texture, saturation = patch_metrics(img, left, right, y_top, y_bottom)
        spine_score = 0.55 * interval_bookness + 0.25 * min(texture * 2.0, 1.0) + 0.20 * min(saturation * 1.5, 1.0)

        # A very wide, very low-texture interval is more likely to be an
        # insertion opening than a normal spine. Leave it for the gap stage.
        if width > 45 and edge_density < 0.08 and texture < 0.09:
            continue

        if interval_bookness < bookness_floor and spine_score < 0.34:
            continue
        spines.append(
            Spine(
                left=left,
                right=right,
                score=float(spine_score),
                width_px=float(width),
                y_top=float(y_top),
                y_bottom=float(y_bottom),
                bookness=interval_bookness,
                edge_density=edge_density,
            )
        )

    # Remove near-duplicate spine strips, keeping the stronger one.
    spines.sort(key=lambda s: 0.5 * (s.left.x_mid + s.right.x_mid))
    filtered: list[Spine] = []
    for spine in spines:
        center = 0.5 * (spine.left.x_mid + spine.right.x_mid)
        if filtered:
            prev = filtered[-1]
            prev_center = 0.5 * (prev.left.x_mid + prev.right.x_mid)
            overlap = min(spine.right.x_mid, prev.right.x_mid) - max(spine.left.x_mid, prev.left.x_mid)
            if abs(center - prev_center) < 8 or overlap > 0.55 * min(spine.width_px, prev.width_px):
                if spine.score > prev.score:
                    filtered[-1] = spine
                continue
        filtered.append(spine)
    return filtered


def split_wide_spines(
    spines: list[Spine],
    boundaries: list[Boundary],
    img: np.ndarray,
    bookness: np.ndarray,
    roi_box,
    min_width: float,
    max_width: float,
) -> list[Spine]:
    if not spines:
        return spines

    normal_widths = [s.width_px for s in spines if s.width_px <= max_width * 0.72]
    median_width = float(np.median(normal_widths)) if normal_widths else 24.0
    split_threshold = max(55.0, median_width * 2.4)
    rebuilt: list[Spine] = []

    for spine in spines:
        if spine.width_px < split_threshold:
            rebuilt.append(spine)
            continue

        inner = [
            b
            for b in boundaries
            if spine.left.x_mid + min_width <= b.x_mid <= spine.right.x_mid - min_width
        ]
        if not inner:
            rebuilt.append(spine)
            continue

        cut = max(inner, key=lambda b: b.support)
        left_candidates = build_spines(
            img,
            [spine.left, cut],
            bookness,
            roi_box,
            min_width,
            max_width,
        )
        right_candidates = build_spines(
            img,
            [cut, spine.right],
            bookness,
            roi_box,
            min_width,
            max_width,
        )
        if left_candidates and right_candidates:
            rebuilt.extend(left_candidates)
            rebuilt.extend(right_candidates)
        else:
            rebuilt.append(spine)

    rebuilt.sort(key=lambda s: 0.5 * (s.left.x_mid + s.right.x_mid))
    return rebuilt


def gap_between_spines(
    img: np.ndarray,
    left_spine: Spine,
    right_spine: Spine,
    bookness: np.ndarray,
    roi_box,
    max_gap_width: float,
):
    y_top = max(left_spine.y_top, right_spine.y_top, roi_box[1])
    y_bottom = min(left_spine.y_bottom, right_spine.y_bottom, roi_box[3])
    if y_bottom - y_top < 100:
        return None

    left_edge = left_spine.right
    right_edge = right_spine.left
    width_top = right_edge.x_at(y_top) - left_edge.x_at(y_top)
    width_bottom = right_edge.x_at(y_bottom) - left_edge.x_at(y_bottom)
    width = 0.5 * (width_top + width_bottom)
    if width < 3 or width > max_gap_width:
        return None

    x0 = roi_box[0]
    col0 = max(0, int(min(left_edge.x_at(y_top), left_edge.x_at(y_bottom)) - x0))
    col1 = min(len(bookness), int(max(right_edge.x_at(y_top), right_edge.x_at(y_bottom)) - x0))
    if col1 <= col0 + 1:
        gap_bookness = 1.0
    else:
        gap_bookness = float(np.mean(bookness[col0:col1]))

    edge_density, texture, saturation = patch_metrics(img, left_edge, right_edge, y_top, y_bottom)
    if gap_bookness > 0.72 and width > 12:
        return None

    # Reject shelf-side blank regions, but allow very wide low-texture gaps
    # bounded by one strong spine and one weaker/tilted spine face.
    if left_spine.right.x_mid - roi_box[0] < 25 or roi_box[2] - right_spine.left.x_mid < 25:
        return None
    strong_two_sides = left_spine.score >= 0.50 and right_spine.score >= 0.50
    wide_opening = width >= 45 and gap_bookness < 0.45 and edge_density < 0.12 and texture < 0.12
    one_strong_side = max(left_spine.score, right_spine.score) >= 0.50 and min(left_spine.score, right_spine.score) >= 0.28
    if not strong_two_sides and not (wide_opening and one_strong_side):
        return None

    score = width * (1.0 - min(gap_bookness, 0.85)) * (1.0 - min(edge_density * 2.2, 0.75))
    return {
        "score": float(score),
        "width_px": float(width),
        "gap_bookness": gap_bookness,
        "edge_density": edge_density,
        "texture": texture,
        "saturation": saturation,
        "left_spine": left_spine,
        "right_spine": right_spine,
        "centerline": {
            "p1": [float(0.5 * (left_edge.x_at(y_top) + right_edge.x_at(y_top))), float(y_top)],
            "p2": [float(0.5 * (left_edge.x_at(y_bottom) + right_edge.x_at(y_bottom))), float(y_bottom)],
        },
        "left_gap_edge": {
            "p1": [float(left_edge.x_at(y_top)), float(y_top)],
            "p2": [float(left_edge.x_at(y_bottom)), float(y_bottom)],
        },
        "right_gap_edge": {
            "p1": [float(right_edge.x_at(y_top)), float(y_top)],
            "p2": [float(right_edge.x_at(y_bottom)), float(y_bottom)],
        },
    }


def choose_largest_spine_gap(img: np.ndarray, spines: list[Spine], bookness: np.ndarray, roi_box, max_gap_width: float):
    gaps = []
    for left_spine, right_spine in zip(spines, spines[1:]):
        gap = gap_between_spines(img, left_spine, right_spine, bookness, roi_box, max_gap_width)
        if gap:
            gaps.append(gap)
    gaps.sort(key=lambda g: g["width_px"], reverse=True)
    return (gaps[0], gaps) if gaps else (None, [])


def spine_payload(spine: Spine):
    return {
        "left_x": spine.left.x_mid,
        "right_x": spine.right.x_mid,
        "width_px": spine.width_px,
        "score": spine.score,
        "bookness": spine.bookness,
        "edge_density": spine.edge_density,
        "y_top": spine.y_top,
        "y_bottom": spine.y_bottom,
    }


def draw(img: np.ndarray, roi_box, spines: list[Spine], best_gap) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = roi_box
    cv2.rectangle(out, (x0, y0), (x1, y1), (80, 255, 80), 1)

    for idx, spine in enumerate(spines, start=1):
        pts = np.array(
            [
                [spine.left.x_at(spine.y_top), spine.y_top],
                [spine.right.x_at(spine.y_top), spine.y_top],
                [spine.right.x_at(spine.y_bottom), spine.y_bottom],
                [spine.left.x_at(spine.y_bottom), spine.y_bottom],
            ],
            dtype=np.int32,
        )
        cv2.polylines(out, [pts], True, (120, 220, 120), 1, cv2.LINE_AA)
        cv2.putText(out, str(idx), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 120), 1)

    if best_gap:
        le = best_gap["left_gap_edge"]
        re = best_gap["right_gap_edge"]
        cl = best_gap["centerline"]
        cv2.line(out, tuple(map(int, le["p1"])), tuple(map(int, le["p2"])), (255, 0, 0), 3, cv2.LINE_AA)
        cv2.line(out, tuple(map(int, re["p1"])), tuple(map(int, re["p2"])), (0, 165, 255), 3, cv2.LINE_AA)
        cv2.line(out, tuple(map(int, cl["p1"])), tuple(map(int, cl["p2"])), (0, 0, 255), 3, cv2.LINE_AA)
        label = f"largest spine gap {best_gap['width_px']:.1f}px"
        cv2.putText(out, label, tuple(map(int, cl["p1"])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--roi-margin-x", type=float, default=0.06)
    ap.add_argument("--roi-top", type=float, default=0.10)
    ap.add_argument("--roi-bottom", type=float, default=0.94)
    ap.add_argument("--min-spine-width", type=float, default=8.0)
    ap.add_argument("--max-spine-width", type=float, default=95.0)
    ap.add_argument("--max-gap-width", type=float, default=120.0)
    args = ap.parse_args()

    img = read_image(args.input)
    roi_box = roi_from_args(img, args.roi_margin_x, args.roi_top, args.roi_bottom)
    x0, y0, x1, y1 = roi_box
    roi = img[y0:y1, x0:x1]

    bookness = column_bookness(roi)
    boundaries = merge_boundaries(
        hough_boundaries(roi, x0, y0)
        + projection_boundaries(roi, x0, y0)
        + color_profile_boundaries(roi, x0, y0)
    )
    spines = build_spines(img, boundaries, bookness, roi_box, args.min_spine_width, args.max_spine_width)
    spines = split_wide_spines(spines, boundaries, img, bookness, roi_box, args.min_spine_width, args.max_spine_width)
    best_gap, gaps = choose_largest_spine_gap(img, spines, bookness, roi_box, args.max_gap_width)

    out = draw(img, roi_box, spines, best_gap)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)

    payload = {
        "ok": best_gap is not None,
        "input": args.input,
        "roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "boundary_count": len(boundaries),
        "spine_count": len(spines),
        "spines": [spine_payload(s) for s in spines],
    }
    if best_gap:
        payload["max_gap"] = {
            k: v
            for k, v in best_gap.items()
            if k not in {"left_spine", "right_spine"}
        }
        payload["max_gap"]["left_spine"] = spine_payload(best_gap["left_spine"])
        payload["max_gap"]["right_spine"] = spine_payload(best_gap["right_spine"])
        payload["gap_candidates"] = [
            {
                k: v
                for k, v in gap.items()
                if k not in {"left_spine", "right_spine"}
            }
            for gap in gaps[:10]
        ]

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
