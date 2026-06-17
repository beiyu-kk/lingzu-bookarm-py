#!/usr/bin/env python3
"""Low-compute book-gap detector for bookshelf images.

The detector is intentionally simple:
1. Crop a central shelf ROI.
2. Compute a per-column seam score from left/right contrast and local texture.
3. Smooth, pick peaks, and suppress nearby duplicates.
4. Draw candidate seam centers back onto the image.

It works on normal RGB images as a proxy and also accepts 16-bit depth PNGs.
For real D435 depth maps, prefer the depth channel and keep RGB only as a helper.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SeamCandidate:
    x: int
    score: float


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        arr = img.astype(np.float32)
    elif img.shape[2] == 4:
        arr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        arr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lo, hi = np.percentile(arr, [2, 98])
    if hi <= lo + 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr.astype(np.float32)


def central_roi(gray: np.ndarray, margin_x: float = 0.05, top: float = 0.10, bottom: float = 0.95):
    h, w = gray.shape[:2]
    x0 = int(w * margin_x)
    x1 = int(w * (1.0 - margin_x))
    y0 = int(h * top)
    y1 = int(h * bottom)
    return gray[y0:y1, x0:x1], (x0, y0, x1, y1)


def seam_score(gray_roi: np.ndarray, half_window: int = 4, center_window: int = 3) -> np.ndarray:
    """Column-wise score for likely seam centers.

    Higher score means stronger difference between left and right neighborhoods
    while the center stays relatively low-texture.
    """
    h, w = gray_roi.shape[:2]
    x0 = half_window + 1
    x1 = w - half_window - 1
    scores = []

    for x in range(x0, x1):
        left = gray_roi[:, x - half_window : x]
        right = gray_roi[:, x : x + half_window]
        center = gray_roi[:, max(0, x - center_window) : min(w, x + center_window)]

        left_mean = float(left.mean())
        right_mean = float(right.mean())
        center_std = float(center.std())
        center_tex = float(np.mean(np.abs(np.diff(center, axis=0)))) if center.shape[0] > 1 else 0.0

        contrast = abs(right_mean - left_mean)
        penalty = 0.08 + center_std + 0.5 * center_tex
        scores.append(contrast / penalty)

    scores = np.asarray(scores, dtype=np.float32)
    scores = cv2.GaussianBlur(scores.reshape(1, -1), (1, 31), 0).ravel()
    return scores


def local_maxima(scores: np.ndarray, threshold: float, min_distance: int) -> list[SeamCandidate]:
    peaks: list[SeamCandidate] = []
    for i in range(1, len(scores) - 1):
        if scores[i] <= threshold:
            continue
        if scores[i] > scores[i - 1] and scores[i] > scores[i + 1]:
            peaks.append(SeamCandidate(i, float(scores[i])))

    peaks.sort(key=lambda p: p.score, reverse=True)
    selected: list[SeamCandidate] = []
    for p in peaks:
        if all(abs(p.x - s.x) >= min_distance for s in selected):
            selected.append(p)
    selected.sort(key=lambda p: p.x)
    return selected


def detect_book_seams(
    img: np.ndarray,
    max_candidates: int = 20,
    min_distance: int = 18,
    keep_top_percent: float = 0.35,
    roi_margin_x: float = 0.05,
    roi_top: float = 0.10,
    roi_bottom: float = 0.95,
) -> tuple[np.ndarray, list[SeamCandidate], tuple[int, int, int, int]]:
    gray = to_gray(img)
    roi, (x0, y0, x1, y1) = central_roi(gray, margin_x=roi_margin_x, top=roi_top, bottom=roi_bottom)
    scores = seam_score(roi)

    thr = float(np.percentile(scores, 100 * (1.0 - keep_top_percent)))
    candidates = local_maxima(scores, threshold=thr, min_distance=min_distance)

    # Keep only strong central seams: rank by score and suppress very weak peaks.
    if candidates:
        strong_floor = max(np.median(scores) * 1.15, np.percentile(scores, 85))
        candidates = [c for c in candidates if c.score >= strong_floor]

    if len(candidates) > max_candidates:
        candidates = sorted(candidates, key=lambda p: p.score, reverse=True)[:max_candidates]
        candidates.sort(key=lambda p: p.x)

    # For grasping, prefer fewer but stronger seams.
    if len(candidates) > 10:
        candidates = sorted(candidates, key=lambda p: p.score, reverse=True)[:10]
        candidates.sort(key=lambda p: p.x)

    return scores, candidates, (x0, y0, x1, y1)


def draw_candidates(img: np.ndarray, candidates: list[SeamCandidate], roi_box, scores: np.ndarray) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = roi_box
    roi_w = max(1, x1 - x0)
    score_max = float(max(scores.max(), 1e-6))

    for idx, cand in enumerate(candidates, start=1):
        x = x0 + cand.x
        conf = cand.score / score_max
        color = (0, int(200 + 55 * min(conf, 1.0)), 255 - int(120 * min(conf, 1.0)))
        cv2.line(out, (x, y0), (x, y1), color, 2)
        cv2.rectangle(out, (max(0, x - 7), y0), (min(out.shape[1] - 1, x + 7), y1), color, 1)
        label = f"{idx}:{cand.score:.2f}"
        ty = max(20, y0 + 18 + (idx % 2) * 18)
        tx = min(max(0, x + 4), out.shape[1] - 100)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.rectangle(out, (x0, y0), (x1, y1), (80, 255, 80), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input image path")
    ap.add_argument("--output", required=True, help="Output annotated image path")
    ap.add_argument("--json", default=None, help="Optional output JSON path")
    ap.add_argument("--max-candidates", type=int, default=20)
    ap.add_argument("--min-distance", type=int, default=18)
    ap.add_argument("--keep-top-percent", type=float, default=0.35)
    ap.add_argument("--roi-margin-x", type=float, default=0.05)
    ap.add_argument("--roi-top", type=float, default=0.10)
    ap.add_argument("--roi-bottom", type=float, default=0.95)
    args = ap.parse_args()

    img = load_image(args.input)
    scores, candidates, roi_box = detect_book_seams(
        img,
        max_candidates=args.max_candidates,
        min_distance=args.min_distance,
        keep_top_percent=args.keep_top_percent,
        roi_margin_x=args.roi_margin_x,
        roi_top=args.roi_top,
        roi_bottom=args.roi_bottom,
    )
    annotated = draw_candidates(img, candidates, roi_box, scores)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)

    result = {
        "input": args.input,
        "candidates": [{"x": int(c.x + roi_box[0]), "score": float(c.score)} for c in candidates],
        "roi": {"x0": roi_box[0], "y0": roi_box[1], "x1": roi_box[2], "y1": roi_box[3]},
    }
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
