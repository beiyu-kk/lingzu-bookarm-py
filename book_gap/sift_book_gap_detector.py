#!/usr/bin/env python3
"""SIFT-based book localization and seam extraction.

This script takes a template book-spine image and a bookshelf scene image.
It localizes the specific book with SIFT + homography, then returns the
left/right seams as the projected left/right book edges.

The seams are not forced to be vertical. They follow the detected perspective
of the book in the scene, which is better for robot insertion planning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MatchResult:
    ok: bool
    homography: np.ndarray | None
    corners: np.ndarray | None
    inliers: int
    matches: int
    confidence: float


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def preprocess(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def make_sift():
    return cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01, edgeThreshold=12, sigma=1.2)


def detect_book(template_bgr: np.ndarray, scene_bgr: np.ndarray) -> MatchResult:
    template_gray = preprocess(cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY))
    scene_gray = preprocess(cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY))

    sift = make_sift()
    kp1, des1 = sift.detectAndCompute(template_gray, None)
    kp2, des2 = sift.detectAndCompute(scene_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return MatchResult(False, None, None, 0, 0, 0.0)

    index_params = dict(algorithm=1, trees=8)
    search_params = dict(checks=100)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    raw = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.78 * n.distance:
            good.append(m)

    if len(good) < 8:
        return MatchResult(False, None, None, 0, len(good), 0.0)

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return MatchResult(False, None, None, 0, len(good), 0.0)

    inliers = int(mask.ravel().sum())
    confidence = float(inliers / max(1, len(good)))
    if inliers < 8:
        return MatchResult(False, H, None, inliers, len(good), confidence)

    h, w = template_gray.shape[:2]
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H)
    return MatchResult(True, H, projected, inliers, len(good), confidence)


def segment_distance(p1, p2):
    v = np.array(p2, dtype=np.float32) - np.array(p1, dtype=np.float32)
    return float(np.linalg.norm(v))


def edge_to_dict(p1, p2):
    return {
        "p1": [float(p1[0]), float(p1[1])],
        "p2": [float(p2[0]), float(p2[1])],
        "length": segment_distance(p1, p2),
    }


def draw_overlay(scene: np.ndarray, corners: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    out = scene.copy()
    pts = corners.reshape(-1, 2).astype(np.int32)
    cv2.polylines(out, [pts], True, color, 3, cv2.LINE_AA)
    labels = ["TL", "TR", "BR", "BL"]
    for pt, label in zip(pts, labels):
        cv2.circle(out, tuple(pt), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(out, label, (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return out


def rectified_view(template: np.ndarray, scene: np.ndarray, H: np.ndarray, out_w: int = 600) -> np.ndarray:
    h, w = template.shape[:2]
    ratio = h / max(1, w)
    out_h = max(1, int(out_w * ratio))
    return cv2.warpPerspective(scene, np.linalg.inv(H), (out_w, out_h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Template spine image")
    ap.add_argument("--scene", required=True, help="Shelf scene image")
    ap.add_argument("--output", required=True, help="Annotated output image")
    ap.add_argument("--json", default=None, help="Optional JSON output")
    ap.add_argument("--rectified", default=None, help="Optional rectified crop output")
    args = ap.parse_args()

    template = read_image(args.template)
    scene = read_image(args.scene)
    result = detect_book(template, scene)

    payload = {
        "ok": result.ok,
        "matches": result.matches,
        "inliers": result.inliers,
        "confidence": result.confidence,
        "template": args.template,
        "scene": args.scene,
    }

    if result.ok and result.corners is not None:
        corners = result.corners
        pts = corners.reshape(-1, 2)
        left_edge = edge_to_dict(pts[0], pts[3])
        right_edge = edge_to_dict(pts[1], pts[2])

        payload["corners"] = {
            "tl": [float(v) for v in pts[0]],
            "tr": [float(v) for v in pts[1]],
            "br": [float(v) for v in pts[2]],
            "bl": [float(v) for v in pts[3]],
        }
        payload["left_seam"] = left_edge
        payload["right_seam"] = right_edge

        annotated = draw_overlay(scene, corners)
        # draw seam lines a little thicker for visibility
        cv2.line(annotated, tuple(pts[0].astype(int)), tuple(pts[3].astype(int)), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(annotated, tuple(pts[1].astype(int)), tuple(pts[2].astype(int)), (0, 165, 255), 2, cv2.LINE_AA)
    else:
        annotated = scene.copy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)

    if args.rectified and result.ok and result.homography is not None:
        rect = rectified_view(template, scene, result.homography)
        rect_path = Path(args.rectified)
        rect_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(rect_path), rect)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
