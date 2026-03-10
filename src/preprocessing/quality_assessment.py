"""
Image Quality Assessment Module — Stage 1
Author: Chenduluru Siva | 7151CEM
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    QA_BRIGHTNESS_MIN, QA_BRIGHTNESS_MAX,
    QA_CONTRAST_MIN, QA_BLUR_THRESHOLD, QA_BLACK_RATIO_MAX
)


@dataclass
class QualityReport:
    filename:      str
    is_acceptable: bool
    overall_score: float
    brightness:    float
    contrast:      float
    sharpness:     float
    black_ratio:   float
    issues:        List[str]
    recommendation: str


class ImageQualityAssessor:
    def __init__(
        self,
        brightness_min=QA_BRIGHTNESS_MIN,
        brightness_max=QA_BRIGHTNESS_MAX,
        contrast_min=QA_CONTRAST_MIN,
        blur_threshold=QA_BLUR_THRESHOLD,
        black_ratio_max=QA_BLACK_RATIO_MAX
    ):
        self.brightness_min  = brightness_min
        self.brightness_max  = brightness_max
        self.contrast_min    = contrast_min
        self.blur_threshold  = blur_threshold
        self.black_ratio_max = black_ratio_max

    def _brightness(self, gray):  return float(np.mean(gray))
    def _contrast(self,   gray):  return float(np.std(gray))
    def _sharpness(self,  gray):
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    def _black_ratio(self, gray):
        return float(np.sum(gray < 10) / gray.size)

    def _score(self, value, lo, hi):
        if lo <= value <= hi: return 1.0
        elif value < lo:      return max(0.0, value / lo)
        else:                 return max(0.0, 1.0 - (value - hi) / hi)

    def assess(self, image_input) -> QualityReport:
        if isinstance(image_input, str):
            filename = os.path.basename(image_input)
            img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return QualityReport(
                    filename=filename, is_acceptable=False,
                    overall_score=0, brightness=0, contrast=0,
                    sharpness=0, black_ratio=1,
                    issues=["Cannot read image file"],
                    recommendation="Reject — unreadable file"
                )
        elif isinstance(image_input, Image.Image):
            filename = "pil_image"
            img = np.array(image_input.convert("L"))
        elif isinstance(image_input, np.ndarray):
            filename = "array_image"
            img = image_input if len(image_input.shape) == 2 \
                  else cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError("image_input must be file path, PIL.Image, or np.ndarray")

        brightness  = self._brightness(img)
        contrast    = self._contrast(img)
        sharpness   = self._sharpness(img)
        black_ratio = self._black_ratio(img)

        issues = []
        scores = []

        b_score = self._score(brightness, self.brightness_min, self.brightness_max)
        scores.append(b_score)
        if brightness < self.brightness_min:
            issues.append(f"Under-exposed (brightness={brightness:.1f})")
        elif brightness > self.brightness_max:
            issues.append(f"Over-exposed (brightness={brightness:.1f})")

        c_score = min(1.0, contrast / (self.contrast_min * 3))
        scores.append(c_score)
        if contrast < self.contrast_min:
            issues.append(f"Low contrast (std={contrast:.1f})")

        s_score = min(1.0, sharpness / (self.blur_threshold * 5))
        scores.append(s_score)
        if sharpness < self.blur_threshold:
            issues.append(f"Blurry image (Laplacian={sharpness:.1f})")

        br_score = max(0.0, 1.0 - (black_ratio / self.black_ratio_max))
        scores.append(br_score)
        if black_ratio > self.black_ratio_max:
            issues.append(f"Excessive black borders ({100*black_ratio:.1f}%)")

        overall_score = 100 * np.mean(scores)
        is_acceptable = len(issues) == 0

        if is_acceptable:
            recommendation = "Accept — proceed to classification"
        elif overall_score >= 50:
            recommendation = "Caution — minor issues; proceed with reduced confidence"
            is_acceptable  = True
        else:
            recommendation = "Reject — significant quality issues"

        return QualityReport(
            filename=filename,
            is_acceptable=is_acceptable,
            overall_score=round(overall_score, 1),
            brightness=round(brightness, 1),
            contrast=round(contrast, 1),
            sharpness=round(sharpness, 1),
            black_ratio=round(black_ratio, 4),
            issues=issues,
            recommendation=recommendation
        )

    def assess_batch(self, image_paths):
        return [self.assess(p) for p in image_paths]

    def summary_stats(self, reports):
        total      = len(reports)
        acceptable = sum(1 for r in reports if r.is_acceptable)
        rejected   = total - acceptable
        avg_score  = np.mean([r.overall_score for r in reports])
        return {
            "total": total, "acceptable": acceptable,
            "rejected": rejected,
            "rejection_rate": round(rejected / total * 100, 2),
            "avg_quality_score": round(avg_score, 1)
        }


if __name__ == "__main__":
    assessor = ImageQualityAssessor()
    # Quick self-test
    normal = np.random.randint(50, 180, (1024, 1024), dtype=np.uint8)
    r = assessor.assess(Image.fromarray(normal))
    print(f"Score: {r.overall_score}/100  |  Acceptable: {r.is_acceptable}")
    print(f"Recommendation: {r.recommendation}")