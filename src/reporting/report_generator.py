"""
Structured Report Generator — Stage 4
Author: Chenduluru Siva | 7151CEM
"""

import os
import sys
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    DISEASE_LABELS, REPORT_OUTPUT_DIR,
    CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CLASSIFICATION_THRESHOLD
)

DISEASE_DESCRIPTIONS = {
    "Atelectasis":       "partial or complete collapse of lung tissue",
    "Cardiomegaly":      "enlargement of the cardiac silhouette",
    "Effusion":          "accumulation of fluid in the pleural space",
    "Infiltration":      "abnormal material filling the airspaces",
    "Mass":              "focal opacity >3 cm — may represent neoplasm",
    "Nodule":            "focal opacity ≤3 cm — further investigation warranted",
    "Pneumonia":         "consolidation consistent with infection",
    "Pneumothorax":      "air in the pleural space — urgent evaluation advised",
    "Consolidation":     "airspace opacification — infection, haemorrhage or oedema",
    "Edema":             "bilateral opacities suggesting pulmonary oedema",
    "Emphysema":         "hyperinflation with reduced vascular markings",
    "Fibrosis":          "reticular opacity suggesting interstitial fibrosis",
    "Pleural_Thickening":"thickening of the pleural lining",
    "Hernia":            "herniation of abdominal contents into the thorax"
}
DISEASE_URGENCY = {
    "Pneumothorax": "URGENT", "Pneumonia": "PRIORITY",
    "Effusion": "PRIORITY",   "Edema": "PRIORITY",
    "Mass": "PRIORITY",       "Consolidation": "PRIORITY",
    "Cardiomegaly": "ROUTINE","Atelectasis": "ROUTINE",
    "Nodule": "ROUTINE",      "Infiltration": "ROUTINE",
    "Emphysema": "ROUTINE",   "Fibrosis": "ROUTINE",
    "Pleural_Thickening": "ROUTINE", "Hernia": "ROUTINE"
}


@dataclass
class Finding:
    disease:     str
    probability: float
    confidence:  str
    urgency:     str


@dataclass
class StructuredReport:
    image_filename: str
    report_id:      str
    timestamp:      str
    qa_score:       float
    qa_acceptable:  bool
    findings:       List[Finding]
    no_finding:     bool
    impression:     str
    recommendation: str
    disclaimer:     str
    full_text:      str


class ReportGenerator:
    def __init__(self, threshold=CLASSIFICATION_THRESHOLD,
                 conf_high=CONFIDENCE_HIGH, conf_medium=CONFIDENCE_MEDIUM):
        self.threshold   = threshold
        self.conf_high   = conf_high
        self.conf_medium = conf_medium

    def _confidence(self, p):
        if p >= self.conf_high:   return "HIGH"
        elif p >= self.conf_medium: return "MEDIUM"
        else:                     return "LOW"

    def _impression(self, findings):
        if not findings:
            return ("No acute cardiopulmonary findings identified. "
                    "Interpret in clinical context.")
        urgent   = [f for f in findings if f.urgency == "URGENT"]
        priority = [f for f in findings if f.urgency == "PRIORITY"]
        routine  = [f for f in findings if f.urgency == "ROUTINE"]
        parts = []
        if urgent:
            parts.append(f"URGENT: {', '.join(f.disease for f in urgent)} — immediate review.")
        if priority:
            parts.append(f"Priority: {', '.join(f.disease for f in priority)} — prompt correlation.")
        if routine:
            parts.append(f"Routine: {', '.join(f.disease for f in routine)}.")
        return " ".join(parts)

    def _recommendation(self, findings):
        if not findings:
            return "Routine follow-up as clinically indicated."
        urgencies = {f.urgency for f in findings}
        if "URGENT"   in urgencies: return "Immediate radiologist verification required."
        if "PRIORITY" in urgencies: return "Expedited radiologist review recommended."
        return "Standard radiologist review recommended."

    def generate(self, probs, filename, qa_score=100.0, qa_acceptable=True):
        ts        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_id = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        findings = []
        for i, label in enumerate(DISEASE_LABELS):
            if probs[i] >= self.threshold:
                findings.append(Finding(
                    disease=label,
                    probability=round(float(probs[i]), 4),
                    confidence=self._confidence(probs[i]),
                    urgency=DISEASE_URGENCY.get(label, "ROUTINE")
                ))

        urgency_order = {"URGENT": 0, "PRIORITY": 1, "ROUTINE": 2}
        findings.sort(key=lambda f: (urgency_order[f.urgency], -f.probability))

        no_finding     = len(findings) == 0
        impression     = self._impression(findings)
        recommendation = self._recommendation(findings)
        disclaimer     = (
            "DISCLAIMER: AI-assisted output. NOT a substitute for radiologist "
            "interpretation. All findings must be verified by a qualified clinician."
        )

        lines = [
            "=" * 65,
            "  AUTOMATED CHEST X-RAY ANALYSIS REPORT",
            "  (AI Decision Support — Research Prototype)",
            "=" * 65,
            f"  Report ID  : {report_id}",
            f"  Generated  : {ts}",
            f"  Image File : {filename}",
            f"  QA Score   : {qa_score}/100  "
            f"({'ACCEPTABLE' if qa_acceptable else 'SUBOPTIMAL'})",
            "-" * 65,
        ]

        if not qa_acceptable:
            lines += ["  ⚠ WARNING: Suboptimal image quality. Results may be unreliable.",
                      "-" * 65]

        lines.append("  FINDINGS:\n")
        if no_finding:
            lines.append("  • No significant thoracic abnormalities detected.")
        else:
            for f in findings:
                star = "★★★" if f.confidence=="HIGH" else "★★ " if f.confidence=="MEDIUM" else "★  "
                lines.append(f"  [{f.urgency:<8}] {star} {f.disease:<25} (p={f.probability:.3f})")
                lines.append(f"             {DISEASE_DESCRIPTIONS.get(f.disease, '')}\n")

        lines += [
            "-" * 65,
            f"  IMPRESSION:\n  {impression}\n",
            f"  RECOMMENDATION:\n  {recommendation}\n",
            "-" * 65,
            f"  {disclaimer}",
            "=" * 65,
        ]
        full_text = "\n".join(lines)

        return StructuredReport(
            image_filename=filename, report_id=report_id,
            timestamp=ts, qa_score=qa_score, qa_acceptable=qa_acceptable,
            findings=findings, no_finding=no_finding,
            impression=impression, recommendation=recommendation,
            disclaimer=disclaimer, full_text=full_text
        )

    def save_report(self, report, output_dir=REPORT_OUTPUT_DIR):
        os.makedirs(output_dir, exist_ok=True)
        safe = report.image_filename.replace("/", "_").replace(".png", "")
        path = os.path.join(output_dir, f"{report.report_id}_{safe}.txt")
        with open(path, "w") as f:
            f.write(report.full_text)
        return path


if __name__ == "__main__":
    np.random.seed(42)
    probs = np.random.rand(14)
    probs[7] = 0.91   # Pneumothorax
    probs[6] = 0.74   # Pneumonia
    gen    = ReportGenerator()
    report = gen.generate(probs, "demo_patient.png", qa_score=91.2)
    print(report.full_text)
    path = gen.save_report(report)
    print(f"\n[✓] Saved → {path}")