"""Montage straight from DICOMs: one row per file, labelled with computed
plane (from ImageOrientationPatient) + short series description.

Blog-friendly: shows that one study yields several planes / series.
"""
from __future__ import annotations
import glob
import os
import re
import numpy as np
import pydicom
from PIL import Image, ImageDraw, ImageFont

SRC = sorted(glob.glob("tests/dcms/*.dcm"))
OUT = "tests/dcms/preview/montage_planes.png"
TARGET_H = 400
GAP = 14
MARGIN = 12


def font(size: int) -> ImageFont.ImageFont:
    for p in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def plane_of(ds) -> str:
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None:
        return "?"
    r = np.asarray(iop[:3], float)
    c = np.asarray(iop[3:], float)
    n = np.cross(r, c)
    return ["Sagittal", "Coronal", "Axial"][int(np.argmax(np.abs(n)))]


def short_desc(ds) -> str:
    d = str(getattr(ds, "SeriesDescription", "")).replace("_", " ").replace("CS", "")
    d = re.sub(r"\s+", " ", d).strip()
    return d or "series"


def to_grayscale(ds) -> Image.Image:
    a = ds.pixel_array.astype(np.float32)
    sl = float(getattr(ds, "RescaleSlope", 1) or 1)
    ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
    a = a * sl + ic
    lo, hi = np.percentile(a, [1, 99])
    b = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        b = 1.0 - b
    return Image.fromarray((b * 255).astype(np.uint8)).convert("L")


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    items = []
    for f in SRC:
        ds = pydicom.dcmread(f, force=True)
        im = to_grayscale(ds)
        w = max(1, round(im.width * TARGET_H / im.height))
        label = f"{plane_of(ds)} · {short_desc(ds)}"
        items.append((im.resize((w, TARGET_H), Image.LANCZOS), label))

    label_h = 36
    total_w = 2 * MARGIN + sum(i.width for i, _ in items) + GAP * (len(items) - 1)
    total_h = 2 * MARGIN + TARGET_H + label_h
    canvas = Image.new("RGB", (total_w, total_h), "white")
    fnt = font(17)
    x = MARGIN
    for im, label in items:
        canvas.paste(im, (x, MARGIN))
        ImageDraw.Draw(canvas).text((x + 2, MARGIN + TARGET_H + 8), label,
                                    fill=(50, 50, 50), font=fnt)
        x += im.width + GAP
    canvas.save(OUT)
    print("saved", OUT, canvas.size)
    for _, label in items:
        print("  label:", label)


if __name__ == "__main__":
    main()
