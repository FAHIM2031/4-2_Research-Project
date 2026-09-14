"""
Vehicle Number Plate Recognition System
Detects, zooms in on, and extracts text from vehicle number plates.

Detection  : Hybrid pipeline —
               Stage 1 : YOLOv8n (ultralytics) finds the vehicle bounding box.
               Stage 2 : Contour search + perspective correction isolates plate.
OCR        : EasyOCR, multi-pass (4 preprocessing variants), allowlist filter,
             regex post-filter [A-Z0-9]{5,12}.
"""

import re
import cv2
import numpy as np
import easyocr
import os

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Plate regex — 4–12 uppercase alphanumeric chars (tune per locale)
PLATE_PATTERN = re.compile(r'^[A-Z0-9]{4,12}$')

# EasyOCR character allowlist — prevents non-plate symbols ever being returned
OCR_ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# ---------------------------------------------------------------------------
# YOLO weight selection
# ---------------------------------------------------------------------------
# Option A (default): yolov8n.pt — general COCO model, auto-downloaded ~6 MB.
#   YOLO detects "car / truck" objects; Stage 2 finds the plate sub-region.
#
# Option B (best accuracy): dedicated ANPR YOLOv8 weight.
#   Download from Roboflow:
#   https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
#   Then set: YOLO_WEIGHTS = "path/to/license_plate_yolov8n.pt"
#
YOLO_WEIGHTS = "yolov8n.pt"

# COCO vehicle class IDs (car=2, motorcycle=3, bus=5, truck=7)
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


class VehiclePlateReader:
    def __init__(self, languages=None, gpu=False, yolo_weights=YOLO_WEIGHTS,
                 yolo_conf=0.20):
        """
        Initialize the plate reader.

        Args:
            languages    : EasyOCR language list (default: ['en']).
            gpu          : True → CUDA for both YOLO and EasyOCR.
            yolo_weights : YOLOv8 weight file (auto-downloaded on first run).
            yolo_conf    : Minimum YOLO detection confidence (lowered to 0.20
                           for higher recall).
        """
        if languages is None:
            languages = ['en']

        self.gpu = gpu
        self.yolo_conf = yolo_conf
        device = 'cuda' if gpu else 'cpu'

        # YOLO
        print(f"Loading YOLO model '{yolo_weights}' on {device.upper()}…")
        self.model = YOLO(yolo_weights)
        self.model.to(device)
        print("YOLO ready!")

        # EasyOCR
        print("Initialising EasyOCR (first run downloads models ~100 MB)…")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("EasyOCR ready!")

    # =========================================================================
    # Stage 1 — YOLO vehicle detection
    # =========================================================================

    def detect_plates_yolo(self, image):
        """
        Run YOLOv8 and return vehicle bounding boxes.

        Returns:
            List of (x, y, w, h, conf) tuples sorted by confidence descending.
        """
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(rgb, conf=self.yolo_conf, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf   = float(box.conf[0].item())

                model_nc = getattr(getattr(self.model, 'model', None), 'nc', 80)
                if model_nc > 10 and cls_id not in VEHICLE_CLASS_IDS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append((int(x1), int(y1),
                                   int(x2 - x1), int(y2 - y1),
                                   conf))

        # Highest-confidence first
        detections.sort(key=lambda d: d[4], reverse=True)
        return detections

    # =========================================================================
    # Stage 2 — Plate sub-region isolation + perspective correction
    # =========================================================================

    @staticmethod
    def _order_points(pts):
        """
        Order 4 points as [top-left, top-right, bottom-right, bottom-left].
        Required for perspective transform.
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]     # top-left     (smallest x+y)
        rect[2] = pts[np.argmax(s)]     # bottom-right (largest  x+y)
        rect[1] = pts[np.argmin(diff)]  # top-right    (smallest x-y)
        rect[3] = pts[np.argmax(diff)]  # bottom-left  (largest  x-y)
        return rect

    def _perspective_correct(self, image, contour):
        """
        Apply a 4-point perspective transform to produce a flat, rectangular
        plate crop.  Falls back to the bounding-rect crop if contour is not
        exactly 4 corners.

        Returns:
            Warped plate image (ndarray).
        """
        peri   = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            pts  = approx.reshape(4, 2).astype(np.float32)
            rect = self._order_points(pts)
            tl, tr, br, bl = rect

            width_top    = np.linalg.norm(tr - tl)
            width_bottom = np.linalg.norm(br - bl)
            max_w = max(int(width_top), int(width_bottom), 1)

            height_right = np.linalg.norm(tr - br)
            height_left  = np.linalg.norm(tl - bl)
            max_h = max(int(height_right), int(height_left), 1)

            dst = np.array([[0, 0],
                            [max_w - 1, 0],
                            [max_w - 1, max_h - 1],
                            [0, max_h - 1]], dtype=np.float32)

            M       = cv2.getPerspectiveTransform(rect, dst)
            warped  = cv2.warpPerspective(image, M, (max_w, max_h))
            return warped

        # Fallback — just use bounding rect crop
        x, y, w, h = cv2.boundingRect(contour)
        return image[y:y + h, x:x + w]

    def _search_for_plate(self, roi):
        """
        Search a ROI for the best plate-shaped contour.

        Returns:
            (plate_crop, best_area) — plate_crop is None if not found.
        """
        gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges    = cv2.Canny(filtered, 30, 200)

        contours, _ = cv2.findContours(
            edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        best_crop = None
        best_area = 0

        for cnt in contours:
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if not (4 <= len(approx) <= 8):
                continue

            rx, ry, rw, rh = cv2.boundingRect(approx)
            area   = rw * rh
            aspect = rw / float(rh) if rh > 0 else 0

            # Match original model's constraints closely
            if 2.0 <= aspect <= 5.5 and area > 300 and area > best_area:
                pad   = 10
                h_roi = roi.shape[0]
                w_roi = roi.shape[1]
                x1    = max(0, rx - pad)
                y1    = max(0, ry - pad)
                x2    = min(w_roi, rx + rw + pad)
                y2    = min(h_roi, ry + rh + pad)
                crop  = roi[y1:y2, x1:x2]

                if crop is not None and crop.size > 0:
                    best_crop = crop
                    best_area = area

        return best_crop, best_area

    def find_plate_in_vehicle_crop(self, vehicle_crop):
        """
        Stage 2: isolate the license plate sub-region inside a vehicle crop.

        Strategy (cascade of ROIs):
          1. Lower 60% of crop  (front/rear bumper zone — most common)
          2. Full crop          (fallback if plate is high-mounted)
          3. Lower 40% of crop  (tight bumper fallback)

        Applies perspective correction when a clean 4-corner contour is found.

        Returns:
            Best plate crop (ndarray), or lower-half of vehicle_crop if nothing found.
        """
        h, w = vehicle_crop.shape[:2]

        rois = [
            vehicle_crop[int(h * 0.4):, :],    # lower 60%
            vehicle_crop,                        # full crop
            vehicle_crop[int(h * 0.6):, :],    # lower 40%
        ]

        overall_best = None
        overall_area = 0

        for roi in rois:
            if roi.size == 0:
                continue
            crop, area = self._search_for_plate(roi)
            if crop is not None and area > overall_area:
                overall_best = crop
                overall_area = area

        if overall_best is not None and overall_best.size > 0:
            return overall_best

        # Hard fallback — return lower half of vehicle crop
        return None

    def detect_plate_contours_global(self, image):
        """Original global contour search (fallback if YOLO/Stage2 fails)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(filtered, 30, 200)

        contours, _ = cv2.findContours(
            edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        best_crop = None
        best_area = 0

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if 4 <= len(approx) <= 8:
                x, y, w, h = cv2.boundingRect(approx)
                area = w * h
                aspect_ratio = w / float(h) if h > 0 else 0

                # Original constraints
                if 500 < area < 100000 and 2.0 < aspect_ratio < 5.5 and area > best_area:
                    pad = 10
                    h_img, w_img = image.shape[:2]
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(w_img, x + w + pad)
                    y2 = min(h_img, y + h + pad)
                    best_crop = image[y1:y2, x1:x2]
                    best_area = area

        return best_crop

    # =========================================================================
    # Enhancement helpers
    # =========================================================================

    def _sharpen(self, gray):
        """Unsharp mask — enhances character edges."""
        blurred   = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        return sharpened

    def enhance_for_ocr(self, plate_img):
        """
        Primary enhancement pipeline.

        Steps:
            1. Grayscale conversion
            2. NL-means denoising      — removes sensor/compression noise
            3. CLAHE                   — local contrast boost (clipLimit=3)
            4. Bilateral filter        — smooth while preserving edges
            5. Unsharp mask            — sharpen character edges
        """
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10,
                                            templateWindowSize=7,
                                            searchWindowSize=21)
        # 2. CLAHE (stronger clipLimit=3 vs previous 2)
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 3. Bilateral filter
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # 4. Sharpen
        sharpened = self._sharpen(filtered)

        return sharpened

    def _get_ocr_variants(self, plate_img):
        """
        Generate 4 differently preprocessed versions of a plate crop.
        Multi-pass OCR across all variants maximises the chance of a clean read.

        Variants:
            0 — CLAHE + bilateral + sharpening  (standard enhanced)
            1 — Otsu binarization               (dark background, white text)
            2 — Inverted Otsu                   (light background, dark text)
            3 — Adaptive Gaussian threshold     (handles uneven illumination)
        """
        # Ensure grayscale base
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # Variant 0 — standard enhanced
        v0 = self.enhance_for_ocr(plate_img)

        # Variant 1 — Otsu (global binarization)
        _, v1 = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Variant 2 — Inverted Otsu
        v2 = cv2.bitwise_not(v1)

        # Variant 3 — Adaptive Gaussian (handles shadows / gradients)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        v3 = cv2.adaptiveThreshold(filtered, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

        # Variant 4 — morphological close on enhanced (fills character gaps)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        v4 = cv2.morphologyEx(v0, cv2.MORPH_CLOSE, kernel)

        return [v0, v1, v2, v3, v4]

    # =========================================================================
    # OCR — multi-pass with allowlist + regex filter
    # =========================================================================

    def extract_text(self, plate_img):
        """
        Extract license plate text using multi-pass EasyOCR.

        Process:
            1. Generate 5 preprocessed variants of the plate crop.
            2. Run EasyOCR on each variant with character allowlist
               restricted to [A-Z0-9].
            3. Collect all word-level candidates.
            4. Validate each against PLATE_PATTERN ([A-Z0-9]{5,12}).
            5. Return the highest-confidence valid match.

        Returns:
            (plate_text: str, confidence: float)
        """
        # Guard against crops too small for reliable OCR
        h, w = plate_img.shape[:2] if len(plate_img.shape) >= 2 else (0, 0)
        if h < 10 or w < 20:
            return "", 0.0

        variants    = self._get_ocr_variants(plate_img)
        all_candidates = []

        for i, variant in enumerate(variants):
            try:
                ocr_results = self.reader.readtext(
                    variant,
                    paragraph=False,
                    detail=1,
                    allowlist=OCR_ALLOWLIST,
                )
                for (_bbox, text, conf) in ocr_results:
                    cleaned = ''.join(c for c in text if c.isalnum()).upper()
                    if PLATE_PATTERN.match(cleaned):
                        all_candidates.append((cleaned, conf))
            except Exception as exc:
                print(f"    OCR variant error: {exc}")
                continue

        if all_candidates:
            def score_candidate(cand):
                text, conf = cand
                score = conf
                has_letters = any(c.isalpha() for c in text)
                has_numbers = any(c.isdigit() for c in text)
                if has_letters and has_numbers:
                    score += 0.5  # Strong preference for alphanumeric mix
                if 6 <= len(text) <= 10:
                    score += 0.2  # Preference for typical plate length
                return score

            best_cand = max(all_candidates, key=score_candidate)
            return best_cand[0], best_cand[1]

        # Last resort — concatenate all words from first variant, validate
        try:
            raw_results = self.reader.readtext(
                variants[0], paragraph=False, detail=1,
                allowlist=OCR_ALLOWLIST)
            if raw_results:
                raw_text   = ''.join(t for (_, t, _) in raw_results)
                avg_conf   = sum(c for (_, _, c) in raw_results) / len(raw_results)
                cleaned    = ''.join(c for c in raw_text if c.isalnum()).upper()
                if PLATE_PATTERN.match(cleaned):
                    return cleaned, avg_conf
        except Exception:
            pass

        return "", 0.0

    # =========================================================================
    # Crop helper
    # =========================================================================

    def crop_and_zoom_plate(self, image, x, y, w, h,
                            zoom_factor=1, padding=10):
        """Extract, pad, and optionally upscale a region."""
        img_h, img_w = image.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        crop = image[y1:y2, x1:x2]

        if zoom_factor != 1:
            new_w = max(1, int(crop.shape[1] * zoom_factor))
            new_h = max(1, int(crop.shape[0] * zoom_factor))
            crop  = cv2.resize(crop, (new_w, new_h),
                               interpolation=cv2.INTER_CUBIC)
        return crop

    # =========================================================================
    # Image pipeline
    # =========================================================================

    def process_image(self, image_path, output_dir='output'):
        """
        Full image pipeline.

        Steps:
            1.  YOLO detects vehicle bounding boxes (Stage 1).
            2.  Contour search + perspective correction finds plate (Stage 2).
            3.  Plate crop upscaled 3× with bicubic interpolation.
            4.  Multi-pass EasyOCR + allowlist + regex extracts text.
            5.  Annotated image, zoomed crop, and raw Stage-2 crop saved.

        Returns:
            List of dicts: {text, position, confidence, yolo_confidence}
        """
        os.makedirs(output_dir, exist_ok=True)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from '{image_path}'")

        annotated = image.copy()

        # --- Stage 1: YOLO ---------------------------------------------------
        detections = self.detect_plates_yolo(image)
        results    = []

        if not detections:
            print("  No vehicles detected by YOLO. Trying global contour search...")
            global_crop = self.detect_plate_contours_global(image)
            if global_crop is not None:
                print("  ✔ Found plate using global contour search!")
                # Fake a detection at (0,0) for the rest of the pipeline
                detections = [(0, 0, image.shape[1], image.shape[0], 0.0)]
            else:
                print("  ✘ No plates found by YOLO or global search.")
                cv2.imwrite(os.path.join(output_dir, 'detected_plates.jpg'), annotated)
                return results
        else:
            print(f"  YOLO found {len(detections)} detection(s).")

        # --- Process each detection ------------------------------------------
        for idx, (x, y, w, h, yolo_conf) in enumerate(detections):
            print(f"\n  Detection {idx + 1}  (YOLO conf: {yolo_conf:.2%})")

            # Draw vehicle bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 255), 2)

            # Stage 1 crop — vehicle region (no zoom, small padding)
            vehicle_crop = self.crop_and_zoom_plate(
                image, x, y, w, h, zoom_factor=1, padding=8)

            # Stage 2 — isolate plate inside vehicle crop
            plate_raw = self.find_plate_in_vehicle_crop(vehicle_crop)

            if plate_raw is None:
                print("    ✘ Stage-2 failed to find plate in vehicle crop.")
                # Fallback to original global contour search on the whole image
                plate_raw = self.detect_plate_contours_global(image)
                if plate_raw is not None:
                    print("    ✔ Found plate using global contour fallback!")
                else:
                    plate_raw = vehicle_crop[int(vehicle_crop.shape[0] * 0.4):, :]

            # Save raw Stage-2 crop for inspection
            raw_path = os.path.join(output_dir, f'det{idx + 1}_stage2_raw.jpg')
            cv2.imwrite(raw_path, plate_raw)

            # Guard: skip crops too small
            ph, pw = plate_raw.shape[:2]
            if ph < 10 or pw < 20:
                print(f"    ✘ Stage-2 crop too small ({pw}×{ph}), skipping.")
                continue

            # Upscale 3× for OCR
            if pw < 200:
                plate_zoomed = cv2.resize(plate_raw,
                                          (pw * 3, ph * 3),
                                          interpolation=cv2.INTER_CUBIC)
            else:
                plate_zoomed = plate_raw.copy()

            # Multi-pass OCR on Stage-2 plate crop
            plate_text, ocr_conf = self.extract_text(plate_zoomed)
            from_fallback = False

            # ------------------------------------------------------------------
            # Fallback 1: if Stage-2 crop gave no text, run OCR directly on the
            # full vehicle crop.  Handles cases where the YOLO detection IS the
            # plate itself (e.g. a close-up plate image).
            # ------------------------------------------------------------------
            if not plate_text:
                vh, vw = vehicle_crop.shape[:2]
                if vh >= 10 and vw >= 20:
                    if vw < 400:
                        fb_zoom = cv2.resize(vehicle_crop,
                                             (vw * 2, vh * 2),
                                             interpolation=cv2.INTER_CUBIC)
                    else:
                        fb_zoom = vehicle_crop.copy()
                    plate_text, ocr_conf = self.extract_text(fb_zoom)
                    if plate_text:
                        from_fallback = True
                        print(f"    ✔ Plate (vehicle-crop fallback): "
                              f"{plate_text}  "
                              f"(OCR: {ocr_conf:.2%} | YOLO: {yolo_conf:.2%})")


            if plate_text:
                if not from_fallback:
                    print(f"    ✔ Plate: {plate_text}  "
                          f"(OCR: {ocr_conf:.2%} | YOLO: {yolo_conf:.2%})")

                results.append({
                    'text':            plate_text,
                    'position':        (x, y, w, h),
                    'confidence':      ocr_conf,
                    'yolo_confidence': yolo_conf,
                })

                # Save zoomed plate crop
                safe  = ''.join(c for c in plate_text if c.isalnum())
                cv2.imwrite(
                    os.path.join(output_dir,
                                 f'det{idx + 1}_{safe}_zoomed.jpg'),
                    plate_zoomed)

                # Annotate: green box + label
                cv2.rectangle(annotated,
                              (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = (f"{plate_text}  "
                         f"OCR:{ocr_conf:.0%}  DET:{yolo_conf:.0%}")
                cv2.putText(annotated, label,
                            (x, max(y - 12, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 0), 2)
            else:
                print(f"    ✘ No valid text extracted.")
                cv2.putText(annotated,
                            f"DET:{yolo_conf:.0%}",
                            (x, max(y - 12, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 140, 255), 2)

        # ------------------------------------------------------------------
        # Fallback 2: full-image OCR — runs ONCE if no plate found at all.
        # Catches cases where YOLO bbox is too poor for Stage 2 but the plate
        # text is still readable in the original image (e.g. marginal detections).
        # ------------------------------------------------------------------
        if not results:
            print("\n  Trying full-image OCR fallback…")
            ih, iw = image.shape[:2]
            full_zoom = image.copy()
            fb_text, fb_conf = self.extract_text(full_zoom)
            if fb_text:
                print(f"  ✔ Plate (full-image fallback): {fb_text}  "
                      f"(OCR: {fb_conf:.2%})")
                # Use the best YOLO detection's bbox for position
                bx, by, bw, bh, by_conf = detections[0]
                results.append({
                    'text':            fb_text,
                    'position':        (bx, by, bw, bh),
                    'confidence':      fb_conf,
                    'yolo_confidence': by_conf,
                })
                cv2.rectangle(annotated,
                              (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(annotated,
                            f"{fb_text}  OCR:{fb_conf:.0%}",
                            (bx, max(by - 12, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 0), 2)

        # Deduplicate by plate text (keep highest OCR confidence per text)
        seen = {}
        for r in results:
            t = r['text']
            if t not in seen or r['confidence'] > seen[t]['confidence']:
                seen[t] = r
        results = list(seen.values())

        # Save annotated image
        out_path = os.path.join(output_dir, 'detected_plates.jpg')
        cv2.imwrite(out_path, annotated)
        print(f"\n  Annotated image → {out_path}")

        return results

    # =========================================================================
    # Video pipeline
    # =========================================================================

    def process_video(self, video_path, output_dir='output', skip_frames=5):
        """
        Process a video frame-by-frame.

        Args:
            video_path  : Input video path.
            output_dir  : Directory for per-plate crop saves.
            skip_frames : Process every Nth frame.

        Returns:
            List of dicts: {text, confidence, yolo_confidence, frame}
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video '{video_path}'")

        os.makedirs(output_dir, exist_ok=True)

        frame_count     = 0
        detected_plates = []
        seen_texts      = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            detections = self.detect_plates_yolo(frame)

            for x, y, w, h, yolo_conf in detections[:3]:
                vehicle_crop = self.crop_and_zoom_plate(
                    frame, x, y, w, h, zoom_factor=1, padding=8)
                plate_raw    = self.find_plate_in_vehicle_crop(vehicle_crop)

                ph, pw = plate_raw.shape[:2]
                if ph < 10 or pw < 20:
                    continue

                if pw < 200:
                    plate_zoomed = cv2.resize(plate_raw, (pw * 2, ph * 2),
                                              interpolation=cv2.INTER_CUBIC)
                else:
                    plate_zoomed = plate_raw.copy()
                plate_text, ocr_conf = self.extract_text(plate_zoomed)

                if plate_text and ocr_conf > 0.25:
                    print(f"  Frame {frame_count}: {plate_text} "
                          f"(OCR {ocr_conf:.2%} / DET {yolo_conf:.2%})")

                    detected_plates.append({
                        'text':            plate_text,
                        'confidence':      ocr_conf,
                        'yolo_confidence': yolo_conf,
                        'frame':           frame_count,
                    })

                    if plate_text not in seen_texts:
                        seen_texts.add(plate_text)
                        cv2.imwrite(
                            os.path.join(output_dir,
                                         f'{plate_text}_f{frame_count}.jpg'),
                            plate_zoomed)

                    cv2.rectangle(frame,
                                  (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame,
                                f"{plate_text} ({ocr_conf:.0%})",
                                (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)

            cv2.imshow('Vehicle Plate Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return detected_plates


# =============================================================================
# Entry point
# =============================================================================

def main():
    """Test run — processes every supported image in Images/."""
    reader = VehiclePlateReader(gpu=False)

    images_dir    = 'Images'
    supported_ext = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

    image_files = sorted([
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in supported_ext
    ]) if os.path.isdir(images_dir) else []

    if not image_files:
        print(f"No images found in '{images_dir}/'.")
        print("Usage: reader.process_image('path/to/car.jpg')")
        return

    all_results = {}

    for image_path in image_files:
        basename = os.path.basename(image_path)
        stem     = os.path.splitext(basename)[0]
        out_dir  = os.path.join('output', stem)

        print(f"\n{'='*60}")
        print(f"IMAGE: {basename}")
        print('='*60)

        all_results[basename] = reader.process_image(image_path,
                                                      output_dir=out_dir)

    # Summary
    print(f"\n{'='*60}")
    print("FULL TEST SUMMARY")
    print('='*60)
    for img_name, results in all_results.items():
        print(f"\n📷  {img_name}")
        if results:
            for i, r in enumerate(results, 1):
                print(f"  {i}. Plate : {r['text']}")
                print(f"     OCR   : {r['confidence']:.2%}  |  "
                      f"YOLO  : {r['yolo_confidence']:.2%}")
        else:
            print("  ✘ No plates detected.")
    print('='*60)

    # Uncomment to process video:
    # plates = reader.process_video('vehicle_video.mp4')
    # print(plates)


if __name__ == '__main__':
    main()
