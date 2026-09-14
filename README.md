# Vehicle Number Plate Recognition System

A high-precision, hybrid vehicle number plate recognition pipeline that combines deep learning (YOLO, EasyOCR) with traditional computer vision (OpenCV contour detection).

This system was completely overhauled to resolve localization accuracy issues, successfully returning to the highly-accurate tight-cropping capabilities of older computer vision models while leveraging the robustness of modern machine learning for vehicle detection and character recognition.

## Architecture

The system uses a 4-tier pipeline to guarantee maximum accuracy and robustness against false positives:

1. **Stage 1: Vehicle Detection (YOLOv8)**
   - A pre-trained YOLOv8 Nano model scans the image to identify vehicles.
   - *Why?* This eliminates background noise (like rectangular signs, windows, or fences) that traditional contour searches often confuse for license plates.

2. **Stage 2: Precision Plate Localization (OpenCV)**
   - Inside the YOLO vehicle bounding box, the system runs a highly constrained OpenCV Canny edge and contour search.
   - It filters for specific geometric constraints (`aspect_ratio` 2.0 to 5.5, `area` > 300) and precisely crops the plate using a 10px padded bounding box. 
   - *Note:* Perspective warping and dilations were deliberately removed because they distorted the plate boundaries, resulting in lower OCR accuracy.

3. **Stage 2 Fallback: Global Contour Search**
   - If YOLO fails to detect a vehicle (e.g., the image is already a close-up of a plate), or if the local contour search fails, the system falls back to a global contour search across the entire image. This matches the exact behavior of the highly successful original model.

4. **Stage 3: Optical Character Recognition (EasyOCR)**
   - The tightly-cropped plate image is enhanced (CLAHE, Bilateral Filtering, Gaussian Blur, and Unsharp Masking) to maximize text clarity.
   - EasyOCR reads the text using a strict alphanumeric allowlist.
   - A custom heuristic scoring system analyzes multiple image variants (grayscale, thresholded, morphologically opened) and aggressively favors strings containing a mix of letters and numbers (to filter out words like "BRASIL" or static bumper text).

## Performance & Accuracy Improvements

The refactored pipeline drastically outperforms the previous YOLO+Tesseract base model in localization precision and OCR accuracy. For a detailed breakdown of test results, see the **[Pipeline Accuracy Report](accuracy_report.md)**.

### Key Improvements Over Previous Version:
1. **Perfect Tight-Cropping Restored**: The bounding boxes perfectly wrap the text boundaries without absorbing surrounding car bumpers, maximizing the resolution given to the OCR engine. This fixes the major complaint about the previous YOLO implementation.
2. **Eliminated "S" vs "5" Confusion**: By adding smart conditional upscaling limits, we stopped EasyOCR from smudging characters on already large images. This completely resolved the issue where a "5" was misread as an "S".
3. **No Perspective Warping Distortion**: Removed the `cv2.warpPerspective` and `cv2.dilate` logic that was distorting characters and causing OCR failures.
4. **Global Fallback Recovery**: When YOLO fails to detect a vehicle (e.g. `vehicle-identification-plate.webp`), the system perfectly falls back to a global image contour search to still extract the plate text.

## Setup & Installation

1. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline:
   ```bash
   python vehicle_plate_reader.py
   ```

## Output
All processed images, cropped plates, and annotated bounding boxes are saved to the `output/` directory for manual inspection.
