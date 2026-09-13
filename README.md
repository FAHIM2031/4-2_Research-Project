# Vehicle Number Plate Recognition System

A Python-based system that automatically detects, zooms in on, and extracts text from vehicle number plates using computer vision and OCR.

## Features

- **Automatic Plate Detection**: Detects number plates using contour analysis and shape recognition
- **Smart Zoom**: Automatically zooms into detected plate regions for better text extraction
- **Image Enhancement**: Applies preprocessing techniques for improved OCR accuracy
- **Text Extraction**: Uses Tesseract OCR to read plate numbers
- **Batch Processing**: Process single images or entire video streams
- **Visual Output**: Saves annotated images with detected plates highlighted

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR

**Windows:**
1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install it (default location: `C:\Program Files\Tesseract-OCR`)
3. If installed elsewhere, update the path in `vehicle_plate_reader.py`:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Path\To\tesseract.exe'
   ```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

## Usage

### Basic Image Processing

```python
from vehicle_plate_reader import VehiclePlateReader

# Initialize the reader
reader = VehiclePlateReader()

# Process an image
results = reader.process_image('path/to/vehicle_image.jpg')

# Print results
for result in results:
    print(f"Plate Number: {result['text']}")
    print(f"Position: {result['position']}")
```

### Process Video Stream

```python
reader = VehiclePlateReader()

# Process video file
detected_plates = reader.process_video('vehicle_video.mp4')

print(f"All detected plates: {detected_plates}")
```

### Command Line Usage

```bash
python vehicle_plate_reader.py
```

## How It Works

1. **Image Preprocessing**
   - Converts image to grayscale
   - Applies bilateral filtering to reduce noise
   - Performs edge detection using Canny algorithm

2. **Plate Detection**
   - Finds contours in the edge-detected image
   - Filters by shape (4 corners) and aspect ratio (2:1 to 5.5:1)
   - Filters by area to eliminate false positives

3. **Zoom & Enhancement**
   - Extracts the plate region with padding
   - Upscales by 2-3x for better OCR accuracy
   - Applies adaptive thresholding
   - Denoises and cleans up the image

4. **Text Extraction**
   - Uses Tesseract OCR with optimized settings
   - Restricts to alphanumeric characters
   - Cleans and validates the output

## Output

The system creates an `output/` directory containing:
- `detected_plates.jpg` - Annotated original image with detected plates
- `plate_X_zoomed.jpg` - Individual zoomed plate images

## Customization

### Adjust Detection Parameters

```python
reader = VehiclePlateReader()

# Modify plate size constraints
reader.min_plate_area = 500      # Minimum plate area in pixels
reader.max_plate_area = 30000    # Maximum plate area in pixels
```

### Change Zoom Factor

```python
# More zoom for better text recognition
zoomed_plate = reader.zoom_and_enhance_plate(image, x, y, w, h, zoom_factor=4)
```

### Custom OCR Configuration

Modify the `extract_text` method to adjust Tesseract settings:
```python
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
```

## Troubleshooting

### "No number plates detected"
- Ensure the image has good lighting and resolution
- Plates should be clearly visible and not too small
- Adjust `min_plate_area` and `max_plate_area` parameters

### Poor Text Recognition
- Increase the zoom factor
- Ensure Tesseract is properly installed
- Try different OCR configurations for your region's plate format

### Tesseract Not Found Error
- Verify Tesseract installation
- Set the correct path in the script:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```

## Improving Accuracy

For better results:
1. **Use high-resolution images** (at least 1280x720)
2. **Ensure good lighting** in source images/videos
3. **Train a custom cascade classifier** for your specific plate format
4. **Fine-tune detection parameters** based on your use case
5. **Use deep learning models** like YOLO for more robust detection

## Advanced: Using Deep Learning

For production systems, consider using:
- **YOLO (You Only Look Once)** for plate detection
- **EasyOCR or PaddleOCR** for better text recognition
- **Custom CNN models** trained on your regional plate formats

## Example Results

```
==================================================
DETECTION RESULTS:
==================================================
1. Plate Number: ABC1234
   Position: (145, 230, 180, 45)
2. Plate Number: XYZ5678
   Position: (520, 310, 175, 42)
==================================================
```

## License

This project is open source and available for educational and commercial use.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
