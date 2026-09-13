"""
Vehicle Number Plate Recognition System
Detects, zooms in on, and extracts text from vehicle number plates
Uses EasyOCR for text recognition (pure Python, no external dependencies)
"""

import cv2
import numpy as np
import easyocr
import os


class VehiclePlateReader:
    def __init__(self, languages=['en'], gpu=False):
        """
        Initialize the plate reader with EasyOCR

        Args:
            languages: List of languages for OCR (default: ['en'] for English)
                      Add more if needed: ['en', 'es'], ['en', 'ar'], etc.
            gpu: Set to True to use GPU acceleration (requires CUDA)
        """
        print("Initializing EasyOCR (first run downloads models ~100MB)...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("EasyOCR ready!")

        # Detection parameters
        self.min_plate_area = 500
        self.max_plate_area = 100000

    def preprocess_image(self, image):
        """Preprocess image for better plate detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Apply edge detection
        edges = cv2.Canny(filtered, 30, 200)

        return gray, filtered, edges

    def detect_plate_contours(self, edges):
        """Detect potential number plate regions using contours"""
        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        plate_candidates = []

        for contour in contours:
            # Approximate the contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Number plates are typically rectangular (4 corners)
            if 4 <= len(approx) <= 8:
                x, y, w, h = cv2.boundingRect(approx)
                area = w * h
                aspect_ratio = w / float(h)

                # Number plates typically have aspect ratio between 2:1 and 5:1
                if (self.min_plate_area < area < self.max_plate_area and
                    2.0 < aspect_ratio < 5.5):
                    plate_candidates.append((x, y, w, h, area))

        return plate_candidates

    def zoom_and_enhance_plate(self, image, x, y, w, h, zoom_factor=2):
        """Extract and enhance the plate region"""
        # Add some padding
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)

        # Extract plate region
        plate_img = image[y_start:y_end, x_start:x_end]

        # Zoom in (resize)
        zoomed_height = int(plate_img.shape[0] * zoom_factor)
        zoomed_width = int(plate_img.shape[1] * zoom_factor)
        zoomed_plate = cv2.resize(plate_img, (zoomed_width, zoomed_height),
                                  interpolation=cv2.INTER_CUBIC)

        return zoomed_plate

    def enhance_for_ocr(self, plate_img):
        """Enhance plate image for better OCR accuracy"""
        # Convert to grayscale if needed
        if len(plate_img.shape) == 3:
            gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_plate = plate_img

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_plate)

        # Apply bilateral filtering to smooth while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return filtered

    def extract_text(self, plate_img):
        """Extract text from plate image using EasyOCR"""
        try:
            # Enhance image for OCR
            enhanced = self.enhance_for_ocr(plate_img)

            # EasyOCR returns list of (bbox, text, confidence)
            results = self.reader.readtext(enhanced)

            if not results:
                return "", 0.0

            # Extract text and average confidence
            extracted_text = ""
            total_confidence = 0.0

            for detection in results:
                text = detection[1]
                confidence = detection[2]
                extracted_text += text
                total_confidence += confidence

            # Calculate average confidence
            avg_confidence = total_confidence / len(results) if results else 0.0

            # Clean up the text - keep alphanumeric
            cleaned_text = ''.join(c for c in extracted_text if c.isalnum()).upper()

            return cleaned_text, avg_confidence

        except Exception as e:
            print(f"Error during OCR: {e}")
            return "", 0.0

    def process_image(self, image_path, output_dir='output'):
        """Main processing pipeline"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        original = image.copy()

        # Preprocess
        gray, filtered, edges = self.preprocess_image(image)

        # Detect plate candidates
        plate_candidates = self.detect_plate_contours(edges)
        

        results = []

        if not plate_candidates:
            print("No number plates detected")
            return results

        print(f"Found {len(plate_candidates)} potential plates")

        # Process each candidate
        for idx, (x, y, w, h, area) in enumerate(plate_candidates):
            print(f"\nProcessing plate candidate {idx + 1}...")

            # Draw rectangle on original
            cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Zoom and extract plate
            zoomed_plate = self.zoom_and_enhance_plate(image, x, y, w, h, zoom_factor=3)

            # Extract text with confidence score
            plate_text, confidence = self.extract_text(zoomed_plate)

            if plate_text and len(plate_text) > 2:  # Basic validation (2+ characters)
                print(f"Detected plate: {plate_text} (confidence: {confidence:.2%})")
                results.append({
                    'text': plate_text,
                    'position': (x, y, w, h),
                    'confidence': confidence
                })

                # Save zoomed plate
                plate_filename = os.path.join(output_dir, f'plate_{idx + 1}_{plate_text}_zoomed.jpg')
                cv2.imwrite(plate_filename, zoomed_plate)

                # Add text to original image with confidence
                display_text = f"{plate_text} ({confidence:.0%})"
                cv2.putText(original, display_text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                print(f"Could not extract valid text from plate {idx + 1}")

        # Save annotated image
        output_path = os.path.join(output_dir, 'detected_plates.jpg')
        cv2.imwrite(output_path, original)
        print(f"\nAnnotated image saved to: {output_path}")

        return results

    def process_video(self, video_path, output_dir='output', skip_frames=5):
        """Process video stream for plate detection"""
        cap = cv2.VideoCapture(video_path)
        os.makedirs(output_dir, exist_ok=True)

        frame_count = 0
        detected_plates = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every nth frame to improve performance
            if frame_count % skip_frames != 0:
                continue

            # Preprocess
            gray, filtered, edges = self.preprocess_image(frame)

            # Detect plates
            plate_candidates = self.detect_plate_contours(edges)

            for x, y, w, h, area in plate_candidates[:3]:  # Top 3 candidates
                # Zoom and extract
                zoomed_plate = self.zoom_and_enhance_plate(frame, x, y, w, h, zoom_factor=2)

                # Extract text with confidence
                plate_text, confidence = self.extract_text(zoomed_plate)

                if plate_text and len(plate_text) > 2 and confidence > 0.3:
                    print(f"Frame {frame_count}: Detected plate {plate_text} (confidence: {confidence:.2%})")

                    detected_plates.append({
                        'text': plate_text,
                        'confidence': confidence,
                        'frame': frame_count
                    })

                    # Save the zoomed plate
                    plate_filename = os.path.join(output_dir, f'plate_{plate_text}_frame{frame_count}.jpg')
                    cv2.imwrite(plate_filename, zoomed_plate)

                    # Draw on frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{plate_text} ({confidence:.0%})", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display frame (optional)
            cv2.imshow('Vehicle Plate Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return detected_plates


def main():
    """Example usage"""
    reader = VehiclePlateReader()

    # Example: Process a single image
    image_path = 'E:\\Claude local session\\Vehicle_plate_reader\\Images\\car3.webp'  # Replace with your image path

    if os.path.exists(image_path):
        print("Processing image...")
        results = reader.process_image(image_path)

        print("\n" + "="*50)
        print("DETECTION RESULTS:")
        print("="*50)
        for i, result in enumerate(results, 1):
            print(f"{i}. Plate Number: {result['text']}")
            print(f"   Position: {result['position']}")
        print("="*50)
    else:
        print(f"Image not found: {image_path}")
        print("\nUsage example:")
        print("reader = VehiclePlateReader()")
        print("results = reader.process_image('path/to/vehicle_image.jpg')")

    # Uncomment to process video
    # video_path = 'vehicle_video.mp4'
    # plates = reader.process_video(video_path)
    # print(f"Detected plates in video: {plates}")


if __name__ == '__main__':
    main()
