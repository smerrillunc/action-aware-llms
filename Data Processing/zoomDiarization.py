import cv2
import numpy as np
import whisper
import os
import tempfile
from moviepy import VideoFileClip
import re
import time
import easyocr
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt


class ZoomSpeakerDiarization:
    def __init__(self, video_path):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.speaker_changes = []
        
        #self.reader = easyocr.Reader(['en'], gpu=True)
        self.reader = ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    gpu=True )
        self.whisper_model = whisper.load_model("large-v2")
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel('/work/users/s/m/smerrill/models/EDSR_Tensorflow/models/EDSR_x4.pb')
        self.sr.setModel("edsr", 4)


    def ocr_name_from_box(self, box_img, debug=True, paragraph=True):
        """
        Attempts OCR on the input box image. Only applies super-resolution
        if OCR confidence is below threshold.
        """
        def run_ocr(image, label=""):
            print(f"Running OCR {label}...")
            start = time.perf_counter()
            ocr_results = self.reader.ocr(image, cls=True)
            end = time.perf_counter()
            print(f"OCR {label} took {end - start:.4f} seconds")
            return ocr_results

        # First try OCR on original image
        ocr_results = run_ocr(box_img, label="(original)")

        # Extract best text and confidence
        text = ""
        confidence = 0.0
        if ocr_results and ocr_results[0]:
            text = ocr_results[0][0][1][0]
            confidence = ocr_results[0][0][1][1]
        print(f"OCR confidence (original): {confidence:.2f}")

        # If confidence is low, apply super-resolution and try again
        # probably no text if less than 0.1
        if (confidence < 0.8) and (confidence > 0.1):
            print("Confidence < 0.8, trying super-resolution...")
            start = time.perf_counter()
            upscaled = self.sr.upsample(box_img)
            end = time.perf_counter()
            print(f"Super-resolution took {end - start:.4f} seconds")

            ocr_results = run_ocr(upscaled, label="(upscaled)")

            # Extract new best result
            if ocr_results and ocr_results[0]:
                text = ocr_results[0][0][1][0]
                confidence = ocr_results[0][0][1][1]
            print(f"OCR confidence (upscaled): {confidence:.2f}")

            if debug:
                plt.figure(figsize=(6,2))
                plt.imshow(upscaled)
                plt.title(f"OCR (upscaled): {text}")
                plt.axis('off')
                plt.show()
        else:
            if debug:
                plt.figure(figsize=(6,2))
                plt.imshow(box_img)
                plt.title(f"OCR (original): {text}")
                plt.axis('off')
                plt.show()

        return text if text else "No Speaker"


    def crop_black_border(self, frame, threshold=10):
        """
        Crops black borders from an image (NumPy array).
        Args:
            frame: np.ndarray, input image (BGR or RGB)
            threshold: int, pixel values <= threshold are considered black
        Returns:
            Cropped image as np.ndarray
        """
        if len(frame.shape) == 3:
            # For color images
            mask = np.any(frame > threshold, axis=2)
        else:
            # For grayscale images
            mask = frame > threshold

        coords = np.argwhere(mask)
        if coords.size == 0:
            raise ValueError("No non-black content found in frame.")

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top

        return frame[y0:y1, x0:x1]

    def find_speaker(self, frame, overlay=True, debug=True):
        """
        Attempts to find the active speaker region in a Zoom screenshot.
        Handles:
          - Green/yellow border (participant grid)
          - Fixed overlay in upper right (screen share mode)
          - Yellow pop-up (Zoom's speaker label)
        Returns:
          frame_with_box, (x, y, w, h) of detected box, cropped box image
        """
        frame = self.crop_black_border(frame)
        frame_disp = frame.copy()
        h, w = frame.shape[:2]
        detected_box = None
        cutout = None

        # 1. Try to find green/yellow border (participant grid)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([20, 40, 60])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_rect = None
        max_area = 0
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                x, y, ww, hh = cv2.boundingRect(approx)
                area = ww * hh
                if area > max_area and ww > 40 and hh > 40:
                    max_rect = (x, y, ww, hh)
                    max_area = area
        if max_rect:
            detected_box = max_rect
            if overlay:
                cv2.rectangle(frame_disp, (detected_box[0], detected_box[1]),
                              (detected_box[0] + detected_box[2], detected_box[1] + detected_box[3]),
                              (0, 255, 0), 3)
            cutout = frame[detected_box[1]:detected_box[1]+detected_box[3], detected_box[0]:detected_box[0]+detected_box[2]]
            if debug:
                print("Detected green/yellow border (participant grid).")
        else:
            # Fallback: Fixed overlay in upper right (screen share mode)
            # Use proportions to handle resizing
            overlay_w = int(w * 0.18)  # ~26% of width (tune as needed)
            overlay_h = int(h * 0.18)  # ~20% of height (tune as needed)
            x1 = w - overlay_w
            y1 = 0
            detected_box = (x1, y1, overlay_w, overlay_h)
            cutout = frame[y1:y1+overlay_h, x1:x1+overlay_w]
            if overlay:
                cv2.rectangle(frame_disp, (x1, y1), (x1+overlay_w, y1+overlay_h), (255, 0, 0), 3)
            if debug:
                print("Detected fixed overlay (upper right).")

        if debug:
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
            plt.title("Detected Speaker Box")
            plt.axis('off')
            plt.show()
            if cutout is not None:
                plt.figure(figsize=(3, 2))
                plt.imshow(cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB))
                plt.title("Speaker Crop")
                plt.axis('off')
                plt.show()

        return frame_disp, detected_box, cutout
    

    def determine_active_speaker(self, frame):
        result_frame, speaker_box, cutout = self.find_speaker(frame, overlay=True, debug=False)
        name = self.ocr_name_from_box(self.crop_name_only(cutout), debug=False)
        return name

   
    def process_video(self, sampling_rate=1):
        """Process the video to create speaker diarization"""
        print(f"Processing video: {self.video_path}")
        print(f"Total frames: {self.total_frames}, FPS: {self.fps}")

        # Process video frames at the specified sampling rate
        frame_interval = int(self.fps / sampling_rate)
        current_frame = 0
        last_speaker = None
        
        while current_frame < self.total_frames:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = self.capture.read()
            
            if not ret:
                break
                
            timestamp = current_frame / self.fps

            try:
                active_speaker = self.determine_active_speaker(frame)
            except:
                current_frame += frame_interval
                continue
                          
            # Record speaker change
            if active_speaker != last_speaker:
                self.speaker_changes.append((timestamp, active_speaker))
                last_speaker = active_speaker
            
            # Output progress
            if current_frame % (frame_interval * 10) == 0:
                progress = (current_frame / self.total_frames) * 100
                print(f"Progress: {progress:.1f}% - Current speaker: {active_speaker}")
            
            current_frame += frame_interval
        
        # Clean up
        self.capture.release()
        return self.speaker_changes
    
    def crop_name_only(self, box_img, height_ratio=0.22, width_ratio=0.5, bottom_padding=2):
        h, w = box_img.shape[:2]
        name_h = int(h * height_ratio)
        name_w = int(w * width_ratio)
        # Remove bottom padding to avoid colored lines
        name_region = box_img[h - name_h:h - bottom_padding, 0:name_w]
        return name_region