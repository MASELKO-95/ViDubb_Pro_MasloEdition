#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/utils.py – Helper functions for ViDubb
Fixed: DeepFace enforce_detection=False for tolerance
"""

import os
import shutil
import cv2
import numpy as np
from deepface import DeepFace


def merge_overlapping_periods(period_dict):
    """Merge overlapping time periods for the same speaker."""
    sorted_periods = sorted(period_dict.items(), key=lambda x: x[0][0])
    if not sorted_periods:
        return {}
    merged_periods = []
    current_period, current_speaker = sorted_periods[0]
    for next_period, next_speaker in sorted_periods[1:]:
        if current_period[1] >= next_period[0]:
            if current_speaker == next_speaker:
                current_period = (current_period[0], max(current_period[1], next_period[1]))
            else:
                merged_periods.append((current_period, current_speaker))
                current_period, current_speaker = next_period, next_speaker
        else:
            merged_periods.append((current_period, current_speaker))
            current_period, current_speaker = next_period, next_speaker
    merged_periods.append((current_period, current_speaker))
    return dict(merged_periods)


def get_speaker(time_frame, speaker_dict):
    """Get speaker label for a given timestamp."""
    for (start, end), speaker in speaker_dict.items():
        if start <= time_frame <= end:
            return speaker
    return None


def extract_frames(video_path, output_folder, periods, num_frames=50):
    """Extract frames from video for each speaker period."""
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return
    fps = video.get(cv2.CAP_PROP_FPS)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for (start_time, end_time), speaker in periods.items():
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        speaker_folder = os.path.join(output_folder, speaker)
        if not os.path.exists(speaker_folder):
            os.makedirs(speaker_folder)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = 0
        step = max(1, (end_frame - start_frame) // num_frames)
        for frame_number in range(start_frame, end_frame, step):
            if frame_count >= num_frames:
                break
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = video.read()
            if not success:
                break
            frame_filename = os.path.join(speaker_folder, f"{speaker}_frame_{frame_number}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
    video.release()


def detect_and_crop_faces(image_path, face_cascade):
    """Detect and crop face from image using OpenCV Haar Cascade."""
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return False
    x, y, w, h = faces[0]
    pad = int(w * 0.1)
    x, y = max(0, x - pad), max(0, y - pad)
    w, h = w + 2*pad, h + 2*pad
    face = img[y:y+h, x:x+w]
    cv2.imwrite(image_path, face)
    return True


def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    e1 = np.array(embedding1, dtype=np.float32)
    e2 = np.array(embedding2, dtype=np.float32)
    norm1, norm2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(e1, e2) / (norm1 * norm2))


def extract_and_save_most_common_face(folder_path, threshold=0.5, enforce_detection=False):
    """
    Extract and save the most common face from folder as 'max_image.jpg'.

    Args:
        folder_path: Path to folder with face images
        threshold: Cosine similarity threshold for grouping faces (0.0-1.0)
        enforce_detection: If False, skip images where DeepFace can't detect a face
    """
    face_encodings = []
    face_images = {}

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            file_path = os.path.join(folder_path, filename)
            try:
                # FIXED: Added enforce_detection parameter
                result = DeepFace.represent(
                    img_path=file_path,
                    model_name="ArcFace",
                    detector_backend="opencv",
                    enforce_detection=enforce_detection  # ← Key fix!
                )
                if result and len(result) > 0:
                    embedding = result[0]["embedding"]
                    face_encodings.append(embedding)
                    face_images[tuple(embedding)] = file_path
            except Exception as e:
                if not enforce_detection:
                    print(f"⚠️ Skipping {filename}: {e}")
                    continue
                raise

    if not face_encodings:
        print(f"⚠️ No valid faces found in {folder_path}")
        return None

    # Group faces by similarity
    unique_faces = []
    grouped_faces = {}
    for encoding in face_encodings:
        found_match = False
        for unique_face in unique_faces:
            if cosine_similarity(encoding, unique_face) > threshold:
                found_match = True
                grouped_faces[tuple(unique_face)].append(encoding)
                break
        if not found_match:
            unique_faces.append(encoding)
            grouped_faces[tuple(encoding)] = [encoding]

    if not grouped_faces:
        return None

    # Find most common face group
    most_common_group = max(grouped_faces, key=lambda x: len(grouped_faces[x]))
    most_common_image = face_images.get(most_common_group)

    if not most_common_image:
        return None

    # Save as max_image.jpg
    new_image_path = os.path.join(folder_path, "max_image.jpg")
    shutil.copy2(most_common_image, new_image_path)
    print(f"✅ Most common face saved: {new_image_path}")
    return new_image_path


def get_overlap(range1, range2):
    """Calculate overlap duration between two time ranges."""
    start1, end1 = range1
    start2, end2 = range2
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0.0, overlap_end - overlap_start)
