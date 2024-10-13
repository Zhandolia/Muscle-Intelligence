#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import imageio
from PIL import Image
from numpy.linalg import norm
from numpy import dot
from dtaidistance import dtw_ndim
import tqdm

# Load the MoveNet model
module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
input_size = 192

def movenet(input_image):
    """Runs detection on an input image."""
    model = module.signatures['serving_default']
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = model(input_image)
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Confidence score to determine whether a keypoint prediction is reliable.
MIN_CROP_KEYPOINT_SCORE = 0.2

def init_crop_region(image_height, image_width):
    """Defines the default crop region."""
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

    return {
        'y_min': y_min,
        'x_min': x_min,
        'y_max': y_min + box_height,
        'x_max': x_min + box_width,
        'height': box_height,
        'width': box_width
    }

def torso_visible(keypoints):
    """Checks whether there are enough torso keypoints."""
    return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] > MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] > MIN_CROP_KEYPOINT_SCORE) and
            (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE))

def determine_torso_and_body_range(keypoints, target_keypoints, center_y, center_x):
    """Calculates the maximum distance from keypoints to the center."""
    torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint in torso_joints:
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        max_torso_yrange = max(dist_y, max_torso_yrange)
        max_torso_xrange = max(dist_x, max_torso_xrange)

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for joint in KEYPOINT_DICT.keys():
        if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
            continue
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        max_body_yrange = max(dist_y, max_body_yrange)
        max_body_xrange = max(dist_x, max_body_xrange)

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

def determine_crop_region(keypoints, image_height, image_width):
    """Determines the region to crop the image for inference."""
    target_keypoints = {}
    for joint in KEYPOINT_DICT.keys():
        target_keypoints[joint] = [
            keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
            keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
        ]

    if torso_visible(keypoints):
        center_y = (target_keypoints['left_hip'][0] +
                    target_keypoints['right_hip'][0]) / 2
        center_x = (target_keypoints['left_hip'][1] +
                    target_keypoints['right_hip'][1]) / 2

        (max_torso_yrange, max_torso_xrange,
         max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
            keypoints, target_keypoints, center_y, center_x)

        crop_length_half = np.amax(
            [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
             max_body_yrange * 1.2, max_body_xrange * 1.2])

        tmp = np.array(
            [center_x, image_width - center_x, center_y, image_height - center_y])
        crop_length_half = np.amin(
            [crop_length_half, np.amax(tmp)])

        crop_corner = [center_y - crop_length_half, center_x - crop_length_half]

        if crop_length_half > max(image_width, image_height) / 2:
            return init_crop_region(image_height, image_width)
        else:
            crop_length = crop_length_half * 2
            return {
                'y_min': crop_corner[0] / image_height,
                'x_min': crop_corner[1] / image_width,
                'y_max': (crop_corner[0] + crop_length) / image_height,
                'x_max': (crop_corner[1] + crop_length) / image_width,
                'height': crop_length / image_height,
                'width': crop_length / image_width
            }
    else:
        return init_crop_region(image_height, image_width)

def crop_and_resize(image, crop_region, crop_size):
    """Crops and resizes the image for model input."""
    boxes = [[crop_region['y_min'], crop_region['x_min'],
              crop_region['y_max'], crop_region['x_max']]]
    output_image = tf.image.crop_and_resize(
        image, box_indices=[0], boxes=boxes, crop_size=crop_size)
    return output_image

def run_inference(movenet, image, crop_region, crop_size):
    """Runs model inference on the cropped region."""
    image_height, image_width, _ = image.shape
    input_image = crop_and_resize(
        tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
    keypoints_with_scores = movenet(input_image)
    # Update the coordinates.
    for idx in range(17):
        keypoints_with_scores[0, 0, idx, 0] = (
            crop_region['y_min'] * image_height +
            crop_region['height'] * image_height *
            keypoints_with_scores[0, 0, idx, 0]) / image_height
        keypoints_with_scores[0, 0, idx, 1] = (
            crop_region['x_min'] * image_width +
            crop_region['width'] * image_width *
            keypoints_with_scores[0, 0, idx, 1]) / image_width
    return keypoints_with_scores

def extract_frames(fpath, downsample_factor=1):
    """Extracts frames from a GIF."""
    frames = imageio.mimread(fpath)
    downsampled_frames = frames[::downsample_factor]
    return downsampled_frames, {"frame_count": len(frames)}

def extract_keypoints_and_crop(frames):
    """Extracts keypoints and adjusts crop regions for each frame."""
    num_frames = len(frames)
    image_height, image_width = frames[0].shape[:2]
    crop_region = init_crop_region(image_height, image_width)
    detected_keypoints = []

    for frame_idx in tqdm.tqdm(range(num_frames)):
        image = np.array(Image.fromarray(frames[frame_idx]).convert("RGB"))
        keypoints_with_scores = run_inference(
            movenet, image, crop_region, crop_size=[input_size, input_size]
        )
        detected_keypoints.append(keypoints_with_scores)
        crop_region = determine_crop_region(
            keypoints_with_scores, image_height, image_width
        )

    return detected_keypoints

def align_sequences(seq1, seq2, path):
    """Aligns two sequences based on DTW path."""
    aligned_seq1 = []
    aligned_seq2 = []
    for i, j in path:
        aligned_seq1.append(seq1[i])
        aligned_seq2.append(seq2[j])
    return np.array(aligned_seq1), np.array(aligned_seq2)

def center_and_normalize_keypoints(keypoints):
    """Centers and normalizes keypoints for scale and position invariance."""
    centered_keypoints = []
    for frame in keypoints:
        keypoint_coords = frame[:, :2]
        keypoint_scores = frame[:, 2]
        # Use the midpoint between hips as the center
        left_hip = keypoint_coords[KEYPOINT_DICT['left_hip']]
        right_hip = keypoint_coords[KEYPOINT_DICT['right_hip']]
        center = (left_hip + right_hip) / 2
        # Center the keypoints
        centered_coords = keypoint_coords - center
        # Compute scale (distance between shoulders)
        left_shoulder = keypoint_coords[KEYPOINT_DICT['left_shoulder']]
        right_shoulder = keypoint_coords[KEYPOINT_DICT['right_shoulder']]
        scale = np.linalg.norm(left_shoulder - right_shoulder)
        if scale == 0:
            scale = 1  # Avoid division by zero
        # Normalize keypoints
        normalized_coords = centered_coords / scale
        # Combine normalized coordinates with scores
        normalized_frame = np.hstack((normalized_coords, keypoint_scores[:, None]))
        centered_keypoints.append(normalized_frame)
    return np.array(centered_keypoints)

def compute_cosine_similarity(reference_kpts, target_kpts):
    """Computes the overall cosine similarity between two keypoint sequences."""
    reference_kpts_flat = reference_kpts[:, :, :2].reshape(len(reference_kpts), -1)
    target_kpts_flat = target_kpts[:, :, :2].reshape(len(target_kpts), -1)
    # Compute cosine similarity for each frame
    cos_sim_frame = np.sum(reference_kpts_flat * target_kpts_flat, axis=1) / (
        norm(reference_kpts_flat, axis=1) * norm(target_kpts_flat, axis=1))
    # Handle NaNs resulting from division by zero
    cos_sim_frame = np.nan_to_num(cos_sim_frame)
    # Compute overall cosine similarity
    overall_cos_sim = np.mean(cos_sim_frame)
    return overall_cos_sim

def compute_keypoint_similarity(reference_kpts, target_kpts):
    """Computes mean cosine similarity per keypoint across frames."""
    ref_kpts_t = reference_kpts.transpose([1, 0, 2])
    target_kpts_t = target_kpts.transpose([1, 0, 2])
    cos_sim_per_kpt = np.sum(ref_kpts_t[:, :, :2] * target_kpts_t[:, :, :2], axis=2) / (
        norm(ref_kpts_t[:, :, :2], axis=2) * norm(target_kpts_t[:, :, :2], axis=2))
    cos_sim_per_kpt = np.nan_to_num(cos_sim_per_kpt)
    mean_cos_sim_per_kpt = np.mean(cos_sim_per_kpt, axis=1)
    return mean_cos_sim_per_kpt

# Process sample GIF
frames_sample, _ = extract_frames('sample_form.gif', downsample_factor=2)

# Process incorrect try GIF
frames_incorrect, _ = extract_frames('wrong_try.gif', downsample_factor=2)

# Process correct try GIF
frames_correct, _ = extract_frames('correct_try.gif', downsample_factor=2)

# Extract keypoints from sample GIF
print("Processing sample GIF...")
detected_keypoints_sample = extract_keypoints_and_crop(frames_sample)

# Extract keypoints from incorrect try GIF
print("Processing incorrect try GIF...")
detected_keypoints_incorrect = extract_keypoints_and_crop(frames_incorrect)

# Extract keypoints from correct try GIF
print("Processing correct try GIF...")
detected_keypoints_correct = extract_keypoints_and_crop(frames_correct)

# Prepare keypoint arrays
reference_kpts = np.array(detected_keypoints_sample).squeeze()
incorrect_kpts = np.array(detected_keypoints_incorrect).squeeze()
correct_kpts = np.array(detected_keypoints_correct).squeeze()

# Center and normalize keypoints
reference_kpts_norm = center_and_normalize_keypoints(reference_kpts)
incorrect_kpts_norm = center_and_normalize_keypoints(incorrect_kpts)
correct_kpts_norm = center_and_normalize_keypoints(correct_kpts)

# Compute the DTW warping path for incorrect try
print("Computing DTW warping path for incorrect try...")
s_sample = reference_kpts_norm[:, :, :2].reshape(len(reference_kpts_norm), -1)
s_incorrect = incorrect_kpts_norm[:, :, :2].reshape(len(incorrect_kpts_norm), -1)
warped_path_incorrect = dtw_ndim.warping_path(s_sample, s_incorrect)

# Compute the DTW warping path for correct try
print("Computing DTW warping path for correct try...")
s_correct = correct_kpts_norm[:, :, :2].reshape(len(correct_kpts_norm), -1)
warped_path_correct = dtw_ndim.warping_path(s_sample, s_correct)

# Align keypoints using the warping paths
aligned_reference_kpts_incorrect, aligned_incorrect_kpts = align_sequences(reference_kpts_norm, incorrect_kpts_norm, warped_path_incorrect)
aligned_reference_kpts_correct, aligned_correct_kpts = align_sequences(reference_kpts_norm, correct_kpts_norm, warped_path_correct)

# Compute overall cosine similarity between sample and incorrect try
cos_sim_incorrect = compute_cosine_similarity(aligned_reference_kpts_incorrect, aligned_incorrect_kpts)
print(f"\n1) Overall cosine similarity between sample and wrong try: {cos_sim_incorrect:.4f}")

# Compute overall cosine similarity between sample and correct try
cos_sim_correct = compute_cosine_similarity(aligned_reference_kpts_correct, aligned_correct_kpts)
print(f"2) Overall cosine similarity between sample and correct try: {cos_sim_correct:.4f}")

# Compute per-keypoint cosine similarities for incorrect try
mean_cos_sim_per_keypoint_incorrect = compute_keypoint_similarity(aligned_reference_kpts_incorrect, aligned_incorrect_kpts)

# Get keypoint names
keypoint_names = [k for k, v in sorted(KEYPOINT_DICT.items(), key=lambda item: item[1])]

# Get the indices of keypoints with lowest similarity (most different)
sorted_indices_dissimilar = np.argsort(mean_cos_sim_per_keypoint_incorrect)

# Output the keypoints in the incorrect GIF that are most different from the sample
print("\n3) Keypoints in the wrong try that are most different from the sample:")
for idx in sorted_indices_dissimilar[:5]:  # Adjust the number to list more or fewer keypoints
    print(f"Keypoint: {keypoint_names[idx]}, Mean Cosine Similarity: {mean_cos_sim_per_keypoint_incorrect[idx]:.4f}")
