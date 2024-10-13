# Import necessary libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import tensorflow as tf
import tensorflow_hub as hub
import imageio
from tqdm import tqdm
from numpy.linalg import norm
from dtaidistance import dtw_ndim
import cv2
import io  # Added import for in-memory buffer

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

# Dictionary mapping joint names to keypoint indices
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

# Edges connecting the keypoints for visualization
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# Confidence score to determine whether a keypoint prediction is reliable.
MIN_CROP_KEYPOINT_SCORE = 0.2

# All the other functions remain the same...
# init_crop_region, torso_visible, determine_torso_and_body_range,
# determine_crop_region, crop_and_resize, run_inference,
# extract_frames, extract_keypoints_and_crop, align_sequences,
# center_and_normalize_keypoints, compute_cosine_similarity,
# compute_keypoint_similarity, _keypoints_and_edges_for_display

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

    for frame_idx in tqdm(range(num_frames)):
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

# Function to prepare keypoints and edges for display
def _keypoints_and_edges_for_display(keypoints_with_scores, height, width):
    """Returns coordinates and edges for visualization."""
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []

    kpts_x = keypoints_with_scores[0, 0, :, 1] * width
    kpts_y = keypoints_with_scores[0, 0, :, 0] * height
    keypoints = np.stack([kpts_x, kpts_y], axis=-1)
    keypoints_all = keypoints

    for edge_pair, color in EDGES.items():
        if keypoints_with_scores[0, 0, edge_pair[0], 2] > MIN_CROP_KEYPOINT_SCORE and \
           keypoints_with_scores[0, 0, edge_pair[1], 2] > MIN_CROP_KEYPOINT_SCORE:
            x_start = keypoints[edge_pair[0], 0]
            y_start = keypoints[edge_pair[0], 1]
            x_end = keypoints[edge_pair[1], 0]
            y_end = keypoints[edge_pair[1], 1]
            keypoint_edges_all.append(np.array([[x_start, y_start], [x_end, y_end]]))
            edge_colors.append(color)

    if keypoint_edges_all:
        edges = np.stack(keypoint_edges_all, axis=0)
    else:
        edges = np.array([])

    return keypoints_all, edges, edge_colors

# Corrected draw_prediction_on_image function
def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None, keypoints_to_mark=None):
    """Draws the keypoint predictions on image, with optional keypoints to mark."""
    height, width, channel = image.shape

    # Create a figure with no borders or axes
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.imshow(image)

    # Prepare keypoints and edges for display
    (keypoint_locs, keypoint_edges, edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

    # Set colors for keypoints
    if keypoint_locs.shape[0]:
        keypoint_colors = []
        for idx in range(len(keypoint_locs)):
            if keypoints_to_mark is not None and idx in keypoints_to_mark:
                keypoint_colors.append('red')
            else:
                keypoint_colors.append('green')  # Default color
        scat = ax.scatter(keypoint_locs[:, 0], keypoint_locs[:, 1], s=60, color=keypoint_colors, zorder=3)

    # Plot edges
    if keypoint_edges.shape[0]:
        line_segments = LineCollection(keypoint_edges, linewidths=(4), linestyle='solid')
        line_segments.set_color(edge_colors)
        ax.add_collection(line_segments)

    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Read the image from the buffer
    image_from_plot = np.array(Image.open(buf))

    plt.close(fig)

    # If the image has an alpha channel, remove it
    if image_from_plot.shape[2] == 4:
        image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGBA2RGB)

    # Resize if necessary
    if output_image_height is not None and output_image_height != image_from_plot.shape[0]:
        output_image_width = int(output_image_height / image_from_plot.shape[0] * image_from_plot.shape[1])
        image_from_plot = cv2.resize(
            image_from_plot, dsize=(output_image_width, output_image_height),
            interpolation=cv2.INTER_CUBIC)
    return image_from_plot

# Function to compute per-frame, per-keypoint cosine similarities
def compute_per_frame_keypoint_similarity(ref_kpts, target_kpts):
    num_frames = ref_kpts.shape[0]
    per_frame_similarities = []
    for frame_idx in range(num_frames):
        ref_frame = ref_kpts[frame_idx, :, :2]
        target_frame = target_kpts[frame_idx, :, :2]
        similarities = np.sum(ref_frame * target_frame, axis=1) / (
            norm(ref_frame, axis=1) * norm(target_frame, axis=1) + 1e-6)
        per_frame_similarities.append(similarities)
    return np.array(per_frame_similarities)

# Now, process the sample, correct try, and wrong try GIFs
frames_sample, _ = extract_frames('sample_form.gif', downsample_factor=2)
frames_incorrect, _ = extract_frames('wrong_try.gif', downsample_factor=2)
frames_correct, _ = extract_frames('correct_try.gif', downsample_factor=2)

print("Processing sample GIF...")
detected_keypoints_sample = extract_keypoints_and_crop(frames_sample)

print("Processing incorrect try GIF...")
detected_keypoints_incorrect = extract_keypoints_and_crop(frames_incorrect)

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

# Compute the DTW warping paths
print("Computing DTW warping path for incorrect try...")
s_sample = reference_kpts_norm[:, :, :2].reshape(len(reference_kpts_norm), -1)
s_incorrect = incorrect_kpts_norm[:, :, :2].reshape(len(incorrect_kpts_norm), -1)
warped_path_incorrect = dtw_ndim.warping_path(s_sample, s_incorrect)

print("Computing DTW warping path for correct try...")
s_correct = correct_kpts_norm[:, :, :2].reshape(len(correct_kpts_norm), -1)
warped_path_correct = dtw_ndim.warping_path(s_sample, s_correct)

# Align keypoints using the warping paths
aligned_reference_kpts_incorrect, aligned_incorrect_kpts = align_sequences(
    reference_kpts_norm, incorrect_kpts_norm, warped_path_incorrect)
aligned_reference_kpts_correct, aligned_correct_kpts = align_sequences(
    reference_kpts_norm, correct_kpts_norm, warped_path_correct)

# Compute per-frame, per-keypoint similarities for the incorrect try
per_frame_keypoint_similarities_incorrect = compute_per_frame_keypoint_similarity(
    aligned_reference_kpts_incorrect, aligned_incorrect_kpts)

# Now, generate the wrong try frames with keypoints marked
wrong_frames = []
num_frames = aligned_reference_kpts_incorrect.shape[0]

for frame_idx in range(num_frames):
    # Get the keypoints with scores for this frame
    wrong_frame_kpts_with_scores = detected_keypoints_incorrect[warped_path_incorrect[frame_idx][1]]
    # Get the similarities for this frame
    similarities = per_frame_keypoint_similarities_incorrect[frame_idx]
    # Identify keypoints with similarity less than 0.9
    keypoints_to_mark = [idx for idx, sim in enumerate(similarities) if sim < 0.9]
    # Draw the frame
    image = frames_incorrect[warped_path_incorrect[frame_idx][1]]
    image = np.array(Image.fromarray(image).convert("RGB"))
    output_image = draw_prediction_on_image(
        image.astype(np.uint8),
        wrong_frame_kpts_with_scores,
        crop_region=None,
        close_figure=True,
        output_image_height=image.shape[0],
        keypoints_to_mark=keypoints_to_mark)
    wrong_frames.append(output_image)

# Save the wrong try GIF with keypoints marked
output_wrong_try_gif = np.stack(wrong_frames, axis=0)
imageio.mimsave('wrong_try_marked.gif', output_wrong_try_gif, format='GIF', duration=100, loop=0)
print("Saved wrong_try_marked.gif with incorrect keypoints highlighted.")

# Optionally, do the same for the correct try
per_frame_keypoint_similarities_correct = compute_per_frame_keypoint_similarity(
    aligned_reference_kpts_correct, aligned_correct_kpts)

correct_frames = []
num_frames_correct = aligned_reference_kpts_correct.shape[0]

for frame_idx in range(num_frames_correct):
    correct_frame_kpts_with_scores = detected_keypoints_correct[warped_path_correct[frame_idx][1]]
    similarities = per_frame_keypoint_similarities_correct[frame_idx]
    keypoints_to_mark = [idx for idx, sim in enumerate(similarities) if sim < 0.9]
    image = frames_correct[warped_path_correct[frame_idx][1]]
    image = np.array(Image.fromarray(image).convert("RGB"))
    output_image = draw_prediction_on_image(
        image.astype(np.uint8),
        correct_frame_kpts_with_scores,
        crop_region=None,
        close_figure=True,
        output_image_height=image.shape[0],
        keypoints_to_mark=keypoints_to_mark)
    correct_frames.append(output_image)

# Save the correct try GIF with keypoints marked
output_correct_try_gif = np.stack(correct_frames, axis=0)
imageio.mimsave('correct_try_marked.gif', output_correct_try_gif, format='GIF', duration=100, loop=0)
print("Saved correct_try_marked.gif with incorrect keypoints highlighted.")
