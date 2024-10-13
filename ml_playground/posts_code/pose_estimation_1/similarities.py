import numpy as np
from scipy.spatial.distance import cosine
import imageio
import tensorflow as tf
import tqdm
from PIL import Image  # Ensure Pillow is installed

# Function to calculate cosine similarity
def cos_sim(a, b):
    x = dot(a, b)/(norm(a)*norm(b))
    return x

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

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
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

def extract_frames(fpath, downsample_factor=1):
    frames = iio.imread(fpath, mode="I")  # 'I' mode for multi-frame images
    downsampled_frames = frames[::downsample_factor]  # Downsample frames
    return downsampled_frames, {"frame_count": len(frames)}



def extract_keypoints_and_crop(frames, metadata):
    # Extract image dimensions from the first frame
    num_frames = len(frames)
    image_height, image_width = frames[0].shape[:2]  # Get height and width from the frame

    # Initialize the crop region
    crop_region = init_crop_region(image_height, image_width)
    detected_keypoints = []

    for frame_idx in tqdm.tqdm(range(num_frames)):
        # Convert the frame to RGB format if needed
        image = np.array(Image.fromarray(frames[frame_idx]).convert("RGB"))

        # Run inference on the current frame
        keypoints_with_scores = run_inference(
            movenet, image, crop_region, crop_size=[input_size, input_size]
        )
        detected_keypoints.append(keypoints_with_scores)

        # Update the crop region based on detected keypoints
        crop_region = determine_crop_region(
            keypoints_with_scores, image_height, image_width
        )

    return detected_keypoints, crop_region



# Main function to load the gifs, extract poses, and calculate similarities
def compare_gifs(sample_gif_path, correct_gif_path, incorrect_gif_path):
    # Extract poses from each gif
    inc_frames_me, metadata_me = extract_frames
    sample_poses = extract_poses(sample_gif_path)
    correct_poses = extract_poses(correct_gif_path)
    incorrect_poses = extract_poses(incorrect_gif_path)

    # Calculate cosine similarity for each body part between Sample-Correct and Sample-Incorrect
    similarities = {}
    for part in sample_poses:
        similarity_correct = calculate_cosine_similarity(sample_poses[part], correct_poses[part])
        similarity_incorrect = calculate_cosine_similarity(sample_poses[part], incorrect_poses[part])
        similarities[part] = {
            "correct_similarity": similarity_correct,
            "incorrect_similarity": similarity_incorrect
        }

    # Find the body part with the highest cosine similarity in the incorrect gif
    highest_similarity_part = max(similarities, key=lambda x: similarities[x]['incorrect_similarity'])
    
    return similarities, highest_similarity_part

def align_warped_kpts(x, path):
    return x[list(map(lambda f: path[f], sorted(path.keys())))]

def frame_sim(a, b):
    num_frames = a.shape[0]
    a = a.reshape(num_frames, -1)
    b = b.reshape(num_frames, -1)
    cos_sim = np.sum(a * b, axis=1)/(norm(a, axis=1) * norm(b, axis=1))
    return cos_sim

# Function to run the comparison and print the results
def main():
    # Provide the correct paths to the gifs
    sample_gif = "sample_form.gif"
    correct_gif = "correct_try.gif"
    incorrect_gif = "wrong_try.gif"

    # Compare the gifs and get the result
    reference_kpts = np.array(detected_keypoints).squeeze()
    correct_kpts = align_warped_kpts(np.array(detected_keypoints_me).squeeze(), path1)
    incorrect_kpts = align_warped_kpts(np.array(detected_keypoints_me_bad).squeeze(), path2)

    cos_sim_correct = cos_sim(reference_kpts[:, :, :2].reshape(-1), correct_kpts[:, :, :2].reshape(-1))
    cos_sim_incorrect = cos_sim(reference_kpts[:, :, :2].reshape(-1), incorrect_kpts[:, :, :2].reshape(-1))
    print(f"Cosine similarity (correct): {cos_sim_correct:.4f}")
    print(f"Cosine similarity (incorrect): {cos_sim_incorrect:.4f}")

    # Output the similarities and the body part with the highest incorrect similarity
    print(f"Similarities for each body part:\n{similarities}")
    print(f"Body part with highest incorrect similarity: {highest_similarity_part} - Similarity: {similarities[highest_similarity_part]['incorrect_similarity']}")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
