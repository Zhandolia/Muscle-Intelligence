#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[77]:


get_ipython().system('pip install imageio[PIL]')
get_ipython().system('pip install --upgrade imageio pillow imageio-ffmpeg')
get_ipython().system('pip install tqdm')


# In[78]:


import tensorflow as tf

# Check for TensorFlow GPU access
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# See TensorFlow version
print(f"TensorFlow version: {tf.__version__}")


# In[79]:


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
import imageio.v3 as iio
from IPython.display import HTML, display, Image


# In[80]:


import os
os.environ["TFHUB_CACHE_DIR"] = "./tfhub_cache"


# In[81]:


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

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (4,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot

import imageio
import numpy as np

def to_gif(images, duration, loop=1):
    """Converts image sequence (4D numpy array) to gif."""
    images = images.astype(np.uint8)  # Ensure correct data type
    duration = duration / 1000  # Convert duration to seconds
    imageio.mimsave('./animation.gif', images, duration=duration, loop=loop)
    return embed.embed_file('./animation.gif')  # Embed the GIF for display


def progress(value, max=100):
  return HTML("""
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value, max=max))


# In[82]:


model_name = "movenet_lightning"

if "tflite" in model_name:
  if "movenet_lightning_f16" in model_name:
    get_ipython().system('wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite')
    input_size = 192
  elif "movenet_thunder_f16" in model_name:
    get_ipython().system('wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite')
    input_size = 256
  elif "movenet_lightning_int8" in model_name:
    get_ipython().system('wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite')
    input_size = 192
  elif "movenet_thunder_int8" in model_name:
    get_ipython().system('wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite')
    input_size = 256
  else:
    raise ValueError("Unsupported model name: %s" % model_name)

  # Initialize the TFLite interpreter
  interpreter = tf.lite.Interpreter(model_path="model.tflite")
  interpreter.allocate_tensors()

  def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

else:
  if "movenet_lightning" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    input_size = 192
  elif "movenet_thunder" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    input_size = 256
  else:
    raise ValueError("Unsupported model name: %s" % model_name)

  def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores


# In[83]:


get_ipython().system('curl -o input_image.jpeg https://images.pexels.com/photos/4384679/pexels-photo-4384679.jpeg --silent')


# In[84]:


# Load the input image.
image_path = 'input_image.jpeg'
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image)


# In[85]:


Image(filename=image_path)


# In[86]:


# Resize and pad the image to keep the aspect ratio and fit the expected size.
input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

# Run model inference.
keypoints_with_scores = movenet(input_image)

# Visualize the predictions with image.
display_image = tf.expand_dims(image, axis=0)
display_image = tf.cast(tf.image.resize_with_pad(
    display_image, 1280, 1280), dtype=tf.int32)
output_overlay = draw_prediction_on_image(
    np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

plt.figure(figsize=(20, 20))
plt.imshow(output_overlay)
_ = plt.axis('off')


# In[87]:


get_ipython().system('wget -q -O dance.gif https://github.com/tensorflow/tfjs-models/raw/master/pose-detection/assets/dance_input.gif')


# In[88]:


# Confidence score to determine whether a keypoint prediction is reliable.
MIN_CROP_KEYPOINT_SCORE = 0.2

def init_crop_region(image_height, image_width):
  """Defines the default crop region.

  The function provides the initial crop region (pads the full image from both
  sides to make it a square image) when the algorithm cannot reliably determine
  the crop region from the previous frame.
  """
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
  """Checks whether there are enough torso keypoints.

  This function checks whether the model is confident at predicting one of the
  shoulders/hips which is required to determine a good crop region.
  """
  return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE) and
          (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE))

def determine_torso_and_body_range(
    keypoints, target_keypoints, center_y, center_x):
  """Calculates the maximum distance from each keypoints to the center location.

  The function returns the maximum distances from the two sets of keypoints:
  full 17 keypoints and 4 torso keypoints. The returned information will be
  used to determine the crop size. See determineCropRegion for more detail.
  """
  torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
  max_torso_yrange = 0.0
  max_torso_xrange = 0.0
  for joint in torso_joints:
    dist_y = abs(center_y - target_keypoints[joint][0])
    dist_x = abs(center_x - target_keypoints[joint][1])
    if dist_y > max_torso_yrange:
      max_torso_yrange = dist_y
    if dist_x > max_torso_xrange:
      max_torso_xrange = dist_x

  max_body_yrange = 0.0
  max_body_xrange = 0.0
  for joint in KEYPOINT_DICT.keys():
    if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
      continue
    dist_y = abs(center_y - target_keypoints[joint][0]);
    dist_x = abs(center_x - target_keypoints[joint][1]);
    if dist_y > max_body_yrange:
      max_body_yrange = dist_y

    if dist_x > max_body_xrange:
      max_body_xrange = dist_x

  return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

def determine_crop_region(
      keypoints, image_height,
      image_width):
  """Determines the region to crop the image for the model to run inference on.

  The algorithm uses the detected joints from the previous frame to estimate
  the square region that encloses the full body of the target person and
  centers at the midpoint of two hip joints. The crop size is determined by
  the distances between each joints and the center point.
  When the model is not confident with the four torso joint predictions, the
  function returns a default crop which is the full image padded to square.
  """
  target_keypoints = {}
  for joint in KEYPOINT_DICT.keys():
    target_keypoints[joint] = [
      keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
      keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
    ]

  if torso_visible(keypoints):
    center_y = (target_keypoints['left_hip'][0] +
                target_keypoints['right_hip'][0]) / 2;
    center_x = (target_keypoints['left_hip'][1] +
                target_keypoints['right_hip'][1]) / 2;

    (max_torso_yrange, max_torso_xrange,
      max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
          keypoints, target_keypoints, center_y, center_x)

    crop_length_half = np.amax(
        [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
          max_body_yrange * 1.2, max_body_xrange * 1.2])

    tmp = np.array(
        [center_x, image_width - center_x, center_y, image_height - center_y])
    crop_length_half = np.amin(
        [crop_length_half, np.amax(tmp)]);

    crop_corner = [center_y - crop_length_half, center_x - crop_length_half];

    if crop_length_half > max(image_width, image_height) / 2:
      return init_crop_region(image_height, image_width)
    else:
      crop_length = crop_length_half * 2;
      return {
        'y_min': crop_corner[0] / image_height,
        'x_min': crop_corner[1] / image_width,
        'y_max': (crop_corner[0] + crop_length) / image_height,
        'x_max': (crop_corner[1] + crop_length) / image_width,
        'height': (crop_corner[0] + crop_length) / image_height -
            crop_corner[0] / image_height,
        'width': (crop_corner[1] + crop_length) / image_width -
            crop_corner[1] / image_width
      }
  else:
    return init_crop_region(image_height, image_width)

def crop_and_resize(image, crop_region, crop_size):
  """Crops and resize the image to prepare for the model input."""
  boxes=[[crop_region['y_min'], crop_region['x_min'],
          crop_region['y_max'], crop_region['x_max']]]
  output_image = tf.image.crop_and_resize(
      image, box_indices=[0], boxes=boxes, crop_size=crop_size)
  return output_image

def run_inference(movenet, image, crop_region, crop_size):
  """Runs model inference on the cropped region.

  The function runs the model inference on the cropped region and updates the
  model output to the original image coordinate system.
  """
  image_height, image_width, _ = image.shape
  input_image = crop_and_resize(
    tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
  # Run model inference.
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


# In[89]:


image_path = 'dance.gif'
image = tf.io.read_file(image_path)
image = tf.image.decode_gif(image)


# In[90]:


Image(filename=image_path)


# In[91]:


# Load the input image.
num_frames, image_height, image_width, _ = image.shape
crop_region = init_crop_region(image_height, image_width)

output_images = []
bar = display(progress(0, num_frames-1), display_id=True)
for frame_idx in range(num_frames):
  keypoints_with_scores = run_inference(
      movenet, image[frame_idx, :, :, :], crop_region,
      crop_size=[input_size, input_size])
  output_images.append(draw_prediction_on_image(
      image[frame_idx, :, :, :].numpy().astype(np.int32),
      keypoints_with_scores, crop_region=None,
      close_figure=True, output_image_height=300))
  crop_region = determine_crop_region(
      keypoints_with_scores, image_height, image_width)
  bar.update(progress(frame_idx, num_frames-1))

# Prepare gif visualization.
output = np.stack(output_images, axis=0)
to_gif(output, duration=100)


# In[92]:


def extract_frames(fpath, downsample_factor=1):
    frames = iio.imread(fpath, mode="I")  # 'I' mode for multi-frame images
    downsampled_frames = frames[::downsample_factor]  # Downsample frames
    return downsampled_frames, {"frame_count": len(frames)}


# In[93]:


frames_target, metadata_target = extract_frames(r'sample_form.gif', 4)


# In[94]:


import tqdm
import numpy as np
from PIL import Image  # Ensure Pillow is installed

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



# In[95]:


detected_keypoints, crop_region = extract_keypoints_and_crop(frames_target, metadata_target)


# In[96]:


from matplotlib import MatplotlibDeprecationWarning
import warnings

def draw_predictions_and_crop(frames, keypoints_with_scores, crop_region, padding=0.02, y_max_padding=0.01, min_crop_ratio=0.85):
    """
    Draw predictions and crop frames with optional padding and a minimum crop ratio.
    This ensures less aggressive cropping.
    """
    # Ensure frames are in RGB format
    frames = np.array([ensure_rgb(frame) for frame in frames])

    # Extract frame dimensions
    image_height, image_width = frames.shape[1:3]

    output_images = []

    # Suppress matplotlib deprecation warnings
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=MatplotlibDeprecationWarning)

        # Draw predictions on each frame
        for frame_idx, frame in enumerate(frames):
            image = frame.astype(np.int32)

            output_image = draw_prediction_on_image(
                image,
                keypoints_with_scores[frame_idx],
                crop_region=None,
                close_figure=True,
                output_image_height=image_height
            )

            output_images.append(output_image)

    # Adjust crop region values with padding but keep the crop size reasonable
    y_min = max(0.0, crop_region.get("y_min", 0.0) - padding)
    y_max = min(1.0, crop_region.get("y_max", 1.0) + y_max_padding)
    x_min = max(0.0, crop_region.get("x_min", 0.0) - padding)
    x_max = min(1.0, crop_region.get("x_max", 1.0) + padding)

    # Ensure crop region retains a minimum size (min_crop_ratio)
    y_min, y_max = enforce_min_crop_size(y_min, y_max, min_crop_ratio)
    x_min, x_max = enforce_min_crop_size(x_min, x_max, min_crop_ratio)

    # Debug: Print clamped crop region values with padding
    print(f"Adjusted Crop Region: y_min={y_min}, y_max={y_max}, x_min={x_min}, x_max={x_max}")

    # Calculate crop coordinates in pixels
    cropped_h1 = round(y_min * image_height)
    cropped_h2 = round(y_max * image_height)
    cropped_w1 = round(x_min * image_width)
    cropped_w2 = round(x_max * image_width)

    # Ensure the crop region has a valid size
    if cropped_h2 <= cropped_h1 or cropped_w2 <= cropped_w1:
        print("Invalid crop region detected. Returning un-cropped frames.")
        return np.array(output_images)

    # Debug: Print final crop coordinates
    print(f"Cropped Coordinates: h1={cropped_h1}, h2={cropped_h2}, w1={cropped_w1}, w2={cropped_w2}")

    # Stack and crop output images with adjusted coordinates
    output = np.stack(output_images, axis=0)[:, cropped_h1:cropped_h2, cropped_w1:cropped_w2, :]

    return output

def enforce_min_crop_size(min_val, max_val, min_ratio):
    """Ensure the crop region retains a minimum size based on min_ratio."""
    crop_size = max_val - min_val
    if crop_size < min_ratio:
        # Adjust min and max to enforce the minimum crop size
        adjustment = (min_ratio - crop_size) / 2
        min_val = max(0.0, min_val - adjustment)
        max_val = min(1.0, max_val + adjustment)
    return min_val, max_val

def ensure_rgb(frame):
    """Ensure that the frame is in RGB format."""
    if len(frame.shape) == 2:  # If grayscale, convert to RGB
        return np.array(Image.fromarray(frame).convert("RGB"))
    return frame  # If already RGB, return as-is


# In[97]:


import imageio.v3 as iio

# Call the function to draw predictions and crop frames
output_frames = draw_predictions_and_crop(frames_target, detected_keypoints, crop_region)


# Save the output as a GIF using imageio.v3
iio.imwrite("output1.gif", output_frames, format="GIF", loop=0)


# In[99]:


frames_me, metadata_me = extract_frames('./correct_try.gif', 2)

# Ensure frames are a NumPy array
frames_me = np.array(frames_me)

# Reverse the color channels (if needed)
# Ensure the frame has at least 3 channels (e.g., RGB) before attempting reversal
if frames_me.shape[-1] == 3:
    frames_me = frames_me[:, :, :, ::-1]  # Reverse color channels

# Extract keypoints and crop region
detected_keypoints_me, crop_region_me = extract_keypoints_and_crop(frames_me, metadata_me)

# Draw predictions and apply cropping
output_gif_me = draw_predictions_and_crop(frames_me, detected_keypoints_me, crop_region_me)

# Save the output as a GIF using imageio.v3
import imageio.v3 as iio
iio.imwrite("output_me.gif", output_gif_me, format="GIF", loop=0)

print("GIF saved as output_me.gif")


# In[100]:


def get_gif_duration(filepath):
    """Extract the duration of a GIF using Pillow."""
    with Image.open(filepath) as gif:
        duration = gif.info.get("duration", 100)  # Default to 100ms per frame if not found
    return duration

def to_gif(frames, duration, loop=0, output_path="output_me.gif"):
    """Save frames as a GIF with duration and loop settings."""
    iio.imwrite(
        output_path,
        frames,
        format="GIF",
        duration=duration / 1000,  # Convert from ms to seconds
        loop=loop
    )
    print(f"GIF saved at {output_path}")

# Extract the GIF duration, falling back to default if missing
gif_duration = metadata_me.get("duration", get_gif_duration('./pushup.gif'))

# Save the GIF using the extracted or default duration
to_gif(output_gif_me, duration=gif_duration, loop=0)


# In[101]:


frames_me, metadata_me = extract_frames('./wrong_try.gif', 2)

# Ensure frames are a NumPy array
frames_me = np.array(frames_me)

# Reverse the color channels (if needed)
# Ensure the frame has at least 3 channels (e.g., RGB) before attempting reversal
if frames_me.shape[-1] == 3:
    frames_me = frames_me[:, :, :, ::-1]  # Reverse color channels

# Extract keypoints and crop region
detected_keypoints_me, crop_region_me = extract_keypoints_and_crop(frames_me, metadata_me)

# Draw predictions and apply cropping
output_gif_me = draw_predictions_and_crop(frames_me, detected_keypoints_me, crop_region_me)

# Save the output as a GIF using imageio.v3
import imageio.v3 as iio
iio.imwrite("output_me.gif", output_gif_me, format="GIF", loop=0)

output_frames = draw_predictions_and_crop(frames_target, detected_keypoints, crop_region)
iio.imwrite("output1.gif", output_frames, format="GIF", loop=0)


# In[102]:


from PIL import ImageOps
from PIL import Image as PilImage

def resize_frames(frames, target_size):
    resized_frames = [
        np.array(Image.fromarray(frame).resize(target_size, Image.Resampling.LANCZOS))
        for frame in frames
    ]
    return np.stack(resized_frames, axis=0)

# Ensure 'output_gif_me' is being used correctly
target_size = output_gif_me.shape[2], output_gif_me.shape[1]  # (width, height)

# Resize the frames to the target size
correct = resize_frames(output_gif_me, target_size)

# Set reference to the correctly processed GIF output
reference = output_gif_me

output_gif_me_bad = output_gif_me if 'output_gif_me_bad' not in locals() else output_gif_me_bad
# Resize the incorrect frames for comparison
incorrect = resize_frames(output_gif_me_bad, target_size)


# In[103]:


# Ensure 'output_gif_me' is being used correctly
target_size = output_gif_me.shape[2], output_gif_me.shape[1]  # Swap dimensions for (width, height)

# Resize the frames to the target size
correct = resize_frames(output_gif_me, target_size)

# Set reference to the correctly processed GIF output
reference = output_gif_me

# Resize the incorrect frames for comparison
incorrect = resize_frames(output_gif_me_bad, target_size)


# In[104]:


get_ipython().system('pip install dtaidistance')
from numpy import linalg as LA
from dtaidistance import dtw_ndim

s1 = np.array(detected_keypoints).squeeze()[:, :, :2].reshape(len(detected_keypoints), -1)
s1 = s1/LA.norm(s1, axis=1)[0]
s2 = np.array(detected_keypoints_me).squeeze()[:, :, :2].reshape(len(detected_keypoints_me), -1)
s2 = s2/LA.norm(s2, axis=1)[0]
correct_rowing_warped_path = dtw_ndim.warping_path(s1, s2)


# In[105]:


from numpy import linalg as LA

detected_keypoints_me_bad = (
    detected_keypoints if 'detected_keypoints_me_bad' not in locals() else detected_keypoints_me_bad
)

s1 = np.array(detected_keypoints).squeeze()[:, :, :2].reshape(len(detected_keypoints), -1)
s1 = s1/LA.norm(s1, axis=1)[0]
s2 = np.array(detected_keypoints_me_bad).squeeze()[:, :, :2].reshape(len(detected_keypoints_me_bad), -1)
s2 = s2/LA.norm(s2, axis=1)[0]
incorrect_rowing_warped_path = dtw_ndim.warping_path(s1, s2)


# In[106]:


correct_rowing_warped_path = {k:v for (k, v) in correct_rowing_warped_path}
incorrect_rowing_warped_path = {k:v for (k, v) in incorrect_rowing_warped_path}


# In[107]:


from numpy import dot
from numpy.linalg import norm

def cos_sim(a, b):
    x = dot(a, b)/(norm(a)*norm(b))
    return x


# In[108]:


from PIL import Image
import numpy as np

g1 = correct
g2 = reference
g3 = incorrect
path1 = correct_rowing_warped_path
path2 = incorrect_rowing_warped_path

concat_gif_frames = []

for frame_idx in range(len(g2)):
    # Validate indices before accessing
    if frame_idx >= len(path1) or frame_idx >= len(path2):
        print(f"Skipping frame {frame_idx}: Index out of bounds.")
        continue

    # Concatenate frames along the width (axis=1)
    concat_frame = np.concatenate([
        g1[path1[frame_idx], :, :, :],
        g2[frame_idx, :, :, :],
        g3[path2[frame_idx], :, :, :]
    ], axis=1)

    # Extract keypoints and compute cosine similarity
    base = np.array(detected_keypoints).squeeze()[frame_idx, :, :2].reshape(-1)
    s1 = np.array(detected_keypoints_me).squeeze()[path1[frame_idx], :, :2].reshape(-1)
    s2 = np.array(detected_keypoints_me_bad).squeeze()[path2[frame_idx], :, :2].reshape(-1)
    
    cos_sim1 = cos_sim(base, s1)
    cos_sim2 = cos_sim(base, s2)

    # Create a PIL image from the concatenated frame and append to the list
    frame_img = Image.fromarray(concat_frame)
    concat_gif_frames.append(np.array(frame_img))

# Save the GIF using the to_gif function
to_gif(np.stack(concat_gif_frames, axis=0), duration=metadata_target.get("duration", 100), loop=0)


# In[109]:


from PIL import Image, ImageDraw

concat_gif_frames = []

for frame_idx in range(len(g2)):
    # Ensure the index is within bounds for both paths
    if frame_idx >= len(path1) or frame_idx >= len(path2):
        print(f"Skipping frame {frame_idx}: Index out of bounds.")
        continue

    # Concatenate frames along the width (axis=1)
    concat_frame = np.concatenate([
        g1[path1[frame_idx], :, :, :],
        g2[frame_idx, :, :, :],
        g3[path2[frame_idx], :, :, :]
    ], axis=1)

    # Extract keypoints and compute cosine similarity
    base = np.array(detected_keypoints).squeeze()[frame_idx, :, :2].reshape(-1)
    s1 = np.array(detected_keypoints_me).squeeze()[path1[frame_idx], :, :2].reshape(-1)
    s2 = np.array(detected_keypoints_me_bad).squeeze()[path2[frame_idx], :, :2].reshape(-1)

    cos_sim1 = cos_sim(base, s1)
    cos_sim2 = cos_sim(base, s2)

    # Create a PIL image from the concatenated frame
    frame_img = Image.fromarray(concat_frame)
    draw = ImageDraw.Draw(frame_img)

    # Add text with the cosine similarity scores
    draw.text((115, 300), f"SCORE: {cos_sim1:.3f}", fill="red", 
              stroke_width=2, stroke_fill="white")
    draw.text((850, 300), f"SCORE: {cos_sim2:.3f}", fill="red", 
              stroke_width=2, stroke_fill="white")

    concat_gif_frames.append(np.array(frame_img))

# Save the GIF using the to_gif function
to_gif(np.stack(concat_gif_frames, axis=0), duration=metadata_target.get("duration", 100), loop=0)


# In[110]:


reference_kpts = np.array(detected_keypoints).squeeze()


# In[111]:


def align_warped_kpts(x, path):
    return x[list(map(lambda f: path[f], sorted(path.keys())))]


# In[112]:


correct_kpts = align_warped_kpts(np.array(detected_keypoints_me).squeeze(), path1)
incorrect_kpts = align_warped_kpts(np.array(detected_keypoints_me_bad).squeeze(), path2)


# In[113]:


### basic similarity - comparing whole vectors
def cos_sim(a, b):
    x = dot(a, b)/(norm(a)*norm(b))
    return x

cos_sim_correct = cos_sim(reference_kpts[:, :, :2].reshape(-1), correct_kpts[:, :, :2].reshape(-1))
cos_sim_incorrect = cos_sim(reference_kpts[:, :, :2].reshape(-1), incorrect_kpts[:, :, :2].reshape(-1))
print(f"Cosine similarity (correct): {cos_sim_correct:.4f}")
print(f"Cosine similarity (incorrect): {cos_sim_incorrect:.4f}")


# In[114]:


def frame_sim(a, b):
    num_frames = a.shape[0]
    a = a.reshape(num_frames, -1)
    b = b.reshape(num_frames, -1)
    cos_sim = np.sum(a * b, axis=1)/(norm(a, axis=1) * norm(b, axis=1))
    return cos_sim

frame_sim_correct = frame_sim(reference_kpts[:, :, :2], correct_kpts[:, :, :2])
frame_sim_incorrect = frame_sim(reference_kpts[:, :, :2], incorrect_kpts[:, :, :2])
print(f"Mean frame by frame cos sim (correct): {np.mean(frame_sim_correct):.4f}")
print(f"Mean frame by frame cos sim (incorrect): {np.mean(frame_sim_incorrect):.4f}")
print(f"Median frame by frame cos sim (correct): {np.median(frame_sim_correct):.4f}")
print(f"Median frame by frame cos sim (incorrect): {np.median(frame_sim_incorrect):.4f}")


# In[115]:


fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(range(len(frame_sim_correct)), frame_sim_correct, label="Correct")
ax.plot(range(len(frame_sim_incorrect)), frame_sim_incorrect, label="Incorrect")

ax.legend()
ax.grid()
ax.set_title("Mean cos sim")
_ = plt.xlabel("Frame")
_ = plt.ylabel("Cos sim")


# In[116]:


### diving deeper into individual keypoints
reference_kpts_scores = reference_kpts[: , :, 2].T
fig, ax = plt.subplots(figsize=(20, 10))
im = ax.imshow(reference_kpts_scores, cmap="RdYlGn")
kpts = list(sorted(KEYPOINT_DICT.items(), key=lambda x: x[1]))
num_kpts = len(kpts)
num_frames = reference_kpts_scores.shape[1]
_ = ax.set_yticks(np.arange(num_kpts), KEYPOINT_DICT.keys())
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75)
cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")

# Loop over data dimensions and create text annotations.
for i in range(num_kpts):
    for j in range(num_frames):
        text = ax.text(j, i, f"{reference_kpts_scores[i, j]:.2f}",
                        ha="center", va="center", color="black")

ax.set_title("Reference recording")
_ = plt.xlabel("Frame")
fig.tight_layout()


# In[117]:


### diving deeper into individual keypoints
correct_kpts_scores = correct_kpts[: , :, 2].T
fig, ax = plt.subplots(figsize=(20, 10))
im = ax.imshow(correct_kpts_scores, cmap="RdYlGn")
kpts = list(sorted(KEYPOINT_DICT.items(), key=lambda x: x[1]))
num_kpts = len(kpts)
num_frames = correct_kpts_scores.shape[1]
_ = ax.set_yticks(np.arange(num_kpts), KEYPOINT_DICT.keys())
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")

# Loop over data dimensions and create text annotations.
for i in range(num_kpts):
    for j in range(num_frames):
        text = ax.text(j, i, f"{correct_kpts_scores[i, j]:.2f}",
                       ha="center", va="center", color="black")

ax.set_title("My recording (correct)")

_ = plt.xlabel("Frame")
fig.tight_layout()


# In[118]:


### diving deeper into individual keypoints
ref_kpts_t = reference_kpts.transpose([1, 0, 2])
correct_kpts_t = correct_kpts.transpose([1, 0 ,2])
incorrect_kpts_t = incorrect_kpts.transpose([1, 0 ,2])
num_kpts = len(KEYPOINT_DICT)

cos_sim_correct = np.sum(ref_kpts_t[:, :, :2] * correct_kpts_t[:, :, :2], axis=2)/(
    norm(ref_kpts_t[:, :, :2], axis=2) * norm(correct_kpts_t[:, :, :2], axis=2))
cos_sim_incorrect = np.sum(ref_kpts_t[:, :, :2] * incorrect_kpts_t[:, :, :2], axis=2)/(
    norm(ref_kpts_t[:, :, :2], axis=2) * norm(incorrect_kpts_t[:, :, :2], axis=2))
ref_kpts_scores = ref_kpts_t[:, :, 2]
correct_kpts_scores = correct_kpts_t[:, :, 2]
incorrect_kpts_scores = incorrect_kpts_t[:, :, 2]

weights = (ref_kpts_scores + correct_kpts_scores)/2
weights2 = (ref_kpts_scores + incorrect_kpts_scores)/2

weighted_correct_mean = np.mean(np.sum(cos_sim_correct * weights, axis=1)/np.sum(weights, axis=1))
weighted_incorrect_mean = np.mean(np.sum(cos_sim_incorrect * weights2, axis=1)/np.sum(weights2, axis=1))
print(f"Mean weighted cos sim (correct): {weighted_correct_mean:.4f}")
print(f"Mean weighted cos sim (incorrect): {weighted_incorrect_mean:.4f}")


# In[ ]:





# In[119]:


from PIL import Image, ImageDraw

concat_gif_frames = []

for frame_idx in range(len(g2)):
    # Ensure the index is within bounds for both paths
    if frame_idx >= len(path1) or frame_idx >= len(path2):
        print(f"Skipping frame {frame_idx}: Index out of bounds.")
        continue

    # Concatenate frames along the width (axis=1)
    concat_frame = np.concatenate([
        g1[path1[frame_idx], :, :, :],
        g2[frame_idx, :, :, :],
        g3[path2[frame_idx], :, :, :]
    ], axis=1)

    # Extract keypoints and compute cosine similarity
    base = np.array(detected_keypoints).squeeze()[frame_idx, :, :2].reshape(-1)
    s1 = np.array(detected_keypoints_me).squeeze()[path1[frame_idx], :, :2].reshape(-1)
    s2 = np.array(detected_keypoints_me_bad).squeeze()[path2[frame_idx], :, :2].reshape(-1)

    cos_sim1 = cos_sim(base, s1)
    cos_sim2 = cos_sim(base, s2)

    # Create a PIL image from the concatenated frame
    frame_img = Image.fromarray(concat_frame)
    draw = ImageDraw.Draw(frame_img)

    # Add text with the cosine similarity scores
    draw.text((115, 300), f"SCORE: {cos_sim1:.3f}", fill="red", 
              stroke_width=2, stroke_fill="white")
    draw.text((850, 300), f"SCORE: {cos_sim2:.3f}", fill="red", 
              stroke_width=2, stroke_fill="white")

    concat_gif_frames.append(np.array(frame_img))

# Save the GIF using the to_gif function
to_gif(np.stack(concat_gif_frames, axis=0), duration=metadata_target.get("duration", 100), loop=0)


# In[ ]:




