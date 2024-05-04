from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display
from abc import ABC, abstractmethod
from functools import wraps
import sys
import io
import os
import hashlib
from IPython.display import HTML
import pickle

def cv2_put_text(image, text, position=(10, 10)):
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    font_scale = 0.5  # Font scale (size)
    color = (0, 255, 0)  # Color in BGR (blue, green, red)
    thickness = 2  # Thickness of the lines used to draw the text
    cv2.putText(image, text, position, font, font_scale, color, thickness)

def load_from_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
def save_to_pickle(data, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
    except Exception as e:
        print(f"An error occurred while saving to pickle: {e}")

def save_or_load_variable(var_name, pickles_loc):
    file_path = f'{pickles_loc}/{var_name}.pkl'
    var_value = None
    if var_name in globals():
        var_value = globals()[var_name];
        save_to_pickle(var_value, file_path)
    else :
        var_value = load_from_pickle(file_path)
        globals()[var_name] = var_value
    return var_value

def play_video(video_path, title="Video Title", autoplay=True):
    video_html = f"""
    <div>
        <h4>{title}</h4>
        <video width="640" controls {'autoplay' if autoplay else ''}>
            <source src="{video_path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    """
    return HTML(video_html)


def calculate_md5(filename, chunk_size=4096):
    """Calculate the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
    except FileNotFoundError:
        print("File not found.")
        return None
    except IOError:
        print("Error reading file.")
        return None

    return hash_md5.hexdigest()


def record_file(filename):
    md5_filename = f'{filename}.md5'
    checksum = calculate_md5(filename)
    try:
        with open(md5_filename, "w") as file:
            file.write(checksum + '\n')
    except IOError:
        print("Error writing to file.")
        
def is_file_recorded(filename):
    if not os.path.exists(filename):
        return False
    
    md5_filename = f'{filename}.md5'
    md5_val = calculate_md5(filename)
    try:
        with open(md5_filename, "r") as file:
            checksum = file.readline().strip()
            return checksum == md5_val
    except FileNotFoundError:
        print("MD5 file not found.")
        return False
    except IOError:
        print("Error reading MD5 file.")
        return False


def find_key(d, value):
    """
    Searches for the first key in the dictionary whose value matches the provided 'value'.
    
    Parameters:
    - d (dict): The dictionary to search through.
    - value: The value to find the corresponding key for.

    Returns:
    - The key corresponding to the first match of 'value' in the dictionary. Returns None if no match is found.

    Example:
    >>> find_key({'a': 1, 'b': 2, 'c': 1}, 1)
    'a'
    >>> find_key({'a': 1, 'b': 2, 'c': 1}, 3)
    None
    """
    return next((k for k, v in d.items() if v == value), None)


def flat_map(scenes, key_getter):
    """
    Applies a function to each element in a list and flattens the result into a single list.
    
    Parameters:
    - scenes (list): A list of elements to be processed.
    - key_getter (function): A function applied to each element in 'scenes' that returns a list (or iterable).

    Returns:
    - list: A single, flattened list containing all the elements obtained by applying 'key_getter' to each element in 'scenes'.

    Example:
    >>> flat_map([{'frames': [1, 2]}, {'frames': [3, 4]}], lambda x: x['frames'])
    [1, 2, 3, 4]
    """
    return sum([key_getter(scene) for scene in scenes], [])


def flat_screen_frames(app_scenes, key_getter):
    """
    Flattens a list of frames from each scene object in 'app_scenes' using a specified key-getter function.

    Parameters:
    - app_scenes (list): A list of scene objects, each potentially containing multiple frames.
    - key_getter (function): A function that extracts frame data from a scene object.

    Returns:
    - list: A flattened list of frame data extracted from each scene object.

    Example:
    >>> flat_screen_frames([{'frames': [1, 2]}, {'frames': [3, 4]}], lambda x: x['frames'])
    [1, 2, 3, 4]
    """
    return [key_getter(app_frame) for scene in app_scenes for app_frame in scene.frames]


def capture_output(func):
    """Wrapper to capture print output."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout

    return wrapper


def get_sample_inputs(input_file):
    """
    Extracts frames from a video at a regular interval defined by the ratio of the video's frame rate to a desired sampling rate.

    The function opens a video file, calculates the sampling interval based on the video's frame rate and a pre-defined SAMPLER_FPS, and then iterates through the video to capture frames at these intervals. It captures the frame indices and their corresponding frames in a dictionary, and also gathers metadata about the video.

    Parameters:
    input_file (str): The path to the video file from which frames will be sampled.

    Returns:
    tuple: A tuple containing two elements:
        - sample_input_indexes (dict): A dictionary where the keys are the frame indices (int) sampled from the video
          and the values are the corresponding frames (numpy arrays).
        - metadata (dict): A dictionary containing metadata of the video with keys 'width' (float), 'height' (float),
          and 'fps' (float) representing the width, height, and frames per second of the video respectively.

    Notes:
    - The function assumes that a global constant `SAMPLER_FPS` is defined outside the function which specifies the desired sampling rate in frames per second.
    - The function will attempt to open the video file specified by `input_file`. If the file cannot be opened, the behavior depends on OpenCV's error handling for `cv2.VideoCapture`.
    - Frames are sampled at intervals determined by `fps/SAMPLER_FPS`. For example, if `fps` is 30 and `SAMPLER_FPS` is 10, it samples every 3rd frame.
    - The function reads frames until the end of the video or until it fails to read a frame, at which point it stops reading and releases the video resource.

    Example Usage:
    ```
    frames, video_info = get_sample_inputs("path/to/video.mp4")
    print("Sampled frames:", len(frames))
    print("Video metadata:", video_info)
    ```

    Raises:
    - The function does not explicitly handle exceptions. Errors during video file opening, reading, or processing need to be handled by the calling code.
    """
    cap = cv2.VideoCapture(input_file)
    fps=cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    incrementer=fps/SAMPLER_FPS
    idx = incrementer
    sample_input_indexes = {}
    metadata = {
        'width': width,
        'height': height,
        'fps': fps
    }
    while idx < frame_count:
        idx_int = int(idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_int)
        ret, frame = cap.read()
        if not ret:
            break
        sample_input_indexes[idx] = frame
        idx = idx + incrementer
    cap.release()
    return sample_input_indexes, metadata


def np_array_to_image(img):
    """
    Converts a NumPy array image from BGR to RGB format and creates a PIL Image object.

    This function is useful for converting images loaded or processed by OpenCV (which uses BGR color order) 
    into PIL Image objects that use the standard RGB color order. This conversion makes the image suitable 
    for tasks that require PIL Image format, such as displaying with certain Python libraries or further 
    processing in applications that expect the RGB format.

    Parameters:
    - img (numpy.ndarray): An image array in BGR format, typically read or processed by OpenCV.

    Returns:
    - PIL.Image.Image: The converted image as a PIL Image object in RGB format.

    Example Usage:
    >>> import cv2
    >>> original_img = cv2.imread('path_to_image.jpg')  # Load image in BGR format
    >>> converted_img = np_array_to_image(original_img)
    >>> converted_img.show()  # Display the image; requires a GUI environment

    Note:
    - This function directly converts the color format from BGR (Blue, Green, Red) to RGB (Red, Green, Blue),
      and wraps the result in a PIL Image object.
    """
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(imgrgb)
    return image


def get_resized_image(np_img, width, height):
    """
    Resizes a given image to the specified width and height using linear interpolation.

    This function takes a NumPy array representing an image and resizes it to the dimensions provided.
    It uses linear interpolation for resizing, which is generally suitable for both upscaling and
    downscaling the image while maintaining a balance between quality and computational efficiency.

    Parameters:
    - np_img (numpy.ndarray): The image to resize, represented as a NumPy array. The image can be in any
      color format recognized by OpenCV, such as grayscale, BGR, or RGB.
    - width (int): The desired width of the resized image.
    - height (int): The desired height of the resized image.

    Returns:
    - numpy.ndarray: The resized image as a NumPy array.

    Example Usage:
    >>> import cv2
    >>> original_img = cv2.imread('path_to_image.jpg')  # Load image
    >>> resized_img = get_resized_image(original_img, 300, 200)  # Resize to 300x200 pixels
    >>> cv2.imshow('Resized Image', resized_img)  # Display the resized image
    >>> cv2.waitKey(0)  # Wait for a key press
    >>> cv2.destroyAllWindows()  # Close the display window

    Notes:
    - INTER_LINEAR interpolation is used here, which is a good default choice for resizing operations
      in most cases. It performs well with both enlarging and reducing the size of images and is faster
      than some other methods like cubic interpolation.
    """
    resized_img = cv2.resize(np_img, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_img


def display_all_images(images, cols=5):
    """
    Displays a list of images in a grid format with a specified number of columns using Matplotlib.

    This function takes a list of images, each represented as a NumPy array, and arranges them in a grid for display.
    Each image is converted to a display format using a helper function `np_array_to_image`, and then shown without axes in a grid layout.

    Parameters:
    - images (list of numpy.ndarray): List of images where each image is represented as a NumPy array.
    - cols (int, optional): Number of columns in the grid display. Defaults to 5.

    Returns:
    - None: Images are displayed using Matplotlib and not returned.

    Example Usage:
    >>> import cv2
    >>> img_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    >>> images = [cv2.imread(path) for path in img_paths]  # Load images with OpenCV
    >>> display_all_images(images, cols=3)  # Display images in a grid with 3 columns

    Notes:
    - Ensure all images are of a compatible type (e.g., all are color images or all are grayscale).
    - The function uses `np_array_to_image` to convert BGR images (typical in OpenCV) to RGB for display.
    - If the number of images is not a perfect multiple of the column count, the last row will have empty spaces.
    """
    num_images = len(images)
    rows = int(np.ceil(num_images / cols))  # Calculate the necessary number of rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))  # Create a figure and axes grid
    cnt = 0  # Counter to track image processing
    
    for ax, img in tqdm(zip(axes.flat, images), total=num_images, desc='Processing images to display'):
        display_image = np_array_to_image(img)  # Convert image for display
        ax.imshow(display_image)  # Display image in corresponding subplot
        ax.axis('off')  # Hide axes for cleaner display
        cnt += 1
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Render the figure

