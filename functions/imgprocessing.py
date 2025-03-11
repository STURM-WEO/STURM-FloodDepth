import os
import random
import cv2
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path

from tqdm import tqdm

import tensorflow as tf

def draw_bounding_boxes(image_path, label_path, show_class=True):
    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Read the labels
    with open(label_path, 'r') as f:
        labels = f.readlines()

    for label in labels:
        data = label.strip().split()
        class_id, x_center, y_center, bbox_width, bbox_height = map(float, data)
        x_center *= width
        y_center *= height
        bbox_width *= width
        bbox_height *= height

        # Calculate the top-left and bottom-right coordinates of the bounding box
        x1 = int(x_center - (bbox_width / 2))
        y1 = int(y_center - (bbox_height / 2))
        x2 = int(x_center + (bbox_width / 2))
        y2 = int(y_center + (bbox_height / 2))

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (128, 0, 32), 2)
        if show_class:
            # Put the label text above the bounding box
            cv2.putText(image, f'{int(class_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (128, 0, 32), 2)

    return image

def visualize_random_batch(image_folder, label_folder, batch_size=5, show_class=True):
    # Get a list of all images and labels
    images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    labels = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    # Ensure there are images and labels
    if not images or not labels:
        print("No images or labels found in the provided folders.")
        return

    # Sample a random batch of images and their corresponding label files
    sampled_images = random.sample(images, batch_size)

    plt.figure(figsize=(15, batch_size * 5))

    for idx, random_image in enumerate(sampled_images):
        corresponding_label = random_image.replace('.jpg', '.txt')

        # Paths to the image and label file
        image_path = os.path.join(image_folder, random_image)
        label_path = os.path.join(label_folder, corresponding_label)

        # Draw bounding boxes on the image
        image_with_boxes = draw_bounding_boxes(image_path, label_path, show_class=show_class)

        # Convert BGR image to RGB
        image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

        # Display the image with bounding boxes
        plt.subplot(batch_size, 1, idx + 1)
        plt.imshow(image_with_boxes_rgb)
        plt.title(f'{random_image}')
        plt.axis('off')

    plt.show()

def enlarge_bbox(x_center, y_center, width, height, scale=1.2):
    new_width = min(1.0, width * scale)
    new_height = min(1.0, height * scale)
    return x_center, y_center, new_width, new_height

def load_yolo_labels(label_path):
    if not os.path.exists(label_path):
        return []
    
    with open(label_path, "r") as f:
        return [list(map(float, line.strip().split())) for line in f]

def crop_detected_vehicles(image_folder, label_folder, crop_output_dir, scale=1.2, verbose=True):
    image_folder, label_folder, crop_output_dir = map(Path, [image_folder, label_folder, crop_output_dir])
    crop_output_dir.mkdir(parents=True, exist_ok=True)

    filename_counter = {}
    image_files = list(image_folder.glob("*.jpg"))

    for image_path in tqdm(image_files, desc="Cropping Vehicles", unit="image"):
        label_path = label_folder / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path))
        img_height, img_width, _ = image.shape
        labels = load_yolo_labels(str(label_path))

        if not labels:
            continue

        base_filename = image_path.stem.replace(" ", "")
        for class_id, x_center, y_center, width, height in labels:
            x_center, y_center, width, height = enlarge_bbox(x_center, y_center, width, height, scale)

            x1 = max(0, int((x_center - width / 2) * img_width))
            y1 = max(0, int((y_center - height / 2) * img_height))
            x2 = min(img_width, int((x_center + width / 2) * img_width))
            y2 = min(img_height, int((y_center + height / 2) * img_height))

            cropped_image = image[y1:y2, x1:x2]
            if cropped_image.size == 0:
                continue

            filename_counter[base_filename] = filename_counter.get(base_filename, 0) + 1
            output_filepath = crop_output_dir / f"{base_filename}_{filename_counter[base_filename]}.jpg"

            cv2.imwrite(str(output_filepath), cropped_image)
            if verbose:
                print(f"Cropped and saved: {output_filepath}")
                
        del image  # Free the original image after processing

# Function to load image as bytes
def load_image_as_bytes(img_path):
    with open(img_path, "rb") as img_file:
        img_bytes = img_file.read()
    return tf.convert_to_tensor([img_bytes], dtype=tf.string) 