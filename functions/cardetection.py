import os

from ultralytics import YOLOWorld
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image, visualize_object_predictions

from tqdm import tqdm

from pathlib import Path

import gc 

OUTPUT_DIR = Path("outputs/0_detection")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_yolo_labels(predictions, image_path, output_dir, class_names):
    """
    Save detection results in YOLO format.

    Args:
        predictions (list): List of SAHI ObjectPrediction objects.
        image_path (str): Path to the input image.
        output_dir (Path): Directory to save YOLO labels.
        class_names (list): List of class names.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Get image dimensions
    image = read_image(image_path)
    height, width = image.shape[:2]

    # Prepare YOLO label file path
    label_file = Path(image_path).stem + ".txt"
    label_path = output_dir / label_file

    # Write YOLO format labels
    with open(label_path, "w") as f:
        for pred in predictions:
            # Convert bbox to YOLO format (x_center, y_center, width, height)
            bbox = pred.bbox
            x_center = (bbox.minx + bbox.maxx) / 2 / width
            y_center = (bbox.miny + bbox.maxy) / 2 / height
            w = (bbox.maxx - bbox.minx) / width
            h = (bbox.maxy - bbox.miny) / height

            # Write to file
            f.write(f"{class_names.index(pred.category.name)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def detect_cars(
    image_path: str,
    model: YOLOWorld,
    output_dir: Path = OUTPUT_DIR,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.1,
    overlap_width_ratio: float = 0.1,
    confidence_threshold: float = 0.25,
    save_labels: bool = True,
    save_visuals: bool = True,
    verbose: bool = True
):
    """
    Detect cars in an image using YOLO-World and SAHI for tiled inference.

    Args:
        image_path (str): Path to the input image.
        model (YOLOWorld): YOLO-World model instance.
        output_dir (Path): Directory to save outputs.
        slice_height (int): Height of each tile.
        slice_width (int): Width of each tile.
        overlap_height_ratio (float): Overlap ratio between tiles (height).
        overlap_width_ratio (float): Overlap ratio between tiles (width).
        confidence_threshold (float): Confidence threshold for detections.
        save_labels (bool): Whether to save labels in YOLO format.
        save_visuals (bool): Whether to save visual outputs.
        verbose (bool): Whether to print verbose output.
    """
    # Load image using SAHI
    image = read_image(image_path)

    # Initialize SAHI detection model with YOLO-World
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model=model,
        confidence_threshold=confidence_threshold,
        device="cuda:0" if model.device.type == "cuda" else "cpu",
    )

    # Perform tiled inference using SAHI
    results = get_sliced_prediction(
        image,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    # Save labels in YOLO format
    if save_labels:
        yolo_labels_dir = output_dir / "labels"
        save_yolo_labels(results.object_prediction_list, image_path, yolo_labels_dir, class_names=["car"])
        if verbose:
            print(f"Labels saved to {yolo_labels_dir}")

    # Save visual outputs
    if save_visuals:
        visuals_dir = output_dir / "visuals"
        output_image_path = visuals_dir / Path(image_path).name
        visualize_object_predictions(
            image=image,
            object_prediction_list=results.object_prediction_list,
            output_dir=visuals_dir,
            file_name=Path(image_path).stem,
        )
        if verbose:
            print(f"Visual results saved to {output_image_path}")

    # Print verbose output
    if verbose:
        print(f"Number of detections: {len(results.object_prediction_list)}")
        for i, pred in enumerate(results.object_prediction_list):
            print(
                f"Detection {i + 1}: Class: {pred.category.name}, "
                f"Confidence (Pred. Score) {pred.score.value:.2f}, BBox={pred.bbox.to_xyxy()}"
            )

    del image  # Explicitly delete image
    gc.collect()