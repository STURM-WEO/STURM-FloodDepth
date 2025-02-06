import os
import cv2
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def get_best_scaling(height, width):
    """ Determine the best scaling strategy to reach 224px minimum dimension. """
    min_dim = min(height, width)

    if min_dim >= 224:
        return None
    elif min_dim * 4 >= 224:
        return [4]
    elif min_dim * 3 >= 224:
        return [3]
    elif min_dim * 2 >= 224:
        return [2]
    elif min_dim * 4 * 2 >= 224:  
        return [4, 2]  # ~5x scaling
    elif min_dim * 3 * 2 >= 224:  
        return [3, 2]  # ~6x scaling
    else:
        return [4, 3]  # ~12x scaling (fallback for very small images)

def superresolve_image(image, sr_models, scales):
    """ Apply sequential upscaling using the best model combination. """
    for scale in scales:
        image = sr_models[scale].upsample(image)
    return image

def superresolve_images(image_dir, output_dir, sr_models, valid_extensions, csv_path):
    image_dir, output_dir = Path(image_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = []

    for image_path in tqdm(image_dir.glob("*"), desc="Rescaling Images", unit="image"):
        if image_path.suffix.lower() not in valid_extensions or not image_path.is_file():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        best_scales = get_best_scaling(height, width)

        output_path = output_dir / image_path.name
        if best_scales is None:
            shutil.copy(image_path, output_path)
            final_size = (width, height)
            scaling_steps = "None"
        else:
            image = superresolve_image(image, sr_models, best_scales)
            cv2.imwrite(str(output_path), image)
            final_size = image.shape[1], image.shape[0]
            scaling_steps = "x".join(map(str, best_scales))

        stats.append([image_path.name, width, height, final_size[0], final_size[1], scaling_steps])

    df = pd.DataFrame(stats, columns=["Filename", "Orig_Width", "Orig_Height", "Rescaled_Width", "Rescaled_Height", "Scaling"])
    df.to_csv(csv_path, index=False)