import os

from transformers import AutoImageProcessor, SuperGlueForKeypointMatching

superglue_processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
superglue_model = SuperGlueForKeypointMatching.from_pretrained("magic-leap-community/superglue_outdoor")

from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def load_and_convert_image(image_path):
    """Load an image and convert it to RGB format."""
    return Image.open(image_path).convert("RGB")

def preprocess_image_pairs(image_pairs, superglue_processor):
    """Preprocess pairs of images using the SuperGlue processor."""
    images = [[load_and_convert_image(pair[0]), load_and_convert_image(pair[1])] for pair in image_pairs]
    inputs = superglue_processor(images, return_tensors="pt")
    return images, inputs

def run_superglue_model(superglue_model, inputs):
    """Run the SuperGlue model on the preprocessed inputs."""
    with torch.no_grad():
        outputs = superglue_model(**inputs)
    return outputs

def postprocess_results(outputs, images, superglue_processor, threshold=0.2):
    """Post-process the results to map keypoints and filter by matching confidence."""
    image_sizes = [[(image.height, image.width) for image in pair] for pair in images]
    results = superglue_processor.post_process_keypoint_matching(outputs, image_sizes, threshold=threshold)
    return results

def print_results(results):
    """Print the results of the keypoint matching."""
    for i, result in enumerate(results):
        print(f"For the image pair #{i} `query_img` match `ref_img` at keypoints:")
        for keypoint0, keypoint1, matching_score in zip(
                result["keypoints0"], result["keypoints1"], result["matching_scores"]
        ):
            xy1 = tuple(keypoint0.numpy())
            xy2 = tuple(keypoint1.numpy())
            print(
                f"  - xy1 {str(xy1):12s} <-> xy2 {str(xy2):12s} confidence = {matching_score:.2f}"
            )

# def visualize_matches(query_img, ref_img, results):
#     """Visualize the matches between the query and reference images."""
#     # Create side by side image
#     # merged_image = np.zeros((max(query_img.height, ref_img.height), query_img.width + ref_img.width, 3))
#     merged_image = np.ones((max(query_img.height, ref_img.height), query_img.width + ref_img.width, 3))
#     merged_image[: query_img.height, : query_img.width] = np.array(query_img) / 255.0
#     merged_image[: ref_img.height, query_img.width :] = np.array(ref_img) / 255.0

#     plt.figure(figsize=(15, 15))
#     plt.imshow(merged_image)
#     plt.axis("off")

#     # Retrieve the keypoints and matches
#     output = results[0]
#     keypoints0 = output["keypoints0"]
#     keypoints1 = output["keypoints1"]
#     matching_scores = output["matching_scores"]
#     keypoints0_x, keypoints0_y = keypoints0[:, 0].numpy(), keypoints0[:, 1].numpy()
#     keypoints1_x, keypoints1_y = keypoints1[:, 0].numpy(), keypoints1[:, 1].numpy()

#     # Plot the matches
#     for keypoint0_x, keypoint0_y, keypoint1_x, keypoint1_y, matching_score in zip(
#             keypoints0_x, keypoints0_y, keypoints1_x, keypoints1_y, matching_scores
#     ):
#         plt.plot(
#             [keypoint0_x, keypoint1_x + query_img.width],
#             [keypoint0_y, keypoint1_y],
#             color=plt.get_cmap("viridis")(matching_score.item()),
#             alpha=0.9,
#             linewidth=0.5,
#         )
        
#         plt.scatter(keypoint0_x, keypoint0_y, c="white", s=2, alpha=0.9)
#         plt.scatter(keypoint1_x + query_img.width, keypoint1_y, c="white", s=2, alpha=0.9)

#     plt.show()

def visualize_matches(query_img, ref_img, results, geo_output_dir, img_path, save_match_png=True):
    """Visualize the matches between the query and reference images with a color legend."""
    # Create side by side image
    merged_image = np.ones((max(query_img.height, ref_img.height), query_img.width + ref_img.width, 3))
    merged_image[: query_img.height, : query_img.width] = np.array(query_img) / 255.0
    merged_image[: ref_img.height, query_img.width :] = np.array(ref_img) / 255.0

    plt.figure(figsize=(15, 15))
    plt.imshow(merged_image)
    plt.axis("off")

    # Retrieve the keypoints and matches
    output = results[0]
    keypoints0 = output["keypoints0"]
    keypoints1 = output["keypoints1"]
    matching_scores = output["matching_scores"]
    keypoints0_x, keypoints0_y = keypoints0[:, 0].numpy(), keypoints0[:, 1].numpy()
    keypoints1_x, keypoints1_y = keypoints1[:, 0].numpy(), keypoints1[:, 1].numpy()

    # Get score range for color normalization
    min_score = matching_scores.min().item()
    max_score = matching_scores.max().item()
    norm = plt.Normalize(vmin=min_score, vmax=max_score)
    cmap = plt.get_cmap("viridis")

    # Plot the matches
    for kp0_x, kp0_y, kp1_x, kp1_y, score in zip(
            keypoints0_x, keypoints0_y, keypoints1_x, keypoints1_y, matching_scores
    ):
        plt.plot(
            [kp0_x, kp1_x + query_img.width],
            [kp0_y, kp1_y],
            color=cmap(norm(score.item())),
            alpha=0.9,
            linewidth=0.5,
        )
        plt.scatter(kp0_x, kp0_y, c="white", s=2, alpha=0.9)
        plt.scatter(kp1_x + query_img.width, kp1_y, c="white", s=2, alpha=0.9)

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Create a separate axes for the colorbar
     
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='horizontal', anchor=(0.05, 1.0), 
                        shrink=0.3, pad=-0.1)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks([np.round(min_score, decimals=2), np.round(max_score, decimals=2)])
    cbar.ax.yaxis.set_label_position('left')

    cbar.ax.set_xlabel("Matching Confidence", fontsize=30)

    if save_match_png:
        save_path = f"{geo_output_dir}/{os.path.basename(img_path).split('.')[0]}_superglue_matches.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Matches saved to: {save_path}")

    plt.show()
