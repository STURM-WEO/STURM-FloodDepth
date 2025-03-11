import os
import numpy as np
import cv2
import matplotlib as plt


# Compute Homography Using OpenCV & RANSAC
def compute_homography(results, query_img, ref_img, show_warped=True):
    """Compute homography matrix using OpenCV and RANSAC."""
    output = results[0]
    keypoints0 = output["keypoints0"].numpy()
    keypoints1 = output["keypoints1"].numpy()

    if len(keypoints0) < 4 or len(keypoints1) < 4:
        print("Not enough keypoints for homography estimation.")
        return None, None

    # Convert keypoints to float32
    src_pts = np.float32(keypoints0).reshape(-1, 1, 2)
    dst_pts = np.float32(keypoints1).reshape(-1, 1, 2)

    # Compute homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

    if H is not None:
        print("Homography matrix computed successfully:")
        print(H)
        # Apply homography transformation to warp image
        height, width = ref_img.size
        warped_image = cv2.warpPerspective(np.array(query_img), H, (width, height))

        # Show warped image
        if show_warped:
            plt.figure(figsize=(10, 5))
            plt.imshow(ref_img)
            plt.imshow(warped_image, alpha=0.7)
            plt.axis("off")
            plt.title("Warped Image using Homography")
            plt.show()

    else:
        print("Failed to compute a valid homography matrix.")
    return H, mask



def load_yolo_detections(txt_path, img_width, img_height):
    """Load YOLO detection centers from the label file and convert to absolute coordinates."""
    detections = []
    with open(txt_path, "r") as file:
        for line in file.readlines():
            values = line.strip().split()
            class_id = int(values[0])  # Object class
            cx_norm, cy_norm, w_norm, h_norm = map(float, values[1:5])
            score = float(values[5])  # Confidence score

            # Convert normalized YOLO coordinates to absolute pixel values
            cx = int(cx_norm * img_width)
            cy = int(cy_norm * img_height)
            w = int(w_norm * img_width)
            h = int(h_norm * img_height)

            detections.append((cx, cy, w, h, class_id))  # Append class info as well
    return detections

def warp_yolo_detections(yolo_detections, H):
    """Warp YOLO detection centers using the computed homography matrix."""
    if H is None:
        print("No valid homography matrix. Cannot warp detections.")
        return None

    centers = np.array([[x, y, 1] for x, y, _, _, _ in yolo_detections])  # Convert to homogeneous coordinates
    centers = centers.T  # Shape: (3, N) for matrix multiplication

    # Apply homography
    transformed_centers = H @ centers  # Matrix multiplication
    transformed_centers /= transformed_centers[2]  # Normalize by depth

    return transformed_centers[:2].T  # Return transformed (x, y) coordinates

def visualize_transformed_detections(ref_img, transformed_centers, yolo_detections, geo_output_dir, img_path, save_warped_detections=True):
    """Overlay transformed detection centers on the reference image."""
    plt.figure(figsize=(15, 10))
    plt.imshow(ref_img)
    plt.axis("off")
    # plt.title("Warped YOLO Detections on Reference Image")

    if transformed_centers is not None:
        for (x, y), (_, _, _, _, class_id) in zip(transformed_centers, yolo_detections):
            plt.scatter(x, y, c="white", s=70, alpha=0.5) # label=f"Level {class_id}"
            plt.text(x, y+80, f"L{class_id}", fontsize=25, color="white",
                    ) # bbox=dict(facecolor='white', alpha=0.1, edgecolor='none', boxstyle='round,pad=0.3')
            
    if save_warped_detections:
        save_path = f"{geo_output_dir}/{os.path.basename(img_path).split('.')[0]}_warped_detections.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Warped detections saved to: {save_path}")
    plt.show()