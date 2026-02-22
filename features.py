import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from preprocessing import load_and_resize

def extract_features(image: np.ndarray, contours: list[np.ndarray], image_name: str) -> list[dict]:
    """
    Extract features (Mean B, G, R) for each grain in the image.
    
    Args:
        image (np.ndarray): The original BGR image.
        contours (list[np.ndarray]): List of contours representing grains.
        image_name (str): Name of the image file.
        
    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - 'Image': The image filename.
            - 'Grain_ID': The index of the grain (0-based).
            - 'Mean_B': Mean Blue channel value.
            - 'Mean_G': Mean Green channel value.
            - 'Mean_R': Mean Red channel value.
    """
    features_list = []
    
    for i, contour in enumerate(contours):
        # Create a mask for the current grain
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Calculate mean color in the masked area
        # cv2.mean returns (B, G, R, A) for 3-channel image with mask
        mean_val = cv2.mean(image, mask=mask)
        
        features = {
            'Image': image_name,
            'Grain_ID': i + 1,  # 1-based index for user friendliness
            'Mean_B': mean_val[0],
            'Mean_G': mean_val[1],
            'Mean_R': mean_val[2]
        }
        features_list.append(features)
        
    return features_list

def visualize_grains(image: np.ndarray, contours: list[np.ndarray], features: list[dict]) -> np.ndarray:
    """
    Visualize grains by drawing contours and their IDs on the image.
    
    Args:
        image (np.ndarray): The original BGR image.
        contours (list[np.ndarray]): List of contours.
        features (list[dict]): Extracted features (used for IDs).
        
    Returns:
        np.ndarray: The annotated image.
    """
    vis_image = image.copy()
    
    # Draw all contours first
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
    
    for i, contour in enumerate(contours):
        # Calculate centroid for text placement
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # Fallback to bounding rect center if moments fail
            x, y, w, h = cv2.boundingRect(contour)
            cX, cY = x + w // 2, y + h // 2
            
        grain_id = features[i]['Grain_ID']
        
        # Draw ID text
        cv2.putText(vis_image, str(grain_id), (cX - 10, cY), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis_image, str(grain_id), (cX - 10, cY), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return vis_image

def save_features_to_csv(features_list: list[dict], filename: str):
    """
    Save the list of feature dictionaries to a CSV file.
    
    Args:
        features_list (list[dict]): List of features to save.
        filename (str): Output CSV filename.
    """
    if not features_list:
        print("No features to save.")
        return
        
    keys = features_list[0].keys()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(features_list)
    print(f"Features saved to {filename}")

def process_segmentation_results(segmentation_results: dict[str, list[np.ndarray]], 
                                 image_folder: str, 
                                 output_csv: str = "results/features.csv", 
                                 output_vis_folder: str = "results/features_vis/"):
    """
    Process segmentation results to extract features, visualize, and save to CSV.
    
    Args:
        segmentation_results (dict): Dictionary mapping filenames to contours.
        image_folder (str): Path to the folder containing original images.
        output_csv (str): Path to save the CSV file.
        output_vis_folder (str): Path to save visualization images.
    """
    # Load original images
    print(f"Loading images from {image_folder}...")
    images = load_and_resize(image_folder)
    
    all_features = []
    
    # Ensure visualization directory exists
    if output_vis_folder:
        os.makedirs(output_vis_folder, exist_ok=True)
    
    print("Extracting features...")
    for filename, contours in segmentation_results.items():
        if filename not in images:
            print(f"Warning: Image {filename} not found in loaded images. Skipping.")
            continue
            
        image = images[filename]
        
        # Extract features
        features = extract_features(image, contours, filename)
        all_features.extend(features)
        
        # Visualize
        if output_vis_folder:
            vis_img = visualize_grains(image, contours, features)
            vis_path = os.path.join(output_vis_folder, f"vis_{filename}")
            cv2.imwrite(vis_path, vis_img)
            
    # Save to CSV
    if all_features:
        save_features_to_csv(all_features, output_csv)
    else:
        print("No features extracted.")

if __name__ == "__main__":
    # Example usage (integration test)
    # This block assumes segmentation_basic is available and runnable
    try:
        from segmentation_basic import segmentation_basic
        
        image_folder = "results/preprocessing/" # Or original images folder if desired
        # Ensure there are images in the folder for testing
        if not os.path.exists(image_folder) or not os.listdir(image_folder):
             # Fallback to a known location or skip test
             print(f"Image folder {image_folder} not found or empty. Skipping test.")
        else:
            print("Running basic segmentation...")
            # We assume segmentation_basic returns the dict of contours
            seg_results = segmentation_basic(image_folder, save=False, visualise=False)
            
            print("Processing results...")
            process_segmentation_results(seg_results, image_folder, 
                                         output_csv="results/grain_features.csv",
                                         output_vis_folder="results/grain_vis/")
                                         
    except ImportError:
        print("segmentation_basic module not found. Run this from the project root.")
    except Exception as e:
        print(f"An error occurred: {e}")
