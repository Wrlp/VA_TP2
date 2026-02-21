import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from preprocessing import load_and_resize 

def show_changes(list_of_images: list[np.ndarray], color: bool = False) -> None:
    """Display a series of images in a grid layout for comparison.
    This function creates a horizontal grid of subplots to visualize multiple
    images side by side.
    
    Args:
        list_of_images (list): A list of images (numpy arrays) to be displayed.
        color (bool, optional): If True, converts BGR images to RGB before display.
            If False, displays images in grayscale. Defaults to False.
    
    Returns:
        None
    """
    
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(list_of_images):
        plt.subplot(1, len(list_of_images), i + 1)
        if color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray')
        plt.title(f"Step {i+1}")
        plt.axis('off')
    plt.show()

def morphological_operations(binary: np.ndarray) -> np.ndarray:
    """
    Applies a series of morphological operations to a binary image.
    The operations help remove noise, fill small holes, and separate touching objects.
    
    Args:
        binary: A binary image (numpy array with values 0 or 255) to be processed.
    
    Returns:
        A binary image (numpy array) after applying the morphological operations.
    """
    kernel = np.ones((5,5), np.uint8)
    step = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, (7,7), iterations=10)
    
    step = cv2.morphologyEx(step, cv2.MORPH_OPEN, kernel, iterations=1)
    step = cv2.morphologyEx(step, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    step = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    step = cv2.morphologyEx(step, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    step = cv2.morphologyEx(step, cv2.MORPH_ERODE, kernel, iterations=3)
    return step

def segmentation_basic(path: str, save: bool = False, visualise: bool = False, out: str = "results/seg_basic/") -> dict[str, list[np.ndarray]]:
    """
    Perform basic image segmentation using Otsu's thresholding and morphological operations.
    Detects and extracts objects from images by applying grayscale conversion, binary
    thresholding, morphological operations, and contour detection.
    
    Args:
        path (str): Directory path containing images to segment.
        save (bool, optional): Save segmented results to disk. Defaults to False.
        visualise (bool, optional): Display segmentation results. Defaults to False.
        out (str, optional): Output directory for saving images. Defaults to "results/seg_basic/".
    
    Returns:
        dict[str, list[np.ndarray]]: Image filenames mapped to detected contours
            (area > 200 pixels).
    """
    
    objects_in_images = {}
    images = load_and_resize(path)
    for name, img in images.items():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        step = morphological_operations(binary)
        
        contours, _ = cv2.findContours(step, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        objects = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
        objects_in_images[name] = objects
        
        # Draw contours and bounding boxes
        result = img.copy()
        for obj in objects:
            x,y,w,h = cv2.boundingRect(obj)
            cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),1)
        
        # cv2.drawContours(result, contours, -1, (255,0,255), 1)
        if save:
            output_path = os.path.join(out, name)
            cv2.imwrite(output_path, result)
        if visualise:
            show_changes([img, result], color=True)
            
    return objects_in_images

if __name__ == "__main__":
    path = "results/preprocessing/"
    objects_in_images = segmentation_basic(path, save=True, visualise=False)
    print(objects_in_images.keys())