
import cv2
import numpy as np
import glob 
import os
from pathlib import Path
from preprocessing import traitement
from segmentation_basic import segmentation_basic
# from segmentation_watershed import watershed_segmentation
# from segmentation_advanced import segmentation_advanced
from features import process_segmentation_results

def main():
    path = "img/"

    preprocessing_path = "results/preprocessing/"
    os.makedirs(preprocessing_path, exist_ok=True)
    traitement(path)
    
    seg_basic_path = "results/seg_basic/"
    os.makedirs(seg_basic_path, exist_ok=True)
    print("--- Segmentation Basique ---")
    objects_in_images = segmentation_basic(preprocessing_path)
    process_segmentation_results(objects_in_images, preprocessing_path, seg_basic_path + "features.csv", seg_basic_path)

    seg_watershed_path = "results/seg_watershed/"
    os.makedirs(seg_watershed_path, exist_ok=True)
    print("--- Segmentation Watershed ---")
    # objects_in_images = watershed_segmentation(preprocessing_path)
    # process_segmentation_results(objects_in_images, preprocessing_path, seg_watershed_path + "features.csv", seg_watershed_path)

    seg_advanced_path = "results/seg_advanced/"
    os.makedirs(seg_advanced_path, exist_ok=True)
    print("--- Segmentation Avancée ---")
    # objects_in_images = segmentation_advanced(preprocessing_path)
    # process_segmentation_results(objects_in_images, preprocessing_path, seg_advanced_path + "features.csv", seg_advanced_path)

if __name__ == "__main__":
    main()
