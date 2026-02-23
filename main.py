
import cv2
import numpy as np
import glob 
import os
from pathlib import Path
from preprocessing import traitement
from segmentation_basic import segmentation_basic
from segmentation_watershed import watershed_segmentation

def main():
    path = "img/"
    traitement(path)
    seg_path = "results/preprocessing/"
    objects_in_images = segmentation_basic(seg_path) # Dict avec key = nom img et value = liste de contours (peux drawContours à partir de ça)
    objects_in_images_watershed = watershed_segmentation(seg_path, save=True, visualise=True)
    

if __name__ == "__main__":
    main()
