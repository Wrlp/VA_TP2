
import cv2
import numpy as np
import glob 
import os
from pathlib import Path
from preprocessing import traitement
from segmentation_basic import segmentation_basic

def main():
    path = "img/"
    traitement(path)
    seg_path = "results/preprocessing/"
    segmentation_basic(seg_path)

if __name__ == "__main__":
    main()
