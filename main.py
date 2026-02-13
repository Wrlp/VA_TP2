
import cv2
import numpy as np
import glob 
import os
from pathlib import Path
from preprocessing import traitement

def main():
    path = "img/"
    traitement(path)

if __name__ == "__main__":
    main()
