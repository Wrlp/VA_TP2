import cv2
import numpy as np
import os
from preprocessing import load_and_resize, preprocess
from skimage.feature import peak_local_max
from scipy import ndimage
from preprocessing import load_and_resize, traitement



def watershed_segmentation(path: str, save: bool = False, visualise: bool = False, out: str = "results/seg_watershed/") -> dict[str, list[np.ndarray]]:
    """
    Segmentation des grains par Watershed.
    Retourne :
        { nom_image : [contours] }
    """

    objects_in_images = {}
    images = load_and_resize(path)

    for name, img in images.items():
        # Treatement
        enhanced, _ = preprocess(img)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # blurred to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Global Ostu
        otsu_val, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = max(20, int(otsu_val * 0.6))
        _, binary_low = cv2.threshold(blurred, low_thresh, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_or(binary_otsu, binary_low)


        # Mophological closing
        kernel_small = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=3)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # Markers
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

        coarse_mask = (dist_transform > 0.22 * dist_transform.max()).astype(np.uint8)
        n_coarse, coarse_labels = cv2.connectedComponents(coarse_mask)
        sizes = [np.sum(coarse_labels == l) for l in range(1, n_coarse)]

        if sizes:
            median_size = np.median(sizes)
            estimated_radius = int(np.sqrt(median_size / np.pi))
            min_dist = max(10, int(estimated_radius * 0.60))  # Avoide double peaks
        else:
            min_dist = 15

        local_max = peak_local_max(
            dist_transform,
            min_distance=min_dist,
            threshold_rel=0.35,   
            labels=opening
        )

        peak_mask = np.zeros(dist_transform.shape, dtype=bool)
        peak_mask[tuple(local_max.T)] = True
        markers, _ = ndimage.label(peak_mask)
        markers[opening == 0] =-1
        
        # Watershed
        markers = cv2.watershed(img, markers.astype(np.int32))
        
        # Extraction 
        objects = []
        for label in np.unique(markers):
            if label <= 1: 
                continue

            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == label] = 255

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 200 < area < 60000:
                    objects.append(cnt)

        objects_in_images[name] = objects

        # Print
        result = img.copy()
        cv2.drawContours(result, objects, -1, (0, 0, 255), 1)
        print(f"{name} → {len(objects)} grains détectés  (min_distance={min_dist})")
        if save:
            os.makedirs(out, exist_ok=True)
            cv2.imwrite(os.path.join(out, name), result)
        if visualise:
            dist_display = (dist_norm * 255).astype(np.uint8)
            dist_display = cv2.applyColorMap(dist_display, cv2.COLORMAP_JET)
            combined = np.hstack((result, dist_display))
            cv2.imshow(f"Watershed | dist_min={min_dist}", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return objects_in_images


if __name__ == "__main__":
    #path = "C:\\Users\\Annam\\Downloads\\Images"
    #traitement(path, visualise=True)
    seg_path = "results/seg_watershed/"
    objects_in_images = watershed_segmentation(seg_path, save=True, visualise=True)
    print(objects_in_images.keys())
