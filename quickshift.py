import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from skimage import color
import cv2
import os

from preprocessing import load_and_resize


def quickshift(image: np.ndarray, 
               kernel_size: float = 3.0,
               max_dist: float = 10.0,
               ratio: float = 1.0) -> np.ndarray:
    """Segmentation Quickshift en regroupant pixels par mode-seeking."""
    # Conversion en espace Lab
    if len(image.shape) == 3 and image.shape[2] == 3:
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        image_lab = color.rgb2lab(image)
    else:
        image_lab = image.copy()
    
    h, w = image_lab.shape[:2]
    n_pixels = h * w
    
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Matrice des caracteristiques (position + couleur)
    if len(image_lab.shape) == 3:
        features = np.zeros((n_pixels, 5))
        features[:, 0] = y_coords.flatten()
        features[:, 1] = x_coords.flatten()
        features[:, 2:5] = image_lab.reshape(-1, 3)
    else:
        features = np.zeros((n_pixels, 3))
        features[:, 0] = y_coords.flatten()
        features[:, 1] = x_coords.flatten()
        features[:, 2] = image_lab.flatten()
    
    features[:, 0:2] *= ratio
    
    # Calcul de la densite
    densities = compute_density(features, kernel_size)
    
    # Trouver les parents (voisins avec densite plus haute)
    parents = find_parent_links(features, densities, max_dist, ratio)
    
    # Creation des segments
    segments = create_segments(parents, h, w)
    
    return segments


def compute_density(features: np.ndarray, kernel_size: float) -> np.ndarray:
    """Calcule la densité locale de chaque pixel avec un noyau gaussien."""
    n_pixels = features.shape[0]
    densities = np.zeros(n_pixels)
    
    batch_size = 1000
    
    for i in range(0, n_pixels, batch_size):
        end_idx = min(i + batch_size, n_pixels)
        batch = features[i:end_idx]
        
        distances = euclidean_distances(batch, features)
        weights = np.exp(-distances**2 / (2 * kernel_size**2))
        densities[i:end_idx] = weights.sum(axis=1)
    
    return densities


def find_parent_links(features: np.ndarray, 
                      densities: np.ndarray,
                      max_dist: float,
                      ratio: float) -> np.ndarray:
    """Trouve pour chaque pixel le voisin avec la densité la plus élevée."""
    n_pixels = features.shape[0]
    parents = np.arange(n_pixels)
    
    batch_size = 1000
    
    for i in range(0, n_pixels, batch_size):
        end_idx = min(i + batch_size, n_pixels)
        batch = features[i:end_idx]
        
        spatial_dist = euclidean_distances(batch[:, :2], features[:, :2])
        mask = spatial_dist <= max_dist
        full_dist = euclidean_distances(batch, features)
        
        for j in range(end_idx - i):
            pixel_idx = i + j
            valid_neighbors = mask[j]
            
            if not valid_neighbors.any():
                continue
            
            neighbor_densities = densities[valid_neighbors]
            current_density = densities[pixel_idx]
            higher_density_mask = neighbor_densities > current_density
            
            if higher_density_mask.any():
                valid_indices = np.where(valid_neighbors)[0]
                higher_density_indices = valid_indices[higher_density_mask]
                distances_to_higher = full_dist[j, higher_density_indices]
                best_neighbor_idx = higher_density_indices[np.argmin(distances_to_higher)]
                parents[pixel_idx] = best_neighbor_idx
    
    return parents


def create_segments(parents: np.ndarray, h: int, w: int) -> np.ndarray:
    """Crée les segments en suivant les liens parents jusqu'aux modes."""
    n_pixels = len(parents)
    segments = np.zeros(n_pixels, dtype=np.int32)
    
    # Suivre les parents jusqu'aux modes
    for i in range(n_pixels):
        current = i
        visited = []
        
        while current != parents[current]:
            visited.append(current)
            current = parents[current]
            if len(visited) > n_pixels:
                break
        
        for pixel in visited:
            segments[pixel] = current
        segments[i] = current
    
    unique_segments = np.unique(segments)
    segment_map = {old: new for new, old in enumerate(unique_segments)}
    segments = np.array([segment_map[s] for s in segments])
    
    return segments.reshape(h, w)


def segmentation_quickshift(path: str, save: bool = False, visualise: bool = False, out: str = "results/seg_quickshift/") -> dict[str, list[np.ndarray]]:
    """Segmente les images avec Quickshift et retourne les contours détectés."""
    objects_in_images = {}
    images = load_and_resize(path)
    
    for name, img in images.items():
        segments = quickshift(img, kernel_size=3.0, max_dist=10.0, ratio=1.0)
        
        unique_segments = np.unique(segments)
        contours_list = []
        result = img.copy()
        
        for segment_id in unique_segments:
            mask = (segments == segment_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    contours_list.append(cnt)
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 1)
        
        objects_in_images[name] = contours_list
        
        if save:
            os.makedirs(out, exist_ok=True)
            output_path = os.path.join(out, name)
            cv2.imwrite(output_path, result)
            
    return objects_in_images


if __name__ == "__main__":
    path = "results/preprocessing/"
    objects_in_images = segmentation_quickshift(path, save=True, visualise=False)
    print(objects_in_images.keys())
