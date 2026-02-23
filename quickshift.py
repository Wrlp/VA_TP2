import numpy as np
from skimage import color
import cv2
import os
from tqdm import tqdm

from preprocessing import load_and_resize


def quickshift(image: np.ndarray, 
               kernel_size: float = 3.0,
               max_dist: float = 10.0,
               ratio: float = 1.0) -> np.ndarray:
    """Segmentation Quickshift optimisée avec recherche locale."""
    # Conversion en espace Lab
    if len(image.shape) == 3 and image.shape[2] == 3:
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        image_lab = color.rgb2lab(image)
    else:
        image_lab = image.copy()
    
    h, w = image_lab.shape[:2]
    
    # Calcul de la densité locale et des parents de manière optimisée
    densities, parents = compute_density_and_parents_optimized(
        image_lab, h, w, kernel_size, max_dist, ratio
    )
    
    # Création des segments
    segments = create_segments(parents, h, w)
    
    return segments


def compute_density_and_parents_optimized(image_lab, h, w, kernel_size, max_dist, ratio):
    """Calcule densité et parents en une passe avec recherche locale uniquement."""
    densities = np.zeros((h, w), dtype=np.float32)
    parents = np.zeros((h, w, 2), dtype=np.int32)
    
    # Initialiser chaque pixel comme son propre parent
    for y in range(h):
        for x in range(w):
            parents[y, x] = [y, x]
    
    # Rayon de recherche spatial
    search_radius = int(np.ceil(max_dist / ratio)) + 1
    
    # Calcul des densités avec voisinage local
    print("Calcul des densités locales...")
    for y in tqdm(range(h), desc="Densité"):
        for x in range(w):
            # Définir la fenêtre locale
            y_min = max(0, y - search_radius)
            y_max = min(h, y + search_radius + 1)
            x_min = max(0, x - search_radius)
            x_max = min(w, x + search_radius + 1)
            
            # Pixel courant
            current_pixel = image_lab[y, x]
            density = 0.0
            
            # Parcourir les voisins dans la fenêtre
            for ny in range(y_min, y_max):
                for nx in range(x_min, x_max):
                    # Distance spatiale
                    spatial_dist = np.sqrt(((y - ny) * ratio)**2 + ((x - nx) * ratio)**2)
                    
                    if spatial_dist <= max_dist:
                        # Distance couleur
                        color_dist = np.linalg.norm(image_lab[ny, nx] - current_pixel)
                        
                        # Distance totale
                        total_dist = np.sqrt(spatial_dist**2 + color_dist**2)
                        
                        # Contribution à la densité
                        weight = np.exp(-total_dist**2 / (2 * kernel_size**2))
                        density += weight
            
            densities[y, x] = density
    
    # Trouver les parents (voisins avec densité plus élevée)
    print("Recherche des parents...")
    for y in tqdm(range(h), desc="Parents"):
        for x in range(w):
            # Définir la fenêtre locale
            y_min = max(0, y - search_radius)
            y_max = min(h, y + search_radius + 1)
            x_min = max(0, x - search_radius)
            x_max = min(w, x + search_radius + 1)
            
            current_pixel = image_lab[y, x]
            current_density = densities[y, x]
            
            best_parent = (y, x)
            best_distance = np.inf
            
            # Parcourir les voisins
            for ny in range(y_min, y_max):
                for nx in range(x_min, x_max):
                    # Distance spatiale
                    spatial_dist = np.sqrt(((y - ny) * ratio)**2 + ((x - nx) * ratio)**2)
                    
                    if spatial_dist <= max_dist and densities[ny, nx] > current_density:
                        # Distance couleur
                        color_dist = np.linalg.norm(image_lab[ny, nx] - current_pixel)
                        total_dist = np.sqrt(spatial_dist**2 + color_dist**2)
                        
                        if total_dist < best_distance:
                            best_distance = total_dist
                            best_parent = (ny, nx)
            
            parents[y, x] = best_parent
    
    return densities, parents


def create_segments(parents: np.ndarray, h: int, w: int) -> np.ndarray:
    """Crée les segments en suivant les liens parents jusqu'aux modes."""
    segments = np.zeros((h, w), dtype=np.int32)
    
    # Suivre les parents jusqu'aux modes
    print("Création des segments...")
    for y in tqdm(range(h), desc="Segments"):
        for x in range(w):
            # Suivre la chaîne de parents
            cy, cx = y, x
            visited = set()
            path = []
            
            while (cy, cx) not in visited:
                visited.add((cy, cx))
                path.append((cy, cx))
                py, px = parents[cy, cx]
                if py == cy and px == cx:
                    break
                cy, cx = py, px
            
            # Le mode est le dernier pixel atteint
            mode = cy * w + cx
            
            # Assigner tous les pixels du chemin au même segment
            for py, px in path:
                segments[py, px] = mode
    
    # Réindexer les segments pour avoir des indices consécutifs
    unique_segments = np.unique(segments)
    segment_map = {old: new for new, old in enumerate(unique_segments)}
    
    for y in range(h):
        for x in range(w):
            segments[y, x] = segment_map[segments[y, x]]
    
    return segments


def segmentation_quickshift(path: str, save: bool = False, visualise: bool = False, out: str = "results/seg_quickshift/") -> dict[str, list[np.ndarray]]:
    """Segmente les images avec Quickshift et retourne les contours détectés."""
    objects_in_images = {}
    images = load_and_resize(path)
    
    if not images:
        print("Aucune image trouvée")
        return objects_in_images
    
    print(f"\n{'='*60}")
    print(f"Traitement de {len(images)} images avec Quickshift optimisé")
    print(f"{'='*60}\n")
    
    for idx, (name, img) in enumerate(images.items(), 1):
        print(f"\n[{idx}/{len(images)}] Image: {name}")
        print("-" * 60)
        
        segments = quickshift(img, kernel_size=10.0, max_dist=10.0, ratio=1.0)
        
        unique_segments = np.unique(segments)
        print(f"Segments créés: {len(unique_segments)}")
        
        contours_list = []
        result = img.copy()
        
        print("Extraction des contours...")
        for segment_id in tqdm(unique_segments, desc="Contours"):
            mask = (segments == segment_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    contours_list.append(cnt)
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 1)
        
        print(f"Objets détectés: {len(contours_list)}")
        objects_in_images[name] = contours_list
        
        if save:
            os.makedirs(out, exist_ok=True)
            output_path = os.path.join(out, name)
            cv2.imwrite(output_path, result)
            print(f"Sauvegardé: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"Traitement terminé")
    print(f"{'='*60}\n")
    
    return objects_in_images


if __name__ == "__main__":
    path = "results/preprocessing/"
    objects_in_images = segmentation_quickshift(path, save=True, visualise=False)
    print(objects_in_images.keys())
