# Vision Artificielle - TP2
> Clément Auvray & Anna-Eve Mercier & Flavien Baron & Ewan Schwaller & Laure Warlop
## Introduction
<p align="justify">
L'objectif du TP2 est de réaliser une segmentation.
La mission globale est d'isoler chaque grain.
Ce projet a été réalisé en groupe de cinq étudoants avec une répartition des tâches.
</p>

- Laure : Analyse des données et prétraitement global
- Clément : Segmentation par opérations morphologiques et seuillage Otsu
- Flavien : Segmentation par quickshift
- Ewan :
- Anna-Eve : Segmentation par watershed

## Description des données
<p align="justify">
Les données que nous traitons dans ce sujet sont des images représentant des sous-parties d'un échantillon de roche broyée en grain. Chaque grain correspond à un minéral.
</p>

## Prétraitements testés
Pour cette partie, l'objectif est d'améliorer la qualité des images pour faciliter la segmentation.\
Dans un premier temps, les images sont chargées automatiquement à l'aide de la fonction **load_and_resize**. Elle permet aussi de pouvoir les rédimensionner si besoin pour un traitement cohérent.\
Ensuite, avec la fonction **preprocess**, les images sont converties de l'espace BGR vers l'espace LAB. On fait ce choix car on peut séparer la luminance (le canal L) ainsi que des informations de couleur ce qui va nous permettre d'améliorer le contraste sans modifier les teintes originales.\
Avec CLAHE, une méthode d'égalisation adaptative d'un histogramme, on l'applique sur le canal de luminance. Cette technique limite la surexposition locale tout en augmentant la dynamique de l'image contrairement à une égalisation gloable.\
Pour pouvoir réduire le bruit et lisser légérement l'image, on réalise un filtre gaussien qui va contribuer à stabiliser les futures méthodes de segmentation.\
Enfin, on trace les histogrammes avant et après le traitement pour vérifier que le contraste ce soit bien amélioré ainsi que la bonne distribution des intensités lumineuses.\
Pour bien comparer, on peut affiche côte à côte les images originales et les images prétraitées.\
Toutes ces étapes sont regroupées dans une fonction globale appelée **traitement** pour faciliter l'utilisation dans le *main.py* qui sera le script principal de ce TP.

## Comparaison des méthodes de segmentation
### Segmentation par opérations morphologiques et seuillage Otsu
Pour cette méthode, l'objectif est de détecter et isoler les grains en utilisant des techniques classiques de traitement d'image.\
Dans un premier temps, les images prétraitées sont converties en niveaux de gris dans la fonction **segmentation_basic**. Ensuite, on applique un seuillage automatique par la méthode d'Otsu.\
Pour améliorer la qualité de la segmentation, on utilise plusieurs opérations morphologiques avec un noyau structurant de taille 5x5. On commence par une fermeture (closing) avec 10 itérations pour combler les petits trous à l'intérieur des grains. Puis, on applique une ouverture (opening) suivie d'une fermeture pour lisser les contours et éliminer le bruit tout en préservant la forme des objets.\
Enfin, on effectue une érosion avec 3 itérations pour séparer les grains qui seraient collés entre eux. La détection des contours est réalisée avec **findContours** en mode RETR_EXTERNAL pour ne récupérer que les contours externes.\
Pour éliminer les artefacts et le bruit résiduel, seuls les objets ayant une surface supérieure à 200 pixels sont conservés. Cette valeur seuil permet de supprimer les petits objets indésirables tout en gardant les grains d'intérêt.\
Les contours détectés peuvent être visualisés en traçant des rectangles englobants autour de chaque grain identifié, ce qui permet de vérifier visuellement la qualité de la segmentation.

Bien que cette méthode soit simple à implémenter et rapide à exécuter, elle présente plusieurs limitations importantes.\
Le seuillage d'Otsu, bien qu'automatique, n'est pas optimal lorsque les grains ont des intensités variables ou que l'image est floue. Cette méthode ne s'adapte pas aux variations locales de luminosité dans l'image.\
Les opérations morphologiques, même avec des paramètres optimisés, ont du mal à séparer correctement les grains qui se touchent ou se chevauchent. L'érosion permet de séparer certains grains, mais fragmente et ou supprime les grains de forme allongée ou irrégulière.\
De plus, le choix des paramètres (taille du noyau, nombre d'itérations, seuil de surface) est empirique et nécessite un ajustement manuel très long et laborieux. Cette approche manque de robustesse face à des variations importantes dans les images.\
Globalement, la méthode ne permet pas de distinguer les grains fusionnés et peut sur-segmenter les grains présentant des variations d'intensité interne, ce qui limite sa précision pour l'analyse quantitative des échantillons.

### Segmenteation Watershed
Pour cette méthode, l'objectif est de détecter et isoler les grains en exploitant la topographie de l'image, en simulant une inondation depuis des marqueurs placés au cœur de chaque grain. Dans un premier temps, les images prétraitées par CLAHE sont converties en niveaux de gris dans la fonction watershed_segmentation. Ensuite, un double seuillage est appliqué : un seuillage automatique par la méthode d'Otsu ainsi qu'un seuillage à 60% de la valeur d'Otsu, dont l'union permet de mieux capturer les grains sombres que le seuillage seul ne détecterait pas.

Pour améliorer la qualité du masque binaire, des opérations morphologiques sont appliquées avec un noyau structurant de taille 3x3. Une fermeture avec 3 itérations comble les reflets et trous internes aux grains, suivie d'une ouverture avec 1 itération pour éliminer le bruit résiduel. La distance transform est ensuite calculée sur ce masque, permettant d'estimer la taille médiane des grains et d'en dériver automatiquement le paramètre min_distance utilisé pour la détection des maxima locaux via peak_local_max. Ces maxima constituent les marqueurs du Watershed, qui sont ensuite propagés par cv2.watershed pour délimiter chaque grain. La détection des contours est réalisée avec findContours en mode RETR_EXTERNAL, et seuls les objets ayant une surface comprise entre 200 et 60 000 pixels sont conservés pour éliminer le bruit et les amas trop importants.

Bien que cette méthode soit plus sophistiquée que le simple seuillage morphologique, elle présente des limitations importantes dans le cadre de ce projet. Le CLAHE appliqué en prétraitement améliore le contraste global mais introduit des variations d'intensité à l'intérieur des grains colorés ou à reflets, ce qui génère de faux creux dans la distance transform et produit des marqueurs parasites responsables de sur-segmentation. À l'inverse, pour les grains sombres collés, le prétraitement ne fait pas apparaître de frontière visible entre deux grains de teinte similaire, privant ainsi le Watershed de l'information nécessaire pour les séparer et provoquant de la sous-segmentation. Le paramètre min_distance, bien que calculé dynamiquement, reste un compromis global appliqué uniformément à toute l'image, alors que les grains présentent des tailles très variables d'une image à l'autre. Globalement, la méthode est efficace sur les grains bien contrastés et isolés, mais échoue précisément sur les cas les plus fréquents et les plus difficiles de ces images, ce qui limite sa robustesse pour une analyse quantitative fiable.

### Segmentation Quickshift

Pour cette méthode, l'objectif est de segmenter les grains en regroupant les pixels similaires dans l'espace couleur et spatial selon un principe de mode-seeking, c'est-à-dire en cherchant des modes de densité locale.
Dans un premier temps, les images sont converties de l'espace RGB vers l'espace LAB dans la fonction quickshift. Ce choix permet de travailler dans un espace perceptuellement uniforme où les distances euclidiennes correspondent mieux à la perception visuelle humaine des différences de couleur.
L'algorithme procède ensuite en deux étapes principales. La première étape consiste à calculer la densité locale de chaque pixel en parcourant son voisinage spatial défini par le paramètre **max_dist**. Pour chaque pixel, on calcule une distance combinée prenant en compte à la fois la distance spatiale (pondérée par **ratio**) et la distance couleur dans l'espace LAB. La densité est alors calculée comme une somme pondérée des contributions des voisins, avec un poids gaussien déterminé par le paramètre **kernel_size**. Plus ce dernier est élevé, moins l'algorithme est sensible aux variations locales et moins il crée de segments.
La seconde étape consiste à construire un graphe de liens parents. Pour chaque pixel, on recherche dans son voisinage le pixel ayant la densité la plus élevée et situé à la distance totale minimale. Ce pixel devient le parent du pixel courant. Cette procédure crée des chemins qui remontent vers les modes locaux de densité, c'est-à-dire les centres des régions homogènes. Les pixels qui sont leurs propres parents correspondent aux modes et définissent le cœur des segments.
Enfin, la création des segments s'effectue en suivant récursivement les liens parents depuis chaque pixel jusqu'à atteindre un mode. Tous les pixels qui convergent vers le même mode sont assignés au même segment. Les segments sont ensuite réindexés pour obtenir des identifiants consécutifs, et les contours sont extraits avec **findContours**. Seuls les objets ayant une surface supérieure à 200 pixels sont conservés pour éliminer le bruit.
Le paramètre **ratio** permet de contrôler l'équilibre entre similarité spatiale et similarité couleur : une valeur faible privilégie la proximité de couleur, tandis qu'une valeur élevée favorise la proximité spatiale. Dans l'implémentation actuelle, les valeurs utilisées sont kernel_size=7.0, max_dist=10.0 et ratio=0.3.

Cette méthode présente plusieurs avantages par rapport aux approches plus classiques. Elle ne nécessite pas de seuillage préalable et s'adapte automatiquement aux variations locales de contraste et de couleur dans l'image. L'algorithme est capable de détecter des régions complexes avec des formes irrégulières sans faire d'hypothèse a priori sur la forme des objets. De plus, l'utilisation de l'espace LAB permet de mieux capturer les différences de teinte entre grains, même lorsque leur luminosité est similaire.\
Cependant, l'implémentation présente des limitations majeures dans le contexte de ce projet. Le problème principal observé est une sur-segmentation massive : l'algorithme crée beaucoup trop de segments à l'intérieur d'un même grain, fragmentant ainsi des objets qui devraient être considérés comme uniques. Ce phénomène affecte également le fond de l'image, qui est divisé en multiples régions distinctes au lieu d'être reconnu comme une seule zone homogène. Cette sensibilité excessive aux variations locales de texture et de couleur rend les résultats inexploitables pour notre objectif d'identification et de comptage des grains.\
En comparant nos résultats avec ceux présentés dans la documentation officielle de scikit-image (https://scikit-image.org/docs/0.12.x/auto_examples/segmentation/plot_segmentations.html?highlight=segmentation), on constate que ce comportement est en réalité typique de l'algorithme Quickshift. Les exemples de la documentation montrent que la méthode est particulièrement adaptée pour la segmentation d'images naturelles complexes (paysages, photos) où l'on cherche à identifier de nombreuses régions texturées distinctes, mais qu'elle produit systématiquement un nombre très élevé de segments, même sur des zones relativement homogènes.\
Malgré les nombreux ajustements tentés sur les paramètres (augmentation du kernel_size, modification du ratio, variation du max_dist), il n'a pas été possible d'obtenir une segmentation satisfaisante pour notre cas d'usage. L'algorithme Quickshift, bien que puissant dans son principe, ne semble donc pas adapté à notre problématique spécifique de segmentation de grains minéraux où l'on recherche des objets individuels bien délimités plutôt qu'une partition fine et texturée de l'image.

## Analyse critique
## Conclusion




