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
- Flavien :
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

## Analyse critique
## Conclusion




