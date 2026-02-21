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
- Anna-Eve :

## Description des données
<p align="justify">
Les données que nous traitons dans ce sujet sont des images représentant des sous-parties d'un échantillon de roche broyée en grain. Chaque grain correspond à un minéral.
</p>

## Prétraitements testés
Pour cette partie, l'objectif est d'améliorer la qualité des images pour faciliter la segmentation.\
Dans un premier temps, les images sont chargées automatiquement à l'aide de la fonction **load_and_resize**. Elle permet aussi de pouvoir les rédimensionner si besoin pour un traitement cohérent.\
Ensuite, avec la fonction **preprocess**, les images sont converties de l'espace BGR vers l'espace LAB. On fait ce choix car on peut séparer la luminance (le canal L) ainsi que des informations de couleur ce qui va nous permettre d'améliorer le contraste sans modifier les teientes originales.\
Avec CLAHE, une méthode d'égalisation adaptative d'un hid=stogramme, on l'applique sur le canal de luminance. Cette technique limite la surexposition locale tout en augmentant la dynamique de l'image contrairement à une égalisation gloable.\
Pour pouvoir réduire le bruit et lisser légérement l'image, on réalise un filte gaussien qui va contrbuer à stabiliser les futures méthodes de segmentation.\
Enfin, on trace les histogrammes avant et après le traitement pour vérifier que le contraste ce soit bien amélioré ainsi que la bonne distribution des intensités lumineuses.\
Pour bien comparer, on affiche côte à côte les images originales et les images prétraitées.\
Toutes ces étapes sont regroupées dans une fonction globale appelée **traitement** pour faciliter l'utilisation dans le *main.py* qui sera le script principal de ce TP.

Voici un exemple d'une comparation entre une image originale et l'image après le prétraitement.
![original-vs-pretraitee-Mod2_301](/results/Echantillion1Mod2_301.png)

## Comparaison des méthodes de segmentation
### Segmentation par opérations morphologiques et seuillage Otsu
Pour cette méthode, l'objectif est de détecter et isoler les grains en utilisant des techniques classiques de traitement d'image.\
Dans un premier temps, les images prétraitées sont converties en niveaux de gris avec la fonction **segmentation_basic**. Ensuite, on applique un seuillage automatique par la méthode d'Otsu.\
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
## Analyse critique
## Conclusion



