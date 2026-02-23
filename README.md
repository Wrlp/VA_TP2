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
- Ewan : Extraction des caractéristiques et création du DataFrame final
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
Pour cette méthode, l'objectif est de détecter et isoler les grains en exploitant la topographie de l'image, en simulant une inondation depuis des marqueurs placés au cœur de chaque grain. Dans un premier temps, les images prétraitées par CLAHE sont converties en niveaux de gris dans la fonction watershed_segmentation. Un flou gaussien avec un noyau 5x5 est ensuite appliqué pour réduire le bruit haute fréquence avant le seuillage. Le seuillage binaire repose uniquement sur la méthode d'Otsu, qui détermine automatiquement le seuil optimal selon la distribution des intensités. Pour améliorer la qualité du masque binaire, des opérations morphologiques sont appliquées avec deux noyaux structurants distincts. Une fermeture avec un noyau 5x5 sur 4 itérations comble les reflets et trous internes aux grains de manière plus agressive, suivie d'une ouverture avec un noyau 3x3 sur 1 itération pour éliminer le bruit résiduel sans déformer les contours.
La distance transform est ensuite calculée sur ce masque, permettant d'estimer la taille médiane des grains via une détection de composantes connexes sur un seuil à 35% du maximum. Le rayon estimé dérive automatiquement le paramètre min_distance (multiplié par un facteur 1.6 pour éviter les doubles pics) utilisé pour la détection des maxima locaux via peak_local_max avec un seuil relatif de 0.3. Ces maxima constituent les marqueurs du Watershed, propagés ensuite par cv2.watershed pour délimiter chaque grain. La détection des contours est réalisée avec findContours en mode RETR_EXTERNAL, et seuls les objets ayant une surface comprise entre 200 et 60 000 pixels sont conservés. Un post-traitement via filter_overlapping_contours fusionne enfin les contours dont les centroïdes sont trop proches, réduisant la sur-segmentation résiduelle.

Cette version du code est le résultat de plusieurs itérations successives. Les premières versions présentaient des défauts bien plus marqués : une sur-segmentation sévère générait des contours à l'intérieur même des grains, tandis qu'une sous-segmentation simultanée fusionnait des groupes entiers de grains en un seul objet détecté. Ces deux phénomènes coexistaient dans la même image, rendant le résultat inexploitable quantitativement. Le code présenté ici est le plus concluant obtenu après ces tests.

Malgré nos ajustements, deux problèmes persistent de manière systématique. D'une part, une sur-segmentation partielle subsiste : certaines roches sont entourées plusieurs fois par des contours distincts, et pour certaines pierres de grande taille ou à texture hétérogène, plusieurs détections sont générées là où il n'y a qu'un seul grain réel. Cela s'explique par le fait que le CLAHE appliqué en prétraitement introduit des variations d'intensité locales à l'intérieur des grains colorés ou à reflets, créant de faux creux dans la distance transform qui génèrent des marqueurs parasites supplémentaires. Le filtre filter_overlapping_contours atténue ce phénomène mais ne l'élimine pas complètement, car il fusionne sur la base de la distance entre centroïdes, ce qui ne couvre pas tous les cas de chevauchement.
D'autre part, et de façon paradoxale, le compte final est généralement inférieur d'environ 5 grains au comptage réel. Cela s'explique par la combinaison de deux effets contraires. Les fragments issus de la sur-segmentation sont souvent trop petits pour passer le filtre 200 < area < 60000, donc éliminés, ce qui réduit le compte. Simultanément, des grains réels sont manqués : le threshold_rel=0.3 dans peak_local_max, bien qu'abaissé par rapport aux versions précédentes, élimine encore les pics de faible amplitude. De plus le paramètre min_distance, calculé dynamiquement mais appliqué uniformément à toute l'image, constitue également un compromis global inadapté aux images présentant une forte variabilité de taille de grains.

Globalement, la méthode est efficace sur les grains bien contrastés et isolés, mais reste insuffisante sur les cas les plus fréquents et les plus difficiles de ces images. La dépendance forte au prétraitement CLAHE, qui améliore le contraste global au prix d'artefacts locaux, constitue la limitation fondamentale de cette approche pour une analyse quantitative fiable.

### Segmentation Quickshift

Pour cette méthode, le but est de segmenter les grains en regroupant les pixels similaires dans l'espace couleur et spatial selon un principe de mode-seeking, c'est-à-dire en cherchant des modes de densité locale.
Dans un premier temps, les images sont converties de l'espace RGB vers l'espace LAB. Ce choix permet de travailler dans un espace où les distances euclidiennes correspondent mieux à la perception visuelle humaine des différences de couleur.
L'algorithme et composé de deux étapes principales. La première étape consiste à calculer la densité locale de chaque pixel en parcourant son voisinage spatial défini par le paramètre **max_dist**. Pour chaque pixel, on calcule une distance combinée prenant en compte à la fois la distance spatiale (pondérée par **ratio**) et la distance couleur dans l'espace LAB. La densité est alors calculée comme une somme pondérée des contributions des voisins, avec un poids gaussien déterminé par le paramètre **kernel_size**. Plus ce dernier est élevé, moins l'algorithme est sensible aux variations locales et moins il va créer de segments.
La seconde étape consiste à construire un graphe de liens parents. Pour chaque pixel, on recherche dans son voisinage le pixel ayant la densité la plus élevée et situé à la distance totale minimale. Ce pixel devient le parent du pixel courant. Cette procédure crée des chemins qui remontent vers les modes locaux de densité, c'est-à-dire les centres des régions. Les pixels qui sont leurs propres parents correspondent aux modes et définissent le cœur des segments.
Enfin, la création des segments s'effectue en suivant récursivement les liens parents depuis chaque pixel jusqu'à atteindre un mode. Tous les pixels qui convergent vers le même mode sont assignés au même segment. Les segments sont ensuite réindexés pour obtenir des identifiants consécutifs. Seuls les objets ayant une surface supérieure à 200 pixels sont conservés pour éliminer le bruit.
Le paramètre **ratio** permet de contrôler l'équilibre entre similarité spatiale et similarité couleur : une valeur faible privilégie la proximité de couleur, tandis qu'une valeur élevée favorise la proximité spatiale. Dans l'implémentation actuelle, les valeurs utilisées sont kernel_size=7.0, max_dist=10.0 et ratio=0.3.

Cette méthode présente plusieurs avantages par rapport aux approches plus classiques. Elle ne nécessite pas de seuillage préalable et s'adapte automatiquement aux variations locales de contraste et de couleur dans l'image. L'algorithme est capable de détecter des régions complexes avec des formes irrégulières sans faire d'hypothèse a priori sur la forme des objets. De plus, l'utilisation de l'espace LAB permet de mieux capturer les différences de teinte entre grains, même lorsque leur luminosité est similaire.
Cependant, l'implémentation présente des limitations majeures dans le contexte de ce projet. Le problème principal observé est une sur-segmentation massive : l'algorithme crée beaucoup trop de segments à l'intérieur d'un même grain, fragmentant ainsi des objets qui devraient être considérés comme uniques. Ce phénomène affecte également le fond de l'image, qui est divisé en multiples régions distinctes au lieu d'être reconnu comme une seule zone homogène. Cette sensibilité excessive aux variations locales de texture et de couleur rend les résultats inexploitables pour notre objectif d'identification et de comptage des grains.
En comparant nos résultats avec ceux présentés dans la documentation officielle de scikit-image (https://scikit-image.org/docs/0.12.x/auto_examples/segmentation/plot_segmentations.html?highlight=segmentation), on constate que ce comportement est en réalité typique de l'algorithme Quickshift. Les exemples de la documentation montrent que la méthode est particulièrement adaptée pour la segmentation d'images plus complexes où l'on cherche à identifier de nombreuses régions texturées distinctes, mais qu'elle produit systématiquement un nombre très élevé de segments, même sur des zones relativement homogènes.
Malgré les nombreux ajustements tentés sur les paramètres (augmentation du kernel_size, modification du ratio, variation du max_dist), il n'a pas été possible d'obtenir une segmentation satisfaisante pour notre cas d'usage. L'algorithme Quickshift, bien que puissant dans son principe, ne semble donc pas adapté à notre problématique où l'on recherche des objets individuels bien délimités plutôt qu'une partition fine et texturée de l'image.

### Extraction des caractéristiques et DataFrame final
Pour cette partie, l'objectif est d'extraire des données quantitatives pour chaque grain segmenté afin de permettre une analyse plus poussée des minéraux présents dans l'échantillon.\
Dans un premier temps, la fonction **extract_features** parcourt chaque grain isolé identifié par les étapes de segmentation précédentes. Pour chaque grain, un masque binaire spécifique est créé à partir de son contour, ce qui permet d'isoler précisément les pixels appartenant à ce minéral.\
Ensuite, nous calculons les valeurs moyennes des canaux Bleu (B), Vert (G) et Rouge (R) pour l'ensemble des pixels couverts par le masque. Ces caractéristiques colorimétriques sont essentielles pour identifier la nature des minéraux, chaque type de roche possédant une signature spectrale propre.\
Toutes ces données sont stockées de manière structurée dans une liste de dictionnaires, incluant le nom de l'image source, un identifiant unique pour le grain (Grain_ID) et les trois moyennes de couleur calculées. Ce format permet une manipulation aisée et une conversion rapide vers un DataFrame.\
Le résultat final est exporté dans un fichier CSV (**features.csv**), regroupant l'ensemble des mesures effectuées sur toutes les images traitées. Cet export facilite l'analyse statistique globale et la validation des résultats par rapport aux données de référence.\
Pour assurer la traçabilité et faciliter le débogage, la fonction **visualize_grains** génère des images de contrôle où chaque grain est entouré de son contour avec son identifiant affiché à son centre de masse. Cette étape de visualisation est cruciale pour valider que les caractéristiques extraites correspondent bien aux bons objets et pour identifier d'éventuels cas d'erreurs (grains mal segmentés ou fusionnés).

L'impact de la qualité de la segmentation est direct sur la précision de l'extraction des caractéristiques. Une sous-segmentation (grains fusionnés) entraîne des moyennes de couleur biaisées par le mélange de plusieurs minéraux, tandis qu'une sur-segmentation fragmente les données d'un seul grain en plusieurs entrées, faussant ainsi les statistiques de taille et de répartition. La validation visuelle par ID permet de confirmer que les grains d'intérêt sont correctement capturés et caractérisés avant l'analyse finale.

## Analyse critique
L'analyse comparative des trois méthodes de segmentation révèle des différences significatives tant sur le plan quantitatif que qualitatif. L'exécution complète sur l'ensemble du jeu de données (6 images) pour les méthodes basique et Watershed, ainsi que le test partiel sur Quickshift, nous permettent de tirer les conclusions suivantes :

### Comparaison quantitative
En prenant pour référence l'image **Echantillion-2Mod2_301.png**, nous observons une disparité majeure dans le nombre de grains détectés :
- **Segmentation Basique** : 18 grains détectés.
- **Segmentation Watershed** : 17 grains détectés.
- **Segmentation Quickshift** : 341 grains détectés.

Sur l'ensemble des 6 images, la méthode basique totalise 230 détections (moyenne de ~25,5 par image) contre 202 pour le Watershed (moyenne de ~22,4). Cette différence s'explique par la tendance du Watershed à être plus sélectif via son filtrage de surface et ses paramètres de distance minimale, évitant certaines fausses détections au prix d'une légère sous-segmentation des grains les moins contrastés.

### Analyse des performances et limites
-2. **Efficacité temporelle** : Les méthodes **Basique** et **Watershed** sont extrêmement rapides et adaptées à un traitement par lot. À l'inverse, **Quickshift** s'est avéré excessivement lent, nécessitant plusieurs minutes par image. Cette complexité algorithmique le rend inexploitable pour des applications en temps réel ou sur de grands volumes de données avec notre matériel.

-1. **Qualité de segmentation (Sur-segmentation vs Sous-segmentation)** :
   - La **méthode basique** souffre principalement de sous-segmentation (grains collés non séparés) et d'une perte de précision sur les contours due aux érosions successives.
   - Le **Watershed** offre le meilleur équilibre pour l'isolation des grains individuels grâce à la transformée de distance, bien qu'il reste sensible aux textures internes des minéraux qui génèrent parfois des marqueurs parasites.
   - Le **Quickshift** présente une sur-segmentation massive et incontrôlable dans notre configuration. Chaque grain est fragmenté en dizaines de petits segments, rendant le comptage et l'analyse morphologique impossibles.

0. **Fiabilité des données extraites** : Le fichier **features.csv** généré par Quickshift contient des données bruitées où une seule entité physique (un grain) est représentée par de multiples lignes de caractéristiques colorimétriques. Les fichiers issus des méthodes Basique et Watershed sont beaucoup plus représentatifs de la réalité physique de l'échantillon, permettant une analyse statistique cohérente de la composition minéralogique.

## Conclusion
Ce projet nous a permis d'explorer différentes approches de segmentation pour l'analyse d'échantillons géologiques. 
La méthode par **opérations morphologiques et seuillage d'Otsu** constitue une base solide et rapide, mais manque de finesse pour séparer les amas de grains. 
La **segmentation par Watershed**, bien que plus complexe à paramétrer, s'impose comme la solution la plus robuste pour notre objectif de comptage et de caractérisation individuelle, malgré une légère tendance à la sous-segmentation.
Enfin, l'expérience avec **Quickshift** souligne l'importance du choix de l'algorithme en fonction de la nature des images : bien que puissant pour des images naturelles complexes, il est inadapté à la segmentation d'objets distincts et homogènes comme des grains minéraux, tant pour des raisons de précision que de coût computationnel.
Pour la suite, une approche hybride combinant la précision des contours du Watershed avec un classifieur de machine learning sur les caractéristiques extraites (RGB/LAB) permettrait une identification précise de la nature de chaque minéral.