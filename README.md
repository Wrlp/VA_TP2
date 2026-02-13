# Vision Artificielle - TP2
> Clément Auvray & Anna-Eve Mercier & Flavien Baron & Ewan Schwaller & Laure Warlop
## Introduction
<p align="justify">
L'objectif du TP2 est de réaliser une segmentation.
La mission globale est d'isoler chaque grain.
Ce projet a été réalisé en groupe de cinq étudoants avec une répartition des tâches.
</p>

- Laure : Analyse des données et prétraitement global
- Clément : 
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
## Analyse critique
## Conclusion



