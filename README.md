# PRE_2024
This repository contains all the code used for my Research Project at ENSTA Paris.

The main jupyter notebook is located in Image-Level-Processing/pca_ood.ipynb

L'objectif du projet est de proposer des espaces de représentation permettant de caractériser
l'environnement dans lequel évolue un robot mobile (p. ex. : environnement routier, chemin,
forêt, etc.), dans le but d'adapter automatiquement un algorithme de vision par ordinateur
(détection, prédiction de profondeur, etc.) au contexte courant.
Pour cela, on se propose d'adapter les techniques métriques de détection d'anomalies [1] fondées
sur les mesures de normes dans les espaces résiduels.
Une de ces techniques consiste à utiliser l'espace supplémentaire à celui engendré par les
premières composantes principales calculées, soit directement à partir de patches d'images
échantillonnés dans chaque environnement, soit à partir d'un espace latent de type deep
features. Cette option est faiblement supervisée car elle ne nécessite pas d'entraîner un
classificateur sur les différents environnements.
On expérimentera dans un premier temps sur la base de données CBIR_15-Scene [2] dédiée à la
catégorisation d'environnements 
