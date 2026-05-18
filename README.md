# Single Image Haze Removal

## Description
Ce script est une implémentation python de la méthode de dehazing présenté par K.He & al dans la publication "Single Image Haze Removal Using Dark Channel Prior". Cette méthode a pour objectif de supprimer le brouillard d'une image en utilisant une méthode originale nommée Dark Channel Prior. Ce code permet à partir d'une image présentant du brouillard de reconstruire l'image claire, estimer une carte de transmission affinee par soft matting et de generer une carte de profondeur relative.

## Dependances requises
python 3.x
opencv-python (cv2)
matplotlib
numpy
scikit-image (skimage)
scipy

## Utilisation
le script a executer est main.py, le fichier src/dehazer contient les différentes fonctions pour chaque étape du dehazing
python main.py -i chemin/vers/image.jpg [options]

### Arguements
-i, --image : Chemin de l'image a traiter (Obligatoire)
-w, --omega : Facteur de conservation du brouillard (Defaut: 0.95)
-t, --t0 : Borne inferieure de la transmission pour limiter le bruit (Defaut: 0.1)
-p, --patch_size : Taille du voisinage pour le Dark Channel (Defaut: 15)
-l, --lambda_ : Poids de regularisation pour le Soft Matting (Defaut: 0.0001)
-save, --save : Indiquer True pour sauvegarder les resultats (Defaut: False)
-show, --show_plots : Indiquer True pour afficher les fenetres graphiques (Defaut: False)

### Exemples de commandes
Execution simple :
python main.py -i data/input/hazy/forest1.jpg

Execution avec sauvegarde et affichage des etapes visuelles :
python main.py -i data/input/hazy/forest1.jpg -show True -save True

### Fichiers de sortie (si sauvegarde activee)
Les resultats sont automatiquement places dans le dossier data/output/ :
[nom]_restored.jpg : L'image finale restauree
[nom]_transmission.jpg : La carte de transmission (niveaux de gris)
[nom]_depth.png : La carte de profondeur (colormap hot)
