import numpy as np

def compute_dark_channel(image,taille_voisinage) :
    h,w,_ = np.shape(image)
    print(h)
    print(w)
    dark_channel = np.zeros((h,w))
    for i in range(0,h,taille_voisinage) :
        for j in range(0,w,taille_voisinage) :

            # calcul de l'intensité min sur le voisinage
            i_min = max(0,i-taille_voisinage//2)
            i_max = min(h,i+taille_voisinage//2)
            j_min = max(0,j-taille_voisinage//2)
            j_max = min(w,j+taille_voisinage//2)
            intensite_min = np.min(image[i_min:i_max,j_min:j_max,:])
            dark_channel[i_min:i_max,j_min:j_max] = intensite_min

    return dark_channel


