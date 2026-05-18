from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import os

def compute_dark_channel(image,taille_voisinage) :
    # Calcul du dark channel selon l'equation (5) du papier de He et al.
    h,w,_ = np.shape(image)
    dark_channel = np.zeros((h,w))
    for i in range(0,h) :
        for j in range(0,w) :
            # calcul de l'intensité min sur le voisinage
            i_min = max(0,i-taille_voisinage//2)
            i_max = min(h,i+taille_voisinage//2)
            j_min = max(0,j-taille_voisinage//2)
            j_max = min(w,j+taille_voisinage//2)
            intensite_min = np.min(image[i_min:i_max,j_min:j_max,:])
            dark_channel[i,j] = intensite_min
    return dark_channel


def estimate_atmospheric_light(image, dark_channel):
    # Estimation de la lumière atmosphérique selon la méthode du papier de He et al. (0.1% brightest pixels in the dark channe)
    h,w,_ = np.shape(image)

    #top 0.1% brightest pixel
    nb_brightest = max(round(h*w*0.001), 1)
    brightest_pixels = []
    for i in range(0,h) :
        for j in range(0,w) :
            dark_channel_ij = dark_channel[i,j]
            if len(brightest_pixels)< nb_brightest :
                brightest_pixels.append([dark_channel_ij,i,j])
                brightest_pixels.sort(reverse=True)
            elif (dark_channel_ij > brightest_pixels[-1][0]) :
                # insertion dans liste triée
                for ii in range(nb_brightest):
                    if dark_channel_ij > brightest_pixels[ii][0]:
                        # print("insert at : "+str(ii))
                        brightest_pixels.insert(ii, [dark_channel_ij,i,j])
                        break
                brightest_pixels.pop()

    #pixel with highest intensity
    I_max = 0
    for p in brightest_pixels :
        c = image[p[1],p[2]]
        I = int(c[0])+int(c[1])+int(c[2])
        if I > I_max :
            I_max = I
            A = c

    return brightest_pixels, A

def transmission_estimation(image,A,omega,taille_voisinage):
    # Estimation de la transmission selon l'equation (12) du papier de He et al.
    h,w,_ = np.shape(image)
    t = np.zeros((h,w))
    nb_patches_h = math.ceil(h/taille_voisinage)
    nb_patches_w = math.ceil(w/taille_voisinage)
    for p_i in range(h) :
        for p_j in range(w) :
            y_min = -1
            i_min = max(0,round(p_i - taille_voisinage/2))
            i_max = min(round(p_i+taille_voisinage/2),h)
            j_min = max(0,round(p_j - taille_voisinage/2))
            j_max = min(round(p_j+taille_voisinage/2),w)
            for c in range(3) :
                y = np.min(image[i_min:i_max,j_min:j_max,c])
                y_c = y/A[c]
                if y_c < y_min or y_min == -1 :
                    y_min = y_c
            t[i_min:i_max,j_min:j_max] = 1-omega*y_min
    
    return t

def soft_matting(image, t, taille_voisinage,epsilon,lambda_):
    # Raffinage de la transmission par soft matting selon les equations (14) et (15) du papier de He et al.
    # image = image.astype(np.float64) / 255.
    
    #  initialisation des listes pour construire la matrice L (sparse)
    i_indices = []
    j_indices = []
    valeurs = []

    h,w,_ = np.shape(image)
    vect_image = image.reshape(-1,3)

    for h_k in range(h) :
        for w_k in range(w) :

            h_min_k = max(0,round(h_k- taille_voisinage/2))
            h_max_k = min(round(h_k+taille_voisinage/2),h)
            w_min_k = max(0,round(w_k - taille_voisinage/2))
            w_max_k = min(round(w_k+taille_voisinage/2),w)

            window_pixels = image[h_min_k:h_max_k, w_min_k:w_max_k, :].reshape(-1, 3)
            # estimation de la moyenne et de la covariance des pixels dans le voisinage k
            mu_k = np.mean(window_pixels, axis=0)
            sigma_k = np.cov(window_pixels, rowvar=False, bias=True)

            card_k = (h_max_k - h_min_k) * (w_max_k - w_min_k)

            # calcul de toutes les paires de pixels dans le voisinage k
            y_coords, x_coords = np.mgrid[h_min_k:h_max_k, w_min_k:w_max_k]
            indices_1d_k = (y_coords * w + x_coords).flatten()
            i_matrix, j_matrix = np.meshgrid(indices_1d_k, indices_1d_k, indexing='ij')
            i_pairs = i_matrix.flatten()
            j_pairs = j_matrix.flatten()

            # calcul matriciel des L[i,j] (valeurs de L pour les paires de pixels i,j du voisinage k)
            inv_sigma = np.linalg.inv(sigma_k + (epsilon / card_k) * np.eye(3))
            diff_I = window_pixels - mu_k
            L_window_terme = diff_I @ inv_sigma @ diff_I.T
            delta_window = np.eye(card_k)
            L_window = delta_window - (1.0 / card_k) * (1.0 + L_window_terme)

            # ajout des valeurs de L_window dans les listes pour construire la matrice L
            i_indices.extend(i_pairs)
            j_indices.extend(j_pairs)
            valeurs.extend(L_window.flatten())

            print((h_k*w+w_k)/(h*w))

    # construction de la matrice L et résolution du système
    L = scipy.sparse.coo_matrix((valeurs, (i_indices, j_indices)), shape=(h*w, h*w))
    print("matrice L construite")
    t_refined = scipy.sparse.linalg.cg(L+lambda_*scipy.sparse.eye(h*w), lambda_*t.flatten(), rtol=1e-6, maxiter=1000)
    print("Soft-matting effectué")
    return t_refined[0].reshape(h,w)