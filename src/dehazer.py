import numpy as np
import matplotlib.pyplot as plt
import math

def compute_dark_channel(image,taille_voisinage) :
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
    h,w,_ = np.shape(image)

    #top 0.1% brightest pixel
    nb_brightest = max(round(h*w*0.001), 1)
    brightest_pixels = []
    for i in range(0,h) :
        for j in range(0,w) :
            dark_channel_ij = dark_channel[i,j]
            # a factoriser someday
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
    h,w,_ = np.shape(image)
    t = np.zeros((h,w))
    nb_patches_h = math.ceil(h/taille_voisinage)
    nb_patches_w = math.ceil(w/taille_voisinage)
    print(nb_patches_h)
    print(nb_patches_w)
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