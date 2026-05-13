import cv2 as cv
from src.dehazer import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def main():
    image_path = "data/input/hazy/forest1.jpg"
    img = cv.imread(image_path)
    img = cv.resize(img,(500,500))

    dark_channel = compute_dark_channel(img,15)
    print(f"Type: {dark_channel.dtype}, Shape: {dark_channel.shape}")


    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Image originale")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(dark_channel, cmap='gray')
    plt.title("Dark Channel")
    plt.axis('off')

    plt.show()

    brigthest_pixels, A = estimate_atmospheric_light(img,dark_channel)

    plt.subplot(1, 1, 1)
    plt.imshow(dark_channel, cmap='gray')
    plt.title("Brightest pixels of dark channel")
    plt.axis('off')

    for _, i, j in brigthest_pixels:
        rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=1, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)

    plt.show()

    t = transmission_estimation(img,A,0.95,5)
    t = np.maximum(t, 0.1)
    
    print(A)

    #Test sans soft matting
    h,w,_ = np.shape(img)

    img_float = img.astype(np.float64) 
    J = np.zeros((h, w, 3), dtype=np.float64)

    for c in range(3):
        # Application de la formule : J(x) = (I(x) - A) / t(x) + A
        J[:, :, c] = (img_float[:, :, c] - A[c]) / t + A[c]
    
    brightness = 5
    # Adjusts the contrast by scaling the pixel values by 2.3
    contrast = 1
    J = cv.addWeighted(J, contrast, np.zeros(J.shape, J.dtype), 0, brightness)

    plt.subplot(1, 3, 1)
    test_uint8 = np.clip(J, 0, 255).astype(np.uint8)
    plt.imshow(cv.cvtColor(test_uint8, cv.COLOR_BGR2RGB))
    plt.title("Reconstruced image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(t, cmap='gray')
    plt.title("Estimated")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Image originale")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()