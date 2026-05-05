import cv2 as cv
from src.dehazer import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main():
    image_path = "data/input/GT/13_outdoor_GT.jpg"
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
    print(A)

    plt.subplot(1, 1, 1)
    plt.imshow(dark_channel, cmap='gray')
    plt.title("Brightest pixels of dark channel")
    plt.axis('off')

    for _, i, j in brigthest_pixels:
        rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=1, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)

    plt.show()

    t = transmission_estimation(img,A,0.95,15)

    #Test sans soft matting
    h,w,_ = np.shape(img)
    test = img.copy()   
    for c in range(3) : 
        test[:,:,c] =  np.divide(img[:,:,c] - A[c],np.maximum(t,0.1)) + A[c]

    plt.subplot(1, 1, 1)
    plt.imshow(cv.cvtColor(test, cv.COLOR_BGR2RGB))
    plt.title("Recovered scene radiance")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()