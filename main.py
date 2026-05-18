import argparse
import cv2 as cv
from matplotlib.pylab import beta
from src.dehazer import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage.metrics import structural_similarity as ssim

def main(image_path, omega, t0, lambda_, taille_voisinage, save, show_plots):
    epsilon = 1e-6
    img = cv.imread(image_path)
    h,w,_ = np.shape(img)
    if h > 500 or w > 500 :
        ratio = 500 / max(h,w)
        new_size = (int(w * ratio), int(h * ratio))
        img = cv.resize(img, new_size)

    dark_channel = compute_dark_channel(img,taille_voisinage)
    print(f"Type: {dark_channel.dtype}, Shape: {dark_channel.shape}")

    if show_plots:
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

    if show_plots:
        plt.show()

    t_ = transmission_estimation(img,A,0.95,taille_voisinage)
    
    t = soft_matting(img,t_,3,epsilon,lambda_)
    t = np.maximum(t, 0.1)

    beta = 1.0 
    depth_map = -np.log(t) / beta
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    
    print(A)

    #Test sans soft matting
    h,w,_ = np.shape(img)

    img_float = img.astype(np.float64) 
    J = np.zeros((h, w, 3), dtype=np.float64)

    for c in range(3):
        # Application de la formule : J(x) = (I(x) - A) / t(x) + A
        J[:, :, c] = (img_float[:, :, c] - A[c]) / t + A[c]
    
    brightness = 5
    contrast = 1
    J = cv.addWeighted(J, contrast, np.zeros(J.shape, J.dtype), 0, brightness)
    test_uint8 = np.clip(J, 0, 255).astype(np.uint8)

    if show_plots:
        plt.subplot(1, 2, 2)
        plt.imshow(depth_map_normalized, cmap='hot')
        plt.title("Depth Map")
        plt.axis('off')

        plt.subplot(1, 2, 1)
        plt.imshow(t, cmap='gray')
        plt.title("Estimated")
        plt.axis('off')

        plt.show()

        plt.subplot(1, 2, 1)
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.title("Image originale")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv.cvtColor(test_uint8, cv.COLOR_BGR2RGB))
        plt.title("Reconstruced image")
        plt.axis('off')

        plt.show()

    # Sauvegarde des resultats

    if save :
        output_dir = "data/output"
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        output_restored_path = os.path.join(output_dir, f"{base_name}_restored.jpg")
        cv.imwrite(output_restored_path, test_uint8)

        output_t_path = os.path.join(output_dir, f"{base_name}_transmission.jpg")
        t_uint8 = (t * 255).astype(np.uint8)
        cv.imwrite(output_t_path, t_uint8)

        output_depth_path = os.path.join(output_dir, f"{base_name}_depth.png")
        plt.imsave(output_depth_path, depth_map_normalized, cmap='hot')

    # Score ssim

    gray_originale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_restaured = cv.cvtColor(test_uint8, cv.COLOR_BGR2GRAY)

    (score, diff) = ssim(gray_originale, gray_restaured, full=True)
    diff = (diff * 255).astype("uint8")
    print(f"SSIM score: {score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image")
    parser.add_argument("-w", "--omega", type=float, default=0.95)
    parser.add_argument("-t", "--t0", type=float, default=0.1)
    parser.add_argument("-p", "--patch_size", type=int, default=15)
    parser.add_argument("-l", "--lambda_", type=float, default=1e-4)
    
    parser.add_argument("-save", "--save", type=bool, default=False)
    parser.add_argument("-show", "--show_plots", type=bool, default=False)


    args = parser.parse_args()
    image_path = args.image
    lambda_ = args.lambda_
    taille_voisinage = args.patch_size
    omega = args.omega
    t0 = args.t0
    save = args.save
    show_plots = args.show_plots

    args = parser.parse_args()
    main(image_path, omega, t0, lambda_, taille_voisinage,save,show_plots)