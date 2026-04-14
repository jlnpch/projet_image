import cv2 as cv
from src.dehazer import *

def main():
    image_path = "data/input/GT/13_outdoor_GT.jpg"
    img = cv.imread(image_path)
    img = cv.resize(img,(500,500))
    
    dark_channel = compute_dark_channel(img,15)
    print(f"Type: {dark_channel.dtype}, Shape: {dark_channel.shape}")
    cv.imshow("Dark channel", dark_channel)
    cv.waitKey(0)

if __name__ == "__main__":
    main()