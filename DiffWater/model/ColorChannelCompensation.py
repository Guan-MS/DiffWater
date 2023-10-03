from skimage.color import rgb2lab, rgb2gray, lab2rgb
import numpy as np
import cv2

def three_c(img_rgb):
    #img_rgb = img_rgb.astype(np.float) / 255.0

    img_lab = rgb2lab(img_rgb)
    img_gray = rgb2gray(img_rgb)
    img_gray[img_gray > 0.847] = 0
    img_gray[img_gray <= 0.847] = 1

    gray_gauss = cv2.GaussianBlur(img_gray, (0, 0), 5)
    ch1, ch2, ch3 = cv2.split(img_lab)
    ch2_gauss = cv2.GaussianBlur(ch2, (0, 0), 100)
    img_lab[:, :, 1] = np.subtract(ch2, np.multiply(gray_gauss, ch2_gauss))

    ch3_gauss = cv2.GaussianBlur(ch3, (0, 0), 100)
    img_lab[:, :, 2] = np.subtract(ch3, np.multiply(gray_gauss, ch3_gauss))

    new = lab2rgb(img_lab)

    return new