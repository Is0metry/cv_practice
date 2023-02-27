import numpy as np
import cv2 as cv
from PIL import Image


def display_image(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    new_img = img
    if scale != 1.0:
        new_img = img.copy()
        img_x = round(scale * img.shape[1])
        img_y = round(scale * img.shape[0])
        new_img = cv.resize(new_img, (img_x, img_y), interpolation=cv.INTER_LINEAR)
    new_img = cv.cvtColor(new_img, cv.COLOR_BGR2RGB)
    return Image.fromarray(new_img)

def resize_image(img:np.ndarray,scale=.75)->np.ndarray:
    image = img.copy()
    img_x = round(scale * img.shape[1])
    img_y = round(scale * img.shape[0])
    return cv.resize(image, (img_x, img_y), interpolation=cv.INTER_LINEAR)
