import os
from dataclasses import dataclass, field
from typing import List, Self, Union
from copy import deepcopy
import cv2 as cv
import numpy as np
from PIL import Image

IMG_PATH = 'images/'


@dataclass
class Img:
    path: str = IMG_PATH
    arr: np.ndarray = field(default_factory=np.ndarray([]))
    scale: float = .5

    @classmethod
    def list(cls) -> List[str]:
        return [p.split('.')[0] for p in os.listdir(cls.path)]
    
    @classmethod
    def load(cls, img: Union[str, int] = 0) -> Self:
        if isinstance(img, int):
            img = cls.list()[img]
        arr = cv.imread(cls.path + img + '.png')
        return cls(cls.path,arr)

    def show(self) -> Image:
        new_img = cv.cvtColor(self.img(), cv.COLOR_BGR2RGB)
        return Image.fromarray(new_img)

    def img(self, scale: Union[float, None] = None) -> np.ndarray:
        if scale is not None:
            self.scale = scale
        image = deepcopy(self.arr)
        img_x = round(self.scale * self.arr.shape[1])
        img_y = round(self.scale * self.arr.shape[0])
        return cv.resize(image, (img_x, img_y), interpolation=cv.INTER_LINEAR)


    def draw(self, detected: np.ndarray) -> Self:
        image = deepcopy(self.arr)
        for x, y, w, h in detected:
            image = cv.rectangle(image, (x, y), (x+w, y+h),(255,0,0),5)
        return Img(self.path,image,self.scale)
    
    def img_bw(self)->np.ndarray:
        new_arr = deepcopy(self.arr)
        return cv.cvtColor(new_arr, cv.COLOR_BGR2GRAY)
        