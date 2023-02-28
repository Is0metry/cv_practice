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
        return cls(cls.path, arr)
    
    @classmethod
    def from_pil(cls,img:Image)-> Self:
        new_arr = np.asarray(img)
        new_arr = cv.cvtColor(new_arr,cv.COLOR_BGR2RGB)
        return cls(arr=new_arr)
    
    def show(self,scale:Union[float,None]=None) -> Image:
        if scale is not None:
            self.scale = scale
        new_img = cv.cvtColor(self.img(), cv.COLOR_BGR2RGB)
        return Image.fromarray(new_img)

    def img(self, scale: Union[float, None] = None) -> np.ndarray:
        if scale is not None:
            self.scale = scale
        image = deepcopy(self.arr)
        img_x = round(self.scale * self.arr.shape[1])
        img_y = round(self.scale * self.arr.shape[0])
        return cv.resize(image, (img_x, img_y), interpolation=cv.INTER_LINEAR)

    def draw_squares(self, detected: np.ndarray) -> Self:
        image = deepcopy(self.arr)
        for x, y, w, h in detected:
            image = cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 5)
        return Img(self.path, image, self.scale)

    def img_bw(self) -> np.ndarray:
        new_arr = deepcopy(self.arr)
        return cv.cvtColor(new_arr,cv.COLOR_BGR2GRAY)

    def blur(self, kx: int=50, ky: int=50) -> Self:
        new_arr = deepcopy(self.arr)
        new_arr = cv.blur(new_arr, (kx, ky), cv.BORDER_DEFAULT)
        return Img(self.path, new_arr, self.scale)
    
    def blur_faces(self,faces:np.ndarray):
        blurred_arr = self.blur(300,300).arr
        arr = deepcopy(self.arr)
        mask = np.zeros((arr.shape[0], arr.shape[1],arr.shape[2]),dtype=np.uint8)
        for x, y, w,h, in faces:
            r = (w+h)//4
            mask = cv.circle(mask,(x+r,y+r),r, (255,255,255),cv.FILLED)
        arr = np.where(mask > 0,blurred_arr,arr)
        return Img(self.path, arr, self.scale)
        
            
        