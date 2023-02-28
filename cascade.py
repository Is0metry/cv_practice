import os
from dataclasses import dataclass
from typing import List, Self, Tuple, Union
from img import Img
from cv2 import CascadeClassifier
import numpy as np


class NotFittedError(AttributeError):
    def __init__(self, *args):
        super().__init__(args)

    def __str__(self):
        return f'Classifier not fitted to cascade'


CAS_PATH = 'cascades/haar/'


@dataclass
class Cascade:
    path: str = CAS_PATH
    cascade: Union[CascadeClassifier, None] = None

    @classmethod
    def load(cls: Self, cascade: Union[str, int] = 0, **kwargs) -> Self:
        if isinstance(cascade, int):
            cascade = cls.list()[cascade]
        classifier = CascadeClassifier(cls.path + cascade + '.xml',
                                       )
        return cls(cascade=classifier)

    @classmethod
    def list(cls) -> List[str]:
        return [s.split('.')[0] for s in os.listdir(CAS_PATH)]

    def detect(self, img: Img, scale_factor: float = 1.1, min_neighbors: int = 3,
               min_size: Tuple[int] = (30, 30), **kwargs) -> Img:
        if self.cascade is None:
            raise NotFittedError()
        detected = self.cascade.detectMultiScale(img.img_bw(), scaleFactor=scale_factor,
                                                 minNeighbors=min_neighbors,
                                                 minSize=min_size)
        print(f'detected {len(detected)} faces')
        return img.draw(detected)
