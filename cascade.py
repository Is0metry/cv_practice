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
        '''
        Loads a given cascade from `path`.
        ## Parameters:
        cascade: either a string of the Haar cascade name or an integer indicating
        its position in `Cascade.list()`
        ## Returns:
        An initialized `Cascade` object with the given Haar cascade
        '''
        if isinstance(cascade, int):
            cascade = cls.list()[cascade]
        classifier = CascadeClassifier(cls.path + cascade + '.xml',
                                       )
        return cls(cascade=classifier)

    @classmethod
    def list(cls) -> List[str]:
        '''
        Lists the available cascades in `path`
        ## Parameters:
        None
        ## Returns:
        a `List` of the available Haar cascades in `path`
        '''
        return [s.split('.')[0] for s in os.listdir(CAS_PATH)]

    def detect(self, img: Img, scale_factor: float = 1.1, min_neighbors: int = 3,
               min_size: Tuple[int] = (30, 30), **kwargs) -> Img:
        '''
        Runs the image against the Haar cascade to detect objects
        ## Parameters:
        img: `img.Img` object to run detection on
        scale_factor: passed to `detectMultiScale` as `scaleFactor`
        min_neighbors: passed to `detectMultiScale` as `minNeighbors`
        min_size: passed to `detectMultiScale` as `minSize`
        ## Returns:
        an Img object with detected objects blurred (see `img.Img.blur_detected`)
        '''
        if self.cascade is None:
            raise NotFittedError()
        detected = self.cascade.detectMultiScale(img.img_bw(), scaleFactor=scale_factor,
                                                 minNeighbors=min_neighbors,
                                                 minSize=min_size)
        print(f'detected {len(detected)} faces')
        return img.blur_detected(detected)
