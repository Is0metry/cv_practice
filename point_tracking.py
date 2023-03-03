import numpy as np
import cv2 as cv
from dataclasses import dataclass
from typing import Self, Tuple, Union, Dict
from ipywidgets.widgets import Video

FEATURE_PARAMS = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

LK_PARAMS = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


class PointTracker(object):
    capture: cv.VideoCapture
    writer: cv.VideoWriter
    feature_params: Dict[str, Union[float, int]] = FEATURE_PARAMS
    color = np.random.randint(0, 255, (100, 3))
    lk_params: Dict[str,Union[int, Tuple[int, int],
                              Tuple[int, int, float]]] = LK_PARAMS

    def __init__(self, vid_path: Union[str, int] = 0,
                 feature_params: Dict[str, Union[float, int]] = FEATURE_PARAMS,
                 lk_params: Dict[str,Union[int, Tuple[int, int],
                                           Tuple[int, int, float]]] = LK_PARAMS) -> Self:
        fourcc = cv.VideoWriter_fourcc(*'h264')
        self.capture = cv.VideoCapture(vid_path)
        fps = self.capture.get(cv.CAP_PROP_FPS)
        width = self.capture.get(cv.CAP_PROP_FRAME_WIDTH)
        height = self.capture.get(cv.CAP_PROP_FRAME_HEIGHT)
        if isinstance(vid_path, int):
            vid_path = 'tracked_webcam.mp4'
        else:
            file_path = vid_path.split('/')
            filename = file_path[len(file_path)-1].split('.')[0]
            vid_path = '/'.join(file_path[:len(file_path) - 1]) + '/tracked_' + filename + '.mp4'

        self.writer = cv.VideoWriter(
            vid_path,
            fourcc,
            fps,
            (int(width), int(height))
        )
        if not self.capture.isOpened():
            raise FileNotFoundError('Video File Not Found')
        elif not self.writer.isOpened():
            raise ValueError('Unable to open video writer')
        self.feature_params = feature_params
        self.lk_params = lk_params

    def __delete__(self):
        self.capture.release()
        self.writer.release()

    def track(self, no_points: int = 0, overlay_original: bool = True) -> bool:
        flag, old_frame = self.capture.read()
        if not flag:
            raise RuntimeError('Unable to read file')
        color = np.random.randint(0, 255, (100, 3))
        mask = np.zeros_like(old_frame)
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        old_points = cv.goodFeaturesToTrack(
            old_gray, mask=None, **self.feature_params)
        if no_points > 0:
            old_points = old_points[:no_points]
        cv.imshow('Frame',old_frame)
        while (flag):
            flag, new_frame = self.capture.read()
            if not flag:
                break
            new_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
            new_points, status, error = cv.calcOpticalFlowPyrLK(
                old_gray, new_gray, old_points, None, **self.lk_params
            )
            if new_points is not None:
                good_old = old_points[status == 1]
                good_new = new_points[status == 1]
            for i, (new, old) in enumerate(zip(good_old, good_new)):
                a, b, = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)),
                               (int(c), int(d)), color[i].tolist(), 2)
            old_gray = new_gray.copy()
            old_points = good_new.reshape(-1, 1, 2)
            img = cv.add(mask, new_frame)
            self.writer.write(img)
            cv.imshow('Frame',img)
        return True


if __name__ == "__main__":
    point_tracker = PointTracker('movies/dvd.mov')
    point_tracker.track()
    cv.destroyAllWindows()
