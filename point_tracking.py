import numpy as np
import cv2 as cv
from dataclasses import dataclass
from typing import Self, Union
from ipywidgets.widgets import Video
fourcc = cv.VideoWriter_fourcc(*'h264')
cap = cv.VideoCapture('movies/dvd.mov')
fps = cap.get(cv.CAP_PROP_FPS)
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
writer = cv.VideoWriter(
    'movies/dvd_tracked.mp4',
    fourcc,
    fps,
    (int(width), int(height))

)
flag = writer.isOpened()
if not flag:
    print('writer not open')
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print("no frame read!")
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while (flag):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)),
                       (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)
    writer.write(mask)
    cv.imshow('DVD', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
cap.release()
writer.release()
cv.destroyAllWindows()


class PointTracker(object):
    capture: cv.VideoCapture
    writer: cv.VideoWriter

    def __init__(self, vid_path: Union[str,int] = 0) -> Self:
        fourcc = cv.VideoWriter_fourcc(*'h264')
        self.capture = cv.VideoCapture(vid_path)
        fps = cap.get(cv.CAP_PROP_FPS)
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        if isinstance(vid_path,int):
            vid_path = 'webcam'
        self.writer = cv.VideoWriter(
            'tracked_' + vid_path,
            fourcc,
            fps,
            (int(width), int(height))
        )
        if not self.capture.isOpened():
            raise FileNotFoundError('Video File Not Found')
        elif not self.writer.isOpened():
            raise ValueError('Unable to open video writer')
    def __delete__(self):
        self.capture.release()
        self.writer.release
