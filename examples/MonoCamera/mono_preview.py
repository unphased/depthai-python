#!/usr/bin/env python3

import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

xoutLeft.setStreamName('left')
xoutRight.setStreamName('right')

# updateInterval: in seconds
# name: if provided, will be verbose, printing FPS when it updates
class FPS:
    def __init__(self, updateInterval=1, name=None):
        self.fps = 0
        self.count = 0
        self.timePrev = None
        self.interval = updateInterval
        self.name = name

    # timestamp: optional to not use current time (but e.g. device timestamp)
    def update(self, timestamp=None):
        import time
        if timestamp is None: timestamp = time.monotonic()
        # When called first, just store the timestamp and return
        if self.timePrev is None:
            self.timePrev = timestamp
            return
        self.count += 1
        tdiff = timestamp - self.timePrev
        if tdiff >= self.interval:
            self.fps = self.count / tdiff
            if self.name:
                print(f'FPS {self.name}: {self.fps:.2f}')
            self.count = 0
            self.timePrev = timestamp

    def get(self):
        return self.fps
    
# Properties
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
fps = 120
monoLeft.setFps(fps)
monoRight.setFps(fps)
# Linking
monoRight.out.link(xoutRight.input)
monoLeft.out.link(xoutLeft.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the grayscale frames from the outputs defined above
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    fpsL = FPS(name='left  ')
    fpsR = FPS(name='right ')
    while True:
        # Instead of get (blocking), we use tryGet (non-blocking) which will return the available data or None otherwise
        inLeft = qLeft.tryGet()
        inRight = qRight.tryGet()

        if inLeft is not None:
            fpsL.update()
            cv2.imshow("left", inLeft.getCvFrame())

        if inRight is not None:
            fpsR.update()
            cv2.imshow("right", inRight.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
