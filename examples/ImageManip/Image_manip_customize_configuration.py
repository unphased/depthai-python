#!/usr/bin/env python3

"""
This example shows usage of ImageManip to crop a rotated rectangle area on a frame,
or perform various image transforms: rotate, mirror, flip, perspective transform.
"""

import depthai as dai
import cv2
import numpy as np

supportedInOutFormats = {0: dai.ImgFrame.Type.YUV420p, 1: dai.ImgFrame.Type.NV12, 2: dai.ImgFrame.Type.BGR888p, 3: dai.ImgFrame.Type.RGB888p, 4: dai.ImgFrame.Type.RGB888i, 5: dai.ImgFrame.Type.BGR888i, 5: dai.ImgFrame.Type.RAW8, 6: dai.ImgFrame.Type.GRAY8} #also raw16 is supported but not validated

keyRotateDecr = 'z'
keyRotateIncr = 'x'
keyWarpTestCycle = 'c'
keyResizeVerInDecr = 'v'
keyResizeHorInDecr = 'b'
keyResizeVerOutDecr = 'n'
keyResizeHorOutDecr = 'm'
keyOutputFormat = 'a'

def printControls():
    print("=== Controls:")
    print("use the up/down/left/right arrow keys - for moving the input croped region possition")
    print(keyRotateDecr, "-rotated rectangle crop, decrease rate")
    print(keyRotateIncr, "-rotated rectangle crop, increase rate")
    print(keyWarpTestCycle, "-warp 4-point transform, cycle through modes")
    print(keyResizeVerInDecr, "-decrease the input croped region vertical size, or disabling it")
    print(keyResizeHorInDecr, "-decrease the input croped region horizontal size , or disabling it")
    print(keyResizeVerOutDecr, "-decrease the output vertical size, or disabling it")
    print(keyResizeHorOutDecr, "-decrease the output horizontal size , or disabling it")
    print(keyOutputFormat, "-change the output format")
    print("h -print controls (help)")

rotateRateMax = 5.0
rotateRateInc = 0.1

resizeOutMaxW = 1920 # 4K - max resolution supported
resizeOutMaxH = 1080 # 4K - max resolution supported
resizeHorOutFactorMax = resizeOutMaxW - 32
resizeVerOutFactorMax = resizeOutMaxH - 32

resizeInMaxW = 1920 # 4K - max resolution supported
resizeInMaxH = 1080 # 4K - max resolution supported
resizeHorInFactorMax = resizeInMaxW - 32
resizeVerInFactorMax = resizeInMaxH - 32

'''
The crop points are specified in clockwise order,
with first point mapped to output top-left, as:
    P0  ->  P1
     ^       v
    P3  <-  P2
'''
P0 = [0, 0]  # top-left
P1 = [1, 0]  # top-right
P2 = [1, 1]  # bottom-right
P3 = [0, 1]  # bottom-left

warpList = [
    # points order, normalized cordinates, description
    # [[[0, 0], [1, 0], [1, 1], [0, 1]], True, "passthrough"],
    # [[[0, 0], [639, 0], [639, 479], [0, 479]], False, "passthrough (pixels)"],
    [[P0, P1, P2, P3], True, "1. passthrough"],
    [[P3, P0, P1, P2], True, "2. rotate 90"],
    [[P2, P3, P0, P1], True, "3. rotate 180"],
    [[P1, P2, P3, P0], True, "4. rotate 270"],
    [[P1, P0, P3, P2], True, "5. horizontal mirror"],
    [[P3, P2, P1, P0], True, "6. vertical flip"],
    [[[-0.1, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.1, 1.1]], True, "7. add black borders"],
    [[[-0.3, 0], [1, 0], [1.3, 1], [0, 1]], True, "8. parallelogram transform"],
    [[[-0.2, 0], [1.8, 0], [1, 1], [0, 1]], True, "9. trapezoid transform"],
]

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
manip = pipeline.create(dai.node.ImageManip)

camOut = pipeline.create(dai.node.XLinkOut)
manipOut = pipeline.create(dai.node.XLinkOut)
manipCfg = pipeline.create(dai.node.XLinkIn)

camOut.setStreamName("preview")
manipOut.setStreamName("manip")
manipCfg.setStreamName("manipCfg")

# Properties
# camRgb.setPreviewSize(1920, 1080)
camRgb.setIspScale(1, 1)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
manip.setMaxOutputFrameSize(1920 * 1080 * 3)
manip.setKeepAspectRatio(False)
# manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

# Linking
camRgb.isp.link(camOut.input)
camRgb.isp.link(manip.inputImage)
manip.out.link(manipOut.input)
manipCfg.out.link(manip.inputConfig)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Create input & output queues
    qPreview = device.getOutputQueue(name="preview", maxSize=4)
    qManip = device.getOutputQueue(name="manip", maxSize=4)
    qManipCfg = device.getInputQueue(name="manipCfg")

    key = -1
    angleDeg = 0
    rotateRate = 0.0001
    resizeHorOutFactor = 0
    resizeVerOutFactor = 0
    resizeOutW = resizeOutMaxW
    resizeOutH = resizeOutMaxH
    resizeHorInFactor = 0
    resizeVerInFactor = 0
    sizeInRoiW = 400#resizeInMaxW
    sizeInRoiH = 400#resizeInMaxH
    testFourPt = False
    warpIdx = -1
    posRoiX = camRgb.getVideoWidth() // 2
    posRoiY = camRgb.getVideoHeight() // 2
    outputFormat = 0
    formatListSize = len(supportedInOutFormats)
    printControls()

    while key != ord('q'):
        if key > 0:
            print("Pressed: ", key)
            if key == ord(keyRotateDecr) or key == ord(keyRotateIncr):
                if key == ord(keyRotateDecr):
                    if rotateRate > -rotateRateMax:
                        rotateRate -= rotateRateInc
                if key == ord(keyRotateIncr):
                    if rotateRate < rotateRateMax:
                        rotateRate += rotateRateInc
                testFourPt = False
                print("Crop rotated rectangle, rate per frame: {:.1f} degrees".format(rotateRate))
            elif key == ord(keyResizeHorOutDecr):
                resizeHorOutFactor += 2
                if resizeHorOutFactor > resizeHorOutFactorMax:
                    resizeHorOutFactor = 0
                    print("Crop region not resized")
                else:
                    resizeOutW = resizeOutMaxW - resizeHorOutFactor
                    print("Output resolution: ", resizeOutW, 'x', resizeOutH)
            elif key == ord(keyResizeVerOutDecr):
                resizeVerOutFactor += 2
                if resizeVerOutFactor > resizeVerOutFactorMax:
                    resizeVerOutFactor = 0
                    print("Crop region not resized")
                else:
                    resizeOutH = resizeOutMaxH - resizeVerOutFactor
                    print("Output resolution: ", resizeOutW, 'x', resizeOutH)
            elif key == ord(keyResizeHorInDecr):
                resizeHorInFactor += 2
                if resizeHorInFactor > resizeHorInFactorMax:
                    resizeHorInFactor = 0
                    print("Input crop region not resized")
                else:
                    sizeInRoiW = resizeInMaxW - resizeHorInFactor
                    print("Input crop region resolution: ", sizeInRoiW, 'x', sizeInRoiH)
            elif key == ord(keyResizeVerInDecr):
                resizeVerInFactor += 2
                if resizeVerInFactor > resizeVerInFactorMax:
                    resizeVerInFactor = 0
                    print("Input crop region not resized")
                else:
                    sizeInRoiH = resizeInMaxH - resizeVerInFactor
                    print("Input crop region resolution: ", sizeInRoiW, 'x', sizeInRoiH)
            elif key == ord(keyWarpTestCycle):
                # Disable resizing initially
                resizeHorOutFactor = 0
                resizeVerOutFactor = 0
                warpIdx = (warpIdx + 1) % len(warpList)
                testFourPt = True
                testDescription = warpList[warpIdx][2]
                print("Warp 4-point transform: ", testDescription)
            elif key == 81:
                posRoiX -= 2
            elif key == 82:
                posRoiY -= 2
            elif key == 83:
                posRoiX += 2
            elif key == 84:
                posRoiY += 2
            elif key == ord(keyOutputFormat):
                outputFormat += 1
                outputFormat = outputFormat % formatListSize
                print("Output format: ", supportedInOutFormats[outputFormat])
            elif key == ord('h'):
                printControls()

        # Send an updated config with continuous rotate, or after a key press
        if key >= 0 or (not testFourPt and abs(rotateRate) >= 0.0001):
            cfg = dai.ImageManipConfig()
            if testFourPt:
                test = warpList[warpIdx]
                points, normalized = test[0], test[1]
                point2fList = []
                for p in points:
                    pt = dai.Point2f()
                    pt.x, pt.y = p[0], p[1]
                    point2fList.append(pt)
                cfg.setWarpTransformFourPoints(point2fList, normalized)
            else:
                angleDeg += rotateRate
                rotatedRect = ((posRoiX, posRoiY), (sizeInRoiW, sizeInRoiH), angleDeg)
                rr = dai.RotatedRect()
                rr.center.x, rr.center.y = rotatedRect[0]
                rr.size.width, rr.size.height = rotatedRect[1]
                rr.angle = rotatedRect[2]
                cfg.setCropRotatedRect(rr, False)
            if resizeHorOutFactor > 0 or resizeVerOutFactor > 0:
                cfg.setResize(resizeOutW, resizeOutH)

            # cfg.setWarpBorderFillColor(255, 0, 0)
            # cfg.setWarpBorderReplicatePixels()
            cfg.setFrameType(supportedInOutFormats[outputFormat])
            cfg.setKeepAspectRatio(False)
            qManipCfg.send(cfg)

        for q in [qPreview, qManip]:
            pkt = q.get()
            name = q.getName()
            shape = (3, pkt.getHeight(), pkt.getWidth())
            frame = pkt.getCvFrame()
            if name == "preview" and not testFourPt:
                # Draw RotatedRect cropped area on input frame
                points = np.int0(cv2.boxPoints(rotatedRect))
                cv2.drawContours(frame, [points], 0, (255, 0, 0), 1)
                # Mark top-left corner
                cv2.circle(frame, tuple(points[1]), 10, (255, 0, 0), 2)
            cv2.imshow(name, frame)
        key = cv2.waitKey(1)
