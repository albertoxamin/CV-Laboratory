import cv2
import numpy as np

image = np.zeros((0,0,0), dtype=np.uint8)

backprojMode = False
selectObj = False
trackObj = 0
showHist = True
origin = [0, 0]
selection = [0, 0, 0, 0]
vmin = 10
vmax = 256
smin = 30


def resize(frame, scale):
    return cv2.resize(
        frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))


def onMouse(event, x, y, flags, param):
    global selectObj, origin, selection, trackObj, image
    if selectObj:
        selection[0] = min(x, origin[0])
        selection[1] = min(y, origin[1])
        selection[2] = abs(x - origin[0])
        selection[3] = abs(y - origin[1])
        selection &= cv2.rectangle(0,0, image.cols, image.rows)

    if event == cv2.EVENT_LBUTTONDOWN:
        origin = [x, y]
        selection = [x, y, 0, 0]
        selectObj = True
    elif event == cv2.EVENT_LBUTTONUP:
        selectObj = False
        if selection[2] > 0 and selection[3] > 0:
            trackObj = -1


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)  # 'material/Video.mp4')

    scale = 0.5
    hsize = 16
    hranges = [0, 180]

    cv2.namedWindow("Histogram")
    cv2.namedWindow("meanShift Demo")
    cv2.setMouseCallback("meanShift Demo", onMouse)

    # frame, hsv, hue, mask, hist, histimg = np.zeros((400,640, 0), dtype=np.uint8)

    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            frame = resize(frame, scale)

        image = frame.copy()

        if not paused:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            if trackObj:
                _vmin, _vmax = vmin, vmax
                mask = cv2.inRange(hsv, np.array([0, smin, min(_vmin, _vmax)], float), np.array(
                    [180, 256, max(_vmin, vmax), float]))
                ch = [0, 0]
                #hue.create(hsv.size(), hsv.depth());
                cv2.mixChannels(hsv, hue)
                # mixChannels(&hsv, 1, &hue, 1, ch, 1);

                if (trackObj < 0):
                    cv2.rectangle(hue, selection)
                    # cv2.

        elif (trackObj < 0):
            paused = False

        if (selectObj and selection[3] > 0):
            cv2.rectangle(
                image,
                (selection[0], selection[1]),
                (selection[2], selection[3]),
                np.array([5, 2, 255], float))

        cv2.imshow("meanShift Demo", image)
        # cv2.imshow("Histogram", histimg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
