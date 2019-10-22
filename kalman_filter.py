import cv2
import numpy as np


def resize(frame, scale):
    return cv2.resize(
        frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

mouse_info = np.array([[np.float32(0)],[np.float32(0)]])
last_mouse = np.array([[np.float32(0)],[np.float32(0)]])


def onMouse(event, x, y, flags, param):
    global last_mouse, mouse_info
    last_mouse = mouse_info
    mouse_info = np.array([np.float32(x),[np.float32(y)]])


def drawCross(img, center, color, d):
    cv2.line(img, (center[0] - d, center[1] - d),
             (center[0] + d, center[1] + d), color, 2, cv2.CV_8U)
    cv2.line(img, ( center[0] + d, center[1] - d ),( center[0] - d, center[1] + d ), color, 2, cv2.CV_8U)


if __name__ == '__main__':
    img = np.zeros((1000, 1000), float)

    cv2.namedWindow("kf")
    cv2.setMouseCallback("kf", onMouse)

    mousev = []
    kalmanv = []

    KF = cv2.KalmanFilter(4, 2)
    KF.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    cv2.setIdentity(KF.measurementMatrix)
    cv2.setIdentity(KF.processNoiseCov, 0.0001)
    cv2.setIdentity(KF.measurementNoiseCov, 0.1)
    cv2.setIdentity(KF.errorCovPost, 0.01)

    print("Press ESC to close.")
    while True:
        prediction = KF.predict()
        mousev.append(mouse_info)

        estimate = KF.correct(mouse_info)
        kalmanv.append(estimate)

        img = np.zeros((1000, 1000), float)
        
        drawCross(img, estimate, np.array([255, 255, 255], float), 5)
        drawCross(img, estimate, np.array([0, 0, 255], float), 5)
        drawCross(img, estimate, np.array([0, 255, 0], float), 5)

        k = cv2.waitKey(30) &0xFF
        if k == 27: break
