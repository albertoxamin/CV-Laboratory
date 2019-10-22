import cv2
import numpy as np


def resize(frame, scale):
    return cv2.resize(
        frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)  # 'material/Video.mp4')

    scale = 0.5

    # Good Features to Track Parameters
    gft = dict(maxCorners=100,
               qualityLevel=0.01,
               minDistance=10,
               blockSize=3,
               useHarrisDetector=False,
               k=0.04)

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame = resize(frame, scale)
        frame_copy = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Select GFF Features
        if frame_index % 30 == 0:
            corners = cv2.goodFeaturesToTrack(frame_gray, **gft)
        else:
            corners, status, err = cv2.calcOpticalFlowPyrLK(
                prev_frame, frame, prev_corners, None)

        np_corners = np.int0(corners)
        for i in np_corners:
            x, y = i.ravel()
            cv2.circle(frame_copy, (x, y), 3,
                       np.array([5*y, 2*y, 255-y], float))

        cv2.imshow('gff', frame_copy)

        prev_frame, prev_corners = frame.copy(), corners

        frame_index += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
