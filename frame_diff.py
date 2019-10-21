import cv2
import os
import numpy as np

if __name__ == '__main__':
    cap = cv2.VideoCapture('rtsp://admin:KEGHKW@192.168.1.4/11')
    ret, current_frame = cap.read()
    frame_look_window = 15
    frame_array = [current_frame, current_frame]

    cv2.namedWindow('win')
    os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')
    split_view = True
    while cap.isOpened():
        current_frame_gray = cv2.cvtColor(frame_array[0], cv2.COLOR_RGB2GRAY)
        previous_frame_gray = cv2.cvtColor(frame_array[-1], cv2.COLOR_RGB2GRAY)

        frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
        th, dst = cv2.threshold(frame_diff, 50, 255, cv2.IMREAD_GRAYSCALE)
        dilated = cv2.dilate(dst, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        # frame_diff = cv2.medianBlur(frame_diff, 1)
        # cv2.imshow('win', frame_diff)
        # cv2.imshow('win', dst)
        if split_view:
            cv2.imshow('win', dst)
            # cv2.imshow('win', np.hstack((current_frame, cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR))))
        else:
            cv2.imshow('win', current_frame)
        # cv2.imshow('frame', current_frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            split_view = not split_view
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(frame_array) % frame_look_window == 0:
            frame_array.pop(0)
        ret, current_frame = cap.read()
        frame_array.append(current_frame)

    cap.release()
    cv2.destroyAllWindows()
