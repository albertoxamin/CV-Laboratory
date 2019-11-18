import cv2
import random

face_cascade_path = '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/cv2/data/haarcascade_'
cascades_strings = ['frontalface_alt2', 'profileface']


def resize(frame, scale):
    return cv2.resize(
        frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    cascades = []
    for casc_str in cascades_strings:
        casc = cv2.CascadeClassifier()
        casc.load(f'{face_cascade_path}{casc_str}.xml')
        cascades.append(casc)
    colors = [(random.randint(0, 255), random.randint(
        0, 255), random.randint(0, 255)) for c in cascades]

    paused = False
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            frame = resize(frame, 0.4)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)

        detected = [c.detectMultiScale(
            frame_gray, 1.1, 2, 0 | cv2.CASCADE_SCALE_IMAGE, (30, 30)) for c in cascades]

        for i in range(0, len(cascades)):
            for face in detected[i]:
                cv2.rectangle(frame, (face[0], face[1]),
                              (face[0] + face[2], face[1] + face[3]), colors[i], 3)
                cv2.putText(
                    frame, cascades_strings[i], (face[0], face[1]-5), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[i])

        cv2.imshow("video", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('p'):
            paused = not paused
