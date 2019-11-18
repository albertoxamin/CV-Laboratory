import cv2

face_cascade_path = '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/cv2/data/'
path_face_cascade_front = face_cascade_path + 'haarcascade_frontalface_alt.xml'
path_face_cascade_profile = face_cascade_path + 'haarcascade_profileface.xml'

def resize(frame, scale):
    return cv2.resize(
        frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

if __name__ == '__main__':
    cap = cv2.VideoCapture('material/cat.mkv')
    face_cascade_f = cv2.CascadeClassifier()
    face_cascade_p = cv2.CascadeClassifier()

    face_cascade_f.load(path_face_cascade_front)
    face_cascade_p.load(path_face_cascade_profile)
    paused = False
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            frame = resize(frame, 0.4)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)

        faces_f = face_cascade_f.detectMultiScale(
            frame_gray, 1.1, 2, 0 | cv2.CASCADE_SCALE_IMAGE, (30, 30))
        faces_p = face_cascade_p.detectMultiScale(
            frame_gray, 1.1, 2, 0 | cv2.CASCADE_SCALE_IMAGE, (30, 30))

        for face in faces_f:
            cv2.rectangle(frame, (face[0], face[1]),
                          (face[0] + face[2], face[1] + face[3]), (255.0, 0, 255.0), 3)
            cv2.putText(frame, 'front', (face[0], face[1]-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (255.0, 0, 255.0)) 
        for face in faces_p:
            cv2.rectangle(frame, (face[0], face[1]),
                          (face[0] + face[2], face[1] + face[3]), (0, 0, 255.0), 3)
            cv2.putText(frame, 'profile', (face[0], face[1]-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0.0, 0, 255.0)) 


        cv2.imshow("video", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('p'):
            paused = not paused
