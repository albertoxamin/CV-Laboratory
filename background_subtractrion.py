import cv2
import os
import numpy as np


def bg_train(target_frame, bg, i):
    if i == 1:
        return target_frame, 2
    return bg, 2


def bg_update(alpha, curr_frame, bg):
    return cv2.multiply(alpha, curr_frame) + cv2.multiply(1 - alpha, bg)


def bg_gaussian_average(alpha, mask, curr_frame, bg):
    return cv2.multiply(mask, bg) + cv2.multiply(1 - mask, bg_update(alpha, curr_frame, bg))


def show(_name, _frame, _mask, _background):
    cv2.imshow(_name,
               np.vstack((
                   np.hstack((
                       _frame,
                       cv2.bitwise_and(_frame, _frame, mask=_mask),
                   )),
                   np.hstack((
                       cv2.cvtColor(_mask, cv2.COLOR_GRAY2RGB),
                       cv2.cvtColor(_background, cv2.COLOR_GRAY2RGB)
                   )),
               )))


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    ctr = 1
    background = None

    mog = cv2.bgsegm.createBackgroundSubtractorMOG()
    gmg = cv2.bgsegm.createBackgroundSubtractorGMG()
    cnt = cv2.bgsegm.createBackgroundSubtractorCNT()
    gsoc = cv2.bgsegm.createBackgroundSubtractorGSOC()
    lsbp = cv2.bgsegm.createBackgroundSubtractorLSBP()
    mog2 = cv2.createBackgroundSubtractorMOG2(history=30, detectShadows=False)
    knn = cv2.createBackgroundSubtractorKNN()

    scale = 0.5

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(
            frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Background update functions
        background, ctr = bg_train(frame_gray, background, ctr)

        motion_mask = cv2.absdiff(background, frame_gray)
        th, motion_mask_t = cv2.threshold(
            motion_mask, 50, 255, cv2.THRESH_BINARY)

        # background = bg_update(0.02, frame_gray, background)
        background = bg_gaussian_average(0.1, cv2.divide(
            motion_mask_t, 255), frame_gray, background)
        # show('gaussian avg', frame, motion_mask_t, background)

        mog2_mask = mog2.apply(frame_gray)
        show('MOG2', frame, mog2_mask, mog2.getBackgroundImage())

        # mog_mask = mog.apply(frame_gray)
        # show('MOG', frame, mog_mask, background)

	# knn_mask = knn.apply(frame_gray)
        # show('knn', frame, knn_mask, knn.getBackgroundImage())

	# gmg_mask = gmg.apply(frame_gray)
        # show('GMG', frame, gmg_mask, background)

	# cnt_mask = cnt.apply(frame_gray)
        # show('cnt', frame, cnt_mask, background)

	# gsoc_mask = gsoc.apply(frame_gray)
        # show('gsoc', frame, gsoc_mask, background)

	# lsbp_mask = lsbp.apply(frame_gray)
        # show('lsbp', frame, lsbp_mask, background)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
