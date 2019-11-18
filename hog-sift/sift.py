import cv2
import numpy as np

if __name__ == '__main__':

    img_obj = cv2.imread('box.png', 0)
    img_scene = cv2.imread('box_in_scene.png', 0)

    sift = cv2.xfeatures2d.SIFT_create(400)
    kp_obj, dsc_obj = sift.detectAndCompute(img_obj, None)
    kp_scene, dsc_scene = sift.detectAndCompute(img_scene, None)

    img_obj = cv2.drawKeypoints(
        img_obj, kp_obj, img_obj, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('obj', img_obj)
    cv2.imshow('scene', img_scene)
    cv2.waitKey(0)

    # Step 2: Matching descriptors between the two images
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.match(dsc_obj, dsc_scene)
    good_matches = list(filter(lambda x: x.distance < 150, matches))

    img3 = cv2.drawMatches(img_obj, kp_obj, img_scene, kp_scene,
                           matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('matches', img3)
    cv2.waitKey(0)

    img3 = cv2.drawMatches(img_obj, kp_obj, img_scene, kp_scene,
                           good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('matches', img3)
    cv2.waitKey(0)

    # Step 3: stitching
    obj_pts = np.float32(
        [kp_obj[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    scn_pts = np.float32(
        [kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(obj_pts, scn_pts, cv2.RANSAC)
    print(img_scene.size)
    # result = cv2.warpPerspective(img_obj, M, ))
    # cv2.imshow('matches', img3)
    cv2.waitKey(0)
