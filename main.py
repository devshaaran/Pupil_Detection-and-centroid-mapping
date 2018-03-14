from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import os
import numpy as np

os.chdir('D:\Python_mytext\cv2_notes')



def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor('D:\Python_mytext\cv2_notes\shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame ,  width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:

        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        M = cv2.moments(leftEyeHull)
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])

        N  = cv2.moments(rightEyeHull)
        cX_r = int(N["m10"] / N["m00"])
        cY_r = int(N["m01"] / N["m00"])

        pts = np.array(leftEyeHull,np.int32)
        pts_reshaped = pts.reshape((-1,1,2))
        print(pts_reshaped)


        points_x = []
        points_y = []


        for row in pts_reshaped:
            for elements in row:
                points_x.append(elements[0])
                points_y.append(elements[1])
        min_x = min(points_x)
        min_y = min(points_y)
        max_x = max(points_x)
        max_y = max(points_y)

        pts_r = np.array(rightEyeHull, np.int32)
        pts_reshaped_r = pts.reshape((-1, 1, 2))
        print(pts_reshaped_r)
        points_x_r = []
        points_y_r = []

        for rows in pts_reshaped_r:
            for elementss in rows:
                points_x_r.append(elementss[0])
                points_y_r.append(elementss[1])
        min_x_r = min(points_x_r)
        min_y_r = min(points_y_r)
        max_x_r = max(points_x_r)
        max_y_r = max(points_y_r)

        eye_roi_r = gray[min_y_r:max_y_r,min_x_r:max_x_r]
        eye_roi = gray[min_y:max_y,min_x:max_x]

        ret,threshold2 = cv2.threshold(eye_roi,100,255,cv2.THRESH_BINARY)
        ret,threshold1 = cv2.threshold(eye_roi_r, 100, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(eye_roi, cv2.HOUGH_GRADIENT, 1,1)

        try:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:

                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
                print('working')


        except Exception:
            pass

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.circle(frame,(cX,cY),1,(255,255,255),1)
        cv2.circle(frame, (cX_r, cY_r), 1, (255, 255, 255), 1)
        cv2.imshow('roi',threshold2)
        cv2.imshow('roi_r',threshold1)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

