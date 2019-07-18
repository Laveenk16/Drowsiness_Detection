import dlib
import cv2
from scipy.spatial import distance
import imutils
from imutils import face_utils

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("C:/Users/HP/Downloads/shape_predictor_68_face_landmarks (1).dat/shape_predictor_68_face_landmarks (1).dat")
thresh = 0.25
frame_check = 20

(lstart, lend) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
flag = 0

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width = 450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[lstart : lend]
        right_eye = shape[rstart : rend]
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        ear = (left_EAR + right_EAR) / 2.0
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)
        if ear < thresh:
            flag += 1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "-------------ALERT--------------", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, "-------------ALERT--------------", (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            flag = 0        
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()            





