from imutils import face_utils
import dlib
import cv2
 
# landmarks on this detected face
# our pre-treined model directory, on my case, it's on the same script's diretory.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/HP/Downloads/shape_predictor_68_face_landmarks (1).dat/shape_predictor_68_face_landmarks (1).dat")

cap = cv2.VideoCapture(0)
 
while True:
    # Getting out image by webcam 
    ret, frame = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # Draw on our image, all the finded cordinate points (x,y) 
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    # Show the image
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
