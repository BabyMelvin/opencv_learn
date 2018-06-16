# this script will detect faces via your webcam
# test with OpenCV3

import cv2

cap = cv2.VideoCapture(0)

# must the absolute path
abs_path = 'C:\\Users\\dell\\PycharmProjects\\opencv\\project\\res\\haarcascade_frontalface_default.xml'
# create the haar cascade
faceCascade = cv2.CascadeClassifier(abs_path)

while True:
    # capture frame-by-frame
    ret, frame = cap.read()

    # our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
        # flags=cv2.CV_HARR_SCALE_IMAGE
    )

    # draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when everything done ,release the capture
cap.release()
cv2.destroyAllWindows()
