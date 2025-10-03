import cv2
from cv2 import waitKey

faceDetector  = cv2.CascadeClassifier('/home/ghost/Coding/detection/assets/haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)

while True:
    val , frame = video.read()
    videoData = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    videoData = cv2.equalizeHist(videoData)
    detectedFace = faceDetector.detectMultiScale(videoData , scaleFactor=1.02, minNeighbors=6, minSize=(60,60))

    for face in detectedFace:
        x, y, w, h = face
        cv2.rectangle(frame, (x,y),(x+w , y+ h) ,(0,255,0),2)


    cv2.imshow("Frame" , frame)
    cv2.resizeWindow("Frame" , 1280,720)
    if waitKey(60) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()