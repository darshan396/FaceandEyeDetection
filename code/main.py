import cv2
from cv2 import waitKey

faceDetector  = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)

while True:
    val , frame = video.read()
    videoData = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    detectedFace = faceDetector.detectMultiScale(videoData , scaleFactor=1.07, minNeighbors=10, minSize=(45,45))
    cv2.imshow("Frame" , videoData)
    cv2.resizeWindow("Frame" , 1280,720)
    if waitKey(60) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()