import cv2
from cv2 import waitKey
import os
import numpy as np
def compare_faces(img_path1:str,img_path2:str)->float:
    img1=cv2.imread(img_path1)
    img2=cv2.imread(img_path2)
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both image paths are not accessible ")
    img1_grayscale = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_grayscale = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_grayscale = cv2.resize(img1_grayscale, (300, 300))
    img2_grayscale = cv2.resize(img2_grayscale, (300, 00))
    hist1, _ = np.histogram(img1_grayscale.ravel(), bins=256, range=(0, 256))
    hist2, _ = np.histogram(img2_grayscale.ravel(), bins=256, range=(0, 256))
    hist1=hist1/np.sum(hist1)
    hist2=hist2/np.sum(hist2)
    hist_similarity = np.sum(hist1 * hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))
    img1_norm = img1_grayscale / 255.0
    img2_norm = img2_grayscale / 255.0
    img1_norm -= np.mean(img1_norm)
    img2_norm -= np.mean(img2_norm)
    numerator = np.sum(img1_norm * img2_norm)
    denominator = np.sqrt(np.sum(img1_norm ** 2) * np.sum(img2_norm ** 2))
    corr_similarity = numerator / denominator if denominator != 0 else 0
    final_score = (0.6 * hist_similarity) + (0.4 * corr_similarity)
    final_score = float(np.clip(final_score, 0.0, 1.0))
    return final_score
faceDetector  = cv2.CascadeClassifier('../assets/haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier('../assets/haarcascade_eye.xml')
video = cv2.VideoCapture(0)
# root directory to store images
imgRootPath = os.path.join(".." , "pics")
while True:
    val , frame = video.read()
    grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    grayFrame = cv2.equalizeHist(grayFrame)
    grayFrame = cv2.GaussianBlur(grayFrame , (5,5) , 0)
    detectedFace = faceDetector.detectMultiScale(grayFrame , scaleFactor=1.02, minNeighbors=6, minSize=(60,60))
    detectedEyes = eyeDetector.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=8,minSize=(30,30) , maxSize=(80,80))
    for face in detectedFace:
        x, y, w, h = face
        cv2.rectangle(frame, (x,y),(x+w , y+h) ,(0,255,0),2)
    for (x,y, w, h) in detectedEyes:
        cv2.rectangle(frame, (x,y) , (x+w , y+h), (0,0,255),thickness=1)

    cv2.imshow("Frame" , frame)
    cv2.resizeWindow("Frame" , 1280,720)
    if cv2.waitKey(20) & 0xFF == ord('c'):
        name = ""
        while not name.strip() :
            name = input("Your Name: ").strip()
            if not name:
                    print("Are you a ghost? Enter valid name please.")

        personPath = os.path.join(imgRootPath , name)
        os.makedirs(personPath, exist_ok=True)

        for index , face in enumerate(detectedFace):
            x , y , w,  h = face
            detFace = frame[ y:y+h , x:x+w]
            filename = os.path.join(personPath , f"image_{index}.png")
            cv2.imwrite(filename , detFace)
            print("Saved!")

    if waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
score = compare_faces(reference_path, current_path)

if score >= 0.75:
    print(f"Attendance marked for {name}")
else:
    print(f"Face matching failed....(no attendance given) to {name}")