import cv2
from cv2 import waitKey
import os
import numpy as np
import glob

faceDetector  = cv2.CascadeClassifier('../assets/haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier('../assets/haarcascade_eye.xml')
video = cv2.VideoCapture(0)

# root directory to store images
imgRootPath = os.path.join(".." , "pics")
def personName():
    while True:
        name = input("Your Name: ").strip()
        if name:
            return name
        print("Are you a ghost? Enter valid name please.")
    return name


def faceDetection():
    while True:
        val, frame = video.read()
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayFrame = cv2.equalizeHist(grayFrame)
        grayFrame = cv2.GaussianBlur(grayFrame, (5, 5), 0)
        detectedFace = faceDetector.detectMultiScale(grayFrame, scaleFactor=1.02, minNeighbors=6, minSize=(80, 80))
        detectedEyes = eyeDetector.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30), maxSize=(80, 80))
        for face in detectedFace:
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y, w, h) in detectedEyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)

        cv2.imshow("Frame", frame)
        cv2.resizeWindow("Frame", 1280, 720)

        if cv2.waitKey(20) & 0xFF == ord('c'):
            name = personName()
            name = name.strip()
            personPath = os.path.join(imgRootPath, name.lower())
            os.makedirs(personPath, exist_ok=True)

            if len(detectedFace) == 0:
                print("No face detected.")
            else:
                for index, face in enumerate(detectedFace):
                    x, y, w, h = face
                    detFace = frame[y:y + h, x:x + w]
                    filename = os.path.join(personPath, f"image_{index}.png")
                    cv2.imwrite(filename, detFace)
                    print("Saved!")
                    print("Image captured sucessfully! ")
        if waitKey(20) & 0xFF == ord('q'):
            break

def compareFaces(path1: str, path2: str) -> float:
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Some image paths are not accessible.")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.resize(gray1, (300, 300))
    gray2 = cv2.resize(gray2, (300, 300))

    hist1, _ = np.histogram(gray1.ravel(), bins=256, range=(0, 256))
    hist2, _ = np.histogram(gray2.ravel(), bins=256, range=(0, 256))

    epsilon = 1e-10
    hist1 = hist1 / (np.sum(hist1) + epsilon)
    hist2 = hist2 / (np.sum(hist2) + epsilon)

    hist_similarity = np.sum(hist1 * hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2) + epsilon)

    img1_norm = gray1 / 255.0
    img2_norm = gray2 / 255.0
    img1_norm -= np.mean(img1_norm)
    img2_norm -= np.mean(img2_norm)
    numerator = np.sum(img1_norm * img2_norm)
    denominator = np.sqrt(np.sum(img1_norm ** 2) * np.sum(img2_norm ** 2))
    corr_similarity = numerator / denominator if denominator != 0 else 0

    final_score = (0.6 * hist_similarity) + (0.4 * corr_similarity)
    return float(np.clip(final_score, 0.0, 1.0))

def identifyPerson():
    if not os.path.exists(imgRootPath):
        print("No person found.")
        return

    print("Press 'i' to mark attendance.")
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detectedFace = faceDetector.detectMultiScale(gray, 1.1, 6, minSize=(100, 100))

        for (x, y, w, h) in detectedFace:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Image", frame)

        if cv2.waitKey(20) & 0xFF == ord('i'):
            if len(detectedFace) == 0:
                print("No face detected.")
                continue

            for (x, y, w, h) in detectedFace:
                capturedFace = frame[y:y +h, x:x + w]
                tempPath = "temp_face.png"
                cv2.imwrite(tempPath, capturedFace)

                best_match_name = "Unknown"
                best_score = 0.0

                for person in os.listdir(imgRootPath):
                    personPath = os.path.join(imgRootPath, person)
                    if os.path.isdir(personPath):
                        for img_path in glob.glob(os.path.join(personPath, "*.png")):
                            try:
                                score = compareFaces(tempPath, img_path)
                                if score > best_score:
                                    best_score = score
                                    best_match_name = person
                            except Exception:
                                continue

                os.remove(tempPath)

                confidence = 0.45
                if best_score > confidence:
                    print(f"Marked attendance for : {best_match_name} . || Similarity : {best_score:.2f}")
                    color = (0,255,0)

                else:
                    print(f"Not Found || Similarity{best_score:.2f}")
                    color = (0,0,255)

        elif waitKey(20) & 0xFF == ord('q'):
            break
def main():
    print("Register face in system. (Press 1)")
    print("Mark your attendance. (Press 2)")
    value = int(input("Your action?  -> "))
    if value == 1:
        print("Press c to capture your image.")
        faceDetection()

    elif value == 2:
        identifyPerson()

    else:
        print("Invalid Argument. Please try again.")
        print("====================================")
        main()

main()
video.release()
cv2.destroyAllWindows()