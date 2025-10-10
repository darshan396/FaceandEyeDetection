import cv2
import os
import numpy as np

def compare_faces(img_path1: str, img_path2: str) -> float:
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both image paths are not accessible.")

    # Convert to grayscale and resize
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.resize(img1_gray, (300, 300))
    img2_gray = cv2.resize(img2_gray, (300, 300))

    # Histogram comparison
    hist1, _ = np.histogram(img1_gray.ravel(), bins=256, range=(0, 256))
    hist2, _ = np.histogram(img2_gray.ravel(), bins=256, range=(0, 256))
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    hist_similarity = np.sum(hist1 * hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))

    # Correlation comparison
    img1_norm = (img1_gray / 255.0) - np.mean(img1_gray / 255.0)
    img2_norm = (img2_gray / 255.0) - np.mean(img2_gray / 255.0)
    numerator = np.sum(img1_norm * img2_norm)
    denominator = np.sqrt(np.sum(img1_norm ** 2) * np.sum(img2_norm ** 2))
    corr_similarity = numerator / denominator if denominator != 0 else 0

    final_score = (0.6 * hist_similarity) + (0.4 * corr_similarity)
    return float(np.clip(final_score, 0.0, 1.0))


def main():
    face_cascade = cv2.CascadeClassifier('../assets/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../assets/haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)
    pics_path = os.path.join("..", "pics")
    os.makedirs(pics_path, exist_ok=True)

    print("Press 'c' to capture and compare | Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=6, minSize=(60, 60))
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30), maxSize=(80, 80))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv2.imshow("Frame", frame)
        cv2.resizeWindow("Frame", 1280, 720)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('c'):
            name = ""
            while not name.strip():
                name = input("Enter your name: ").strip()
                if not name:
                    print("Please enter a valid name.")

            person_folder = os.path.join(pics_path, name)
            os.makedirs(person_folder, exist_ok=True)

            if len(faces) == 0:
                print("No face detected. Try again.")
                continue

            ref_image_path = os.path.join(person_folder, "image_0.png")

            for idx, (x, y, w, h) in enumerate(faces):
                face_img = frame[y:y+h, x:x+w]
                current_image_path = os.path.join(person_folder, f"current_{idx}.png")
                cv2.imwrite(current_image_path, face_img)
                print(f"Saved current image: {current_image_path}")

                if not os.path.exists(ref_image_path):
                    saved = cv2.imwrite(ref_image_path, face_img)
                    if saved:
                        print(f"Reference image saved at: {ref_image_path}")
                    else:
                        print(f"Could not save reference image at: {ref_image_path}")
                else:
                    print("Comparing with reference image...")
                    score = compare_faces(ref_image_path, current_image_path)
                    print(f"Similarity score: {score:.2f}")
                    if score >= 0.75:
                        print(f"Attendance marked for {name}")
                    else:
                        print(f"Face does not match for {name}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()