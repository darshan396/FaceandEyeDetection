import os
import cv2
import numpy as np


def compareFaces(img_path1:str,img_path2:str)->float:
    img1=cv2.imread(img_path1)
    img2=cv2.imread(img_path2)
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both image paths are not accessible ")
    img1_grayscale = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_grayscale = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_grayscale = cv2.resize(img1_grayscale, (300, 300))
    img2_grayscale = cv2.resize(img2_grayscale, (300, 300))
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