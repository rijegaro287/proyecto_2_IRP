import cv2
import os

original_dataset_dir = './datasets/unfiltered_images'
bilateral_filtered_dataset_dir = './datasets/bilateral_filtered_images'
feature_extracted_dataset_dir = './datasets/feature_extracted_images'

if (not os.path.exists(bilateral_filtered_dataset_dir)):
    os.mkdir(bilateral_filtered_dataset_dir)

if (not os.path.exists(feature_extracted_dataset_dir)):
    os.mkdir(feature_extracted_dataset_dir)

for file_name in os.listdir(original_dataset_dir):
    original_image_path = os.path.join(original_dataset_dir, file_name)
    filtered_image_path = os.path.join(
        bilateral_filtered_dataset_dir, file_name)

    print(filtered_image_path)
    img = cv2.imread(original_image_path)

    bilateral_filtered_image = cv2.bilateralFilter(img, 5, 75, 75)

    cv2.imwrite(filtered_image_path, bilateral_filtered_image)


for file_name in os.listdir(original_dataset_dir):
    original_image_path = os.path.join(original_dataset_dir, file_name)
    feature_extracted_image_path = os.path.join(
        feature_extracted_dataset_dir, file_name)

    print(feature_extracted_image_path)
    img = cv2.imread(original_image_path)

    orb = cv2.ORB_create()
    kp = orb.detect(img, None)

    kp, des = orb.compute(img, kp)

    img2 = cv2.drawKeypoints(img, kp, None, color=(255, 255, 255), flags=0)

    cv2.imwrite(feature_extracted_image_path, img2)
