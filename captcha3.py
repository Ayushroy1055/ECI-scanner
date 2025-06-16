import requests
from io import BytesIO
from PIL import Image, ImageEnhance
import base64
import pandas as pd
import cv2
import numpy as np
from easyocr import Reader
import os

def fetch_captcha_and_id(captcha_url):
    """
    Fetch the ECI captcha and its ID from the given URL.
    Returns the captcha ID and captcha image as a PIL image.
    """
    try:
        response = requests.get(captcha_url)
        response.raise_for_status()

        captcha_response = response.json()
        captcha_data = captcha_response.get('captcha')
        captcha_id = captcha_response.get('id')

        if not captcha_data or not captcha_id:
            raise ValueError("Captcha data or captcha ID is missing.")

        image_bytes = base64.b64decode(captcha_data)
        image_stream = BytesIO(image_bytes)
        captcha_image = Image.open(image_stream)

        print(f"Captcha and ID fetched successfully: {captcha_id}")

        # Save original captcha for reference
        captcha_image.save("original_captcha.jpg")

        return captcha_image, captcha_id

    except requests.exceptions.RequestException as e:
        print(f"Error fetching captcha: {e}")
        return None, None

def replace_pixels_with_white(captcha_img, coordinates_csv):
    """Replace specific pixels in an image with white color to remove noise."""
    coordinates_df = pd.read_csv(coordinates_csv, header=None, names=['coordinates'])
    for _, row in coordinates_df.iterrows():
        x, y = map(int, row['coordinates'].split(','))
        captcha_img.putpixel((x, y), (255, 255, 255))
    print("Noise pixels replaced successfully.")
    return captcha_img

def preprocess_captcha(captcha_img):
    """Preprocess the captcha image for better OCR accuracy."""
    # Convert to grayscale
    gray_image = cv2.cvtColor(np.array(captcha_img), cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding
    thresholded = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 31, 2)

    # Apply dilation to enhance characters
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=1)

    # Apply Gaussian blur to smooth edges
    blurred = cv2.GaussianBlur(dilated, (3, 3), 0)

    # Convert back to PIL format
    processed_img = Image.fromarray(blurred)

    # Save intermediate results for debugging
    processed_img.save("processed_captcha.jpg")

    print("Captcha image processed successfully.")
    return processed_img

def extract_text_from_image(captcha_img):
    """Extract text from an image using EasyOCR with optimized settings."""
    reader = Reader(['en'], gpu=False)
    result = reader.readtext(np.array(captcha_img), allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

    extracted_text = "".join([text for (_, text, _) in result])
    print(f"Extracted Captcha Text: {extracted_text}")

    return extracted_text

def main():
    captcha_url = "https://gateway-voters.eci.gov.in/api/v1/captcha-service/generateCaptcha"
    coordinates_csv = "pixels.csv"

    # Step 1: Fetch captcha image and ID
    captcha_img, captcha_id = fetch_captcha_and_id(captcha_url)

    if captcha_img is None or captcha_id is None:
        print("Failed to fetch captcha and ID. Exiting.")
        return

    # Save the captcha and ID in variables
    saved_captcha_image = captcha_img
    saved_captcha_id = captcha_id
    print(f"Saved Captcha ID: {saved_captcha_id}")

    # Step 2: Remove noise by replacing specific pixels
    modified_img = replace_pixels_with_white(saved_captcha_image, coordinates_csv)

    # Step 3: Preprocess the image for better OCR results
    final_img = preprocess_captcha(modified_img)

    # Step 4: Extract text from the processed captcha image
    extracted_captcha_text = extract_text_from_image(final_img)

if __name__ == "__main__":
    main()
