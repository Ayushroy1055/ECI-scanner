import os
import cv2
import json
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import base64
import psycopg2
import re
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
from easyocr import Reader
from psycopg2 import sql
from sympy import true
import time

# Specify paths for required tools
poppler_path = r"C:/Program Files/Poppler/poppler-24.08.0/Library/bin"
tesseract_path = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Database connection details
DB_HOST = "localhost"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "admin"

# Constants for API request
IS_PORTAL = True
SECURITY_KEY = "na"

# EPIC ID Regex Pattern
EPIC_ID_PATTERN = r"[A-Z]{3}[0-9]{7}"

# List of columns to be added to the 'epic_ids' table
columns_to_add = [
    "applicant_first_name", "applicant_first_name_l1", "applicant_last_name", "applicant_last_name_l1",
    "relation_name", "relation_name_l1", "age", "gender", "part_number", "part_id", "part_name", 
    "part_name_l1", "part_serial_number", "asmbly_name", "asmbly_name_l1", "ac_id", "ac_number", 
    "prlmnt_name", "district_value", "state_name", "is_active", "created_dttm", "modified_dttm", 
    "ps_building_name", "ps_room_details", "full_name", "relative_full_name", "building_address"
]



# EPIC ID Extraction Function
def extract_epic_id(text):
    if re.search(EPIC_ID_PATTERN, text):
        return re.search(EPIC_ID_PATTERN, text).group()
    return None

# Database connection setup
def setup_database():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    with conn.cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS epic_ids (
            id SERIAL PRIMARY KEY,
            epic_id TEXT NOT NULL,
            fetched_status BOOLEAN DEFAULT FALSE
        )
        """)
        conn.commit()
    return conn

# Add Columns to Database Table
def add_column_to_table(db_params, table_name, column_name, column_type, default_value=None):
    try:
        conn = psycopg2.connect(
            host=db_params['host'],
            database=db_params['database'],
            user=db_params['user'],
            password=db_params['password']
        )
        cursor = conn.cursor()
        query = sql.SQL("ALTER TABLE {table} ADD COLUMN {column} {type}").format(
            table=sql.Identifier(table_name),
            column=sql.Identifier(column_name),
            type=sql.SQL(column_type)
        )
        if default_value is not None:
            query += sql.SQL(" DEFAULT {default_value}").format(default_value=sql.SQL(str(default_value)))
        cursor.execute(query)
        conn.commit()
        print(f"Column '{column_name}' added successfully to '{table_name}' table.")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

# Fetch Captcha and ID from URL
def fetch_captcha_and_id(captcha_url):
    try:
        # Send GET request to the captcha URL
        response = requests.get(captcha_url)
        response.raise_for_status()  # Raise an error for bad status codes (4xx, 5xx)

        # Parse the JSON response
        captcha_response = response.json()
        captcha_data = captcha_response.get('captcha')  # Base64 encoded captcha image
        captcha_id = captcha_response.get('id')  # Captcha ID

        if not captcha_data or not captcha_id:
            raise ValueError("Captcha data or captcha ID is missing in the response.")

        # Decode the captcha image
        image_bytes = base64.b64decode(captcha_data)
        image_stream = BytesIO(image_bytes)
        captcha_image = Image.open(image_stream)
        captcha_image.save("captcha.jpg")
        return captcha_image, captcha_id
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching captcha: {e}")
        return None, None

def replace_pixels_with_white(captcha_img, coordinates_csv):
    """Replace specific pixels in an image with white color."""
    coordinates_df = pd.read_csv(coordinates_csv, header=None, names=['coordinates'])
    for _, row in coordinates_df.iterrows():
        x, y = map(int, row['coordinates'].split(','))
        captcha_img.putpixel((x, y), (255, 255, 255))
    return captcha_img

def adjust_image(captcha_img):
    """Adjust brightness, contrast, and sharpen the image."""
    gray_image = cv2.cvtColor(np.array(captcha_img), cv2.COLOR_RGB2GRAY)
    alpha, beta = 1.5, 10
    adjusted_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)
    blurred_image = cv2.GaussianBlur(adjusted_image, (0, 0), 3)
    sharpened_image = cv2.addWeighted(adjusted_image, 1.5, blurred_image, -0.5, 0)
    smoothed_image_result = cv2.bilateralFilter(sharpened_image, 9, 75, 75)
    return Image.fromarray(smoothed_image_result)

def extract_text_from_image(captcha_img):
    """Extract text from an image using EasyOCR."""
    reader = Reader(['en'], gpu=False)
    result = reader.readtext(np.array(captcha_img), allowlist="abcdefghijklmnopqrstuvwxyz0123456789")
    extracted_text = "".join([text for (_, text, _) in result])
    print(f"Text: {extracted_text}")
    return extracted_text
                
# Convert PDF to Images with Boundaries
def convert_pdf_to_images_with_boundaries(pdf_file_path, boundary_output_folder):
    os.makedirs(boundary_output_folder, exist_ok=True)
    images = convert_from_path(pdf_file_path, poppler_path=poppler_path)
    for i, page in enumerate(images):
        open_cv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        boundary_image_path = os.path.join(boundary_output_folder, f"{i+1}.png")
        draw_and_save_green_boundaries(open_cv_image, boundary_image_path)
        print(f"Processed and saved: {boundary_image_path}")

def draw_and_save_green_boundaries(image, output_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 20:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(output_path, image)

# Extract and Save EPIC IDs from Blocks
def extract_and_save_epic_ids(image_path, db_connection):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    with db_connection.cursor() as cursor:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 20:
                cropped_box = image[y:y+h, x:x+w]
                cropped_box_rgb = cv2.cvtColor(cropped_box, cv2.COLOR_BGR2RGB)
                text = pytesseract.image_to_string(cropped_box_rgb, config='--psm 6')
                epic_id = extract_epic_id(text)
                if epic_id:
                    cursor.execute("INSERT INTO epic_ids (epic_id, fetched_status) VALUES (%s, %s)", (epic_id, False))
        db_connection.commit()
    print(f"EPIC IDs extracted and saved to database from {image_path}")

#extract the epic id info from eci api url
def fetch_epic_id_info(extracted_text, captcha_id, epic_id,IS_PORTAL,SECURITY_KEY):
    
    url = "https://gateway-voters.eci.gov.in/api/v1/elastic/search-by-epic-from-national-display"
    
    payload = {
        
        "captchaData": extracted_text,
        "captchaId": captcha_id,
        "epicNumber": epic_id,
        "isPortal": IS_PORTAL,
        "securityKey": SECURITY_KEY
    }

    
        
    response = requests.post(url, json=payload)
    print(response)
    if response.status_code == 200:
        json_data = response.json()  # Convert response to Python dictionary
        print("Data fetched successfully!")
        print(json_data)
        return json_data
    else:
        print(f"Error fetching data: {response.status_code}, Message: {response.text}")
        json_data = None
    
# Main Function
def main():
    captcha_url = "https://gateway-voters.eci.gov.in/api/v1/captcha-service/generateCaptcha"
    pdf_file_path = r"D:/ayush/CapApi/voterlist.pdf"
    boundary_output_folder = r"path_to_output_folder/voter/pages"
    coordinates_csv = r"pixels.csv"

    # Step 1: Convert PDF to Images with Boundaries and Extract EPIC IDs
    convert_pdf_to_images_with_boundaries(pdf_file_path, boundary_output_folder)
    
    # Step 2: Setup Database and Add Columns
    db_conn = setup_database()
    for column in columns_to_add:
        add_column_to_table({
            'host': DB_HOST,
            'database': DB_NAME,
            'user': DB_USER,
            'password': DB_PASSWORD
        }, 'epic_ids', column, 'TEXT')

    # Step 3: Extract and Save EPIC IDs from Boundary Images
    for boundary_image_file in os.listdir(boundary_output_folder):
        boundary_image_path = os.path.join(boundary_output_folder, boundary_image_file)
        extract_and_save_epic_ids(boundary_image_path, db_conn)


    # Step 8: Fetch EPIC IDs sequentially from DB
    with db_conn.cursor() as cursor:
        cursor.execute("SELECT id , epic_id FROM epic_ids WHERE fetched_status = false ORDER BY id ASC")
        rows = cursor.fetchall()

        for row in rows:
            epic_id = row[1]
            print(f"Processing EPIC ID: {epic_id}")

            # Loop to fetch captcha and process for each EPIC ID
            captcha_img, captcha_id = fetch_captcha_and_id(captcha_url)


            if captcha_img is None or captcha_id is None:
                print("Failed to fetch captcha and ID. Skipping this EPIC ID.")
                continue  # Skip if there's an issue with captcha fetching
            
            # Step 5: Replace pixels in captcha image
            modified_img = replace_pixels_with_white(captcha_img, coordinates_csv)

            # Step 6: Adjust and smooth the image
            final_img = adjust_image(modified_img)

            # Step 7: Extract text from the processed image
            captcha_text = extract_text_from_image(final_img)

            # Step 9: Fetch EPIC ID Info
            epicIdInfo = fetch_epic_id_info(captcha_text, captcha_id, epic_id,IS_PORTAL,SECURITY_KEY)
            if epicIdInfo:
                print(f"Fetched EPIC ID Info: {epicIdInfo}")

                # Update the fetched status in DB
                cursor.execute("""
                    UPDATE epic_ids
                    SET fetched_status = TRUE
                    WHERE epic_id = %s
                """, (epic_id,))
                db_conn.commit()
             # Sleep for 10 seconds before processing the next EPIC ID
        time.sleep(15)
                
    # Close the DB connection
    db_conn.close()

if __name__ == "__main__":
    main()
