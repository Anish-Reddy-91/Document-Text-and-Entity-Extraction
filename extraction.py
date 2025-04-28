import streamlit as st
import cv2
import numpy as np
from PIL import Image
import spacy
from paddleocr import PaddleOCR
import re

# Load PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load spaCy's pre-trained model for NLP entity extraction
nlp = spacy.load('en_core_web_sm')

# Function to perform OCR using PaddleOCR
def perform_ocr(image_array):
    ocr_result = ocr.ocr(image_array, cls=True)
    text = "\n".join([line[1][0] for line in ocr_result[0]])
    print("extracted ocr text: ", text)
    print("--------------------")
    return text

# Function to determine the card type based on ID number
def determine_card_type(text):
    patterns = {
        'passport': r'\b([A-Z]\d{7,8})\b',
        'aadhaar': r'\b\d{4} \d{4} \d{4}\b|\b\d{8} \d{4}\b|\b\d{12}\b',
        'pan': r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
        'driving_license': r'\b(([A-Z]{2})\d{14})\b'
    }
    
    for card_type, pattern in patterns.items():
        if re.search(pattern, text):
            return card_type
    return None

# Function to extract entities based on card type using NLP and regex
def extract_entities_by_card_type(text, card_type):
    doc = nlp(text)
    
    #------pan------#
    if card_type == 'pan':
        print("PAN DETAILS:")
        extracted_data = {
            "Name": "",
            "Father's Name": "",
            "Date of Birth": "",
            "ID Number": ""
        }
        
        pan_number = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', text)
        if pan_number:
            extracted_data["ID Number"] = pan_number.group()

        dob_match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', text)
        if dob_match:
            extracted_data["Date of Birth"] = dob_match.group()

        name_match = re.search(r'Name\s*([A-Z\s]+)', text)
        if name_match:
            extracted_data["Name"] = name_match.group(1).strip()

        father_name_match = re.search(r"Father['s]* Name\s*([A-Za-z\s]+)", text)
        if father_name_match:
            extracted_data["Father's Name"] = father_name_match.group(1).strip()

        if extracted_data["Name"] == "" or extracted_data["Father's Name"] == "":
            person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if len(person_names) >= 2:
                extracted_data["Name"] = person_names[0].strip()  
                extracted_data["Father's Name"] = person_names[1].strip() 

    #-----driving-license-----#
    elif card_type == 'driving_license':
        print("DRIVERS DETAILS:")
        extracted_data = {
            "Name": "",
            "Father's/Husband's Name": "",
            "Date of Birth": "",
            "ID Number": "",
            "Date of Issue": "",
            "Valid Till": "",
            "Blood Group": "",
            "State of Issue": "",
            "Address": ""
        }

        license_number = re.search(r'\b(([A-Z]{2})\d{14})\b', text)
        if license_number:
            extracted_data["ID Number"] = license_number.group()

        dob_match = re.search(r'[Dd][Aa]te\s*[0o]f\s*[Bb]irth\s*:\s*(\d{2}/\w{2}/\d{4})|[Dd]are\s*[0o]f\s*[Bb]irtn\s*:\s*(\d{2}/\w{2}/\d{4})', text)
        if dob_match:
            extracted_data["Date of Birth"] = dob_match.group(1) if dob_match.group(1) else dob_match.group(2)

        issue_valid_match = re.search(r'Issue Date\s*(\d{2}/\d{2}/\d{4})\s*Validity\(TR\)\s*(\d{2}/\d{2}/\d{4})', text)
        if not issue_valid_match:
            issue_valid_match = re.search(r'(\d{2}/\d{2}/\d{4})(\d{2}/\d{2}/\d{4})', text)

        if issue_valid_match:
            extracted_data["Date of Issue"] = issue_valid_match.group(1)
            extracted_data["Valid Till"] = issue_valid_match.group(2)

        name_match = re.search(r'Name:\s*([A-Z\s]+)(?=\s*\n|Holder\'s Signature)', text)
        if name_match:
            extracted_data["Name"] = name_match.group(1).strip()

        father_husband_name_match = re.search(r'Son/Daughter/Wife of:\s*([A-Za-z\s]+)(?=\n|$)', text)
        if father_husband_name_match:
            extracted_data["Father's/Husband's Name"] = father_husband_name_match.group(1).strip()

        address_match = re.search(r'Address:\s*([^0-9\n]*[\w\s.,-]*)', text)
        if address_match:
            extracted_data["Address"] = address_match.group(1).strip().replace('\n', ' ')

        blood_group_match = re.search(r'Blood Group\s*:\s*([A-Z+/-]+)', text)
        if blood_group_match:
            extracted_data["Blood Group"] = blood_group_match.group(1).strip()

        state_match = re.search(r'Issued by\s*([A-Za-z\s]+)', text)
        if state_match:
            extracted_data["State of Issue"] = state_match.group(1).strip()

    #-----aadhaar-----#    
    elif card_type == 'aadhaar':
        print("AADHAAR DETAILS:")
        extracted_data = {
            "Name": "",
            "Father's Name": "",
            "Date of Birth": "",
            "Aadhaar Number": "",
            "Gender": "",
            "Address": "",
            "Pincode": "",
            "Mobile Number": ""
        }

        aadhaar_number = re.search(r'\b\d{4} \d{4} \d{4}\b|\b\d{8} \d{4}\b|\b\d{12}\b', text)
        if aadhaar_number:
            extracted_data["Aadhaar Number"] = aadhaar_number.group().replace(" ", "")

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if not extracted_data["Name"]:
                    extracted_data["Name"] = ent.text.strip()
                elif not extracted_data["Father's Name"]:
                    extracted_data["Father's Name"] = ent.text.strip()

        dob_match = re.search(r'\b(?:DOB|D\.O\.B|D\.O\.B\.)[: ]*(\d{2}/\d{2}/\d{4})\b', text)
        if dob_match:
            extracted_data["Date of Birth"] = dob_match.group(1)

        gender_match = re.search(r'\b(Male|Female)\b', text, re.IGNORECASE)
        if gender_match:
            extracted_data["Gender"] = gender_match.group(1).capitalize()

        pincode_match = re.search(r'\b\d{6}\b', text)
        if pincode_match:
            extracted_data["Pincode"] = pincode_match.group()

        mobile_match = re.search(r'\b\d{10}\b', text)
        if mobile_match:
            extracted_data["Mobile Number"] = mobile_match.group()

        address_match = re.search(r'(?:[A-Za-z\s]+(?:\s(?:Street|Road|Nagar|Colony|Block|Bazar|Sector|Phase|Avenue|Lane|Place|Park|Gully|P.O.))?[, ]*(?:\d{1,}[- ]*)*.*?)(?=\s*PIN Code\s*:\s*\d{6})', text, re.DOTALL)
        if address_match:
            extracted_data["Address"] = address_match.group(0).strip()

    #------passport-----#
    elif card_type == 'passport':
        print("PASSPORT DETAILS:")
        extracted_data = {
            "Full Name": "",
            "Passport Number": "",
            "Country of Issue": "",
            "Date of Birth": "",
            "Gender": "",
            "Date of Issue": "",
            "Expiry Date": "",
            "Nationality": "",
            "Address (Place of Birth)": ""
        }

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                extracted_data["Full Name"] = ent.text.strip()  
                break
            if ent.label_ == "NORP":  # NER label for Nationalities or religions
                extracted_data["Nationality"] = ent.text.strip()
                break  # Exit loop once nationality is found

        surname_match = re.search(r'Surname\s*[:]*\s*([A-Z\s]+)', text)
        given_name_match = re.search(r'Given Names\s*[:]*\s*([A-Z\s]+)', text)

        if surname_match and given_name_match:
            extracted_data["Full Name"] = f"{surname_match.group(1).strip()} {given_name_match.group(1).strip()}"


        if not extracted_data["Nationality"]:
            nationality_match = re.search(r'REPUBLIC OF\s*([A-Z\s]+)', text, re.IGNORECASE)
            if nationality_match:
                extracted_data["Nationality"] = nationality_match.group(1).strip()


        passport_number = re.search(r'\b([A-Z]\d{7,8})\b', text)
        if passport_number:
            extracted_data["Passport Number"] = passport_number.group()

        surname_match = re.search(r'(?:Surname|Surnam[e]*)\s*[:]*\s*([A-Z\s]+)', text)
        given_name_match = re.search(r'(?:Given Names|Given Name)\s*[:]*\s*([A-Z\s]+)', text)
        if surname_match and given_name_match:
            extracted_data["Full Name"] = f"{surname_match.group(1).strip()} {given_name_match.group(1).strip()}"

        dob_match = re.search(r'Date of Birth\s*[:/]*\s*(\d{2}/\d{2}/\d{4})|(\d{2}[-/](\d{2})[-/](\d{4}))', text)
        if dob_match:
            extracted_data["Date of Birth"] = dob_match.group(1) if dob_match.group(1) else dob_match.group(2)


        doi_match = re.search(r'f/Date of Expiry\s*\d{2}/\d{2}/\d{4}\s*(\d{2}/\d{2}/\d{4})', text)
        if doi_match:
            extracted_data["Date of Issue"] = doi_match.group(1)


        expiry_match = re.search(r'Date of Expiry\s*[:/]*\s*(\d{2}/\d{2}/\d{4})', text)
        if expiry_match:
            extracted_data["Expiry Date"] = expiry_match.group(1)

        gender_match = re.search(r'(?<=\d{2}/\d{2}/\d{4})\s*([MF])', text)
        if gender_match:
            extracted_data["Gender"] = gender_match.group(0).strip() 

        country_issue_match = re.search(r'REPUBLIC OF\s*([A-Z\s]+)', text, re.IGNORECASE)
        if country_issue_match:
            extracted_data["Country of Issue"] = country_issue_match.group(1).strip()

        place_of_birth_match = re.search(r'Place of Birth\s*[:/]*\s*([A-Z\s,]+)', text)
        if place_of_birth_match:
            extracted_data["Address (Place of Birth)"] = place_of_birth_match.group(1).strip()
    return extracted_data

# Main function to process ID card
def process_id_card(image_array):
    text = perform_ocr(image_array)
    card_type = determine_card_type(text)

    if card_type:
        extracted_data = extract_entities_by_card_type(text, card_type)
        return card_type, extracted_data
    else:
        return None, None

# Streamlit UI
st.title("OCR & Entity Recognition for ID Cards")

# Upload image
uploaded_file = st.file_uploader("Choose an ID card image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    # Display the uploaded image
    st.image(image, caption='Uploaded ID Card', use_column_width=True)

    # Process the ID card and display results
    card_type, extracted_data = process_id_card(image_array)

    if extracted_data:
        st.subheader(f"Extracted Data for {card_type.capitalize()} Card")
        st.json(extracted_data)

        # Create a downloadable string from the extracted data
        extracted_data_str = f"Extracted Data for {card_type.capitalize()} Card:\n\n" + \
                              "\n".join([f"{key}: {value}" for key, value in extracted_data.items()])

        # Add a download button
        st.download_button(
            label="Download Extracted Data",
            data=extracted_data_str,
            file_name=f"{card_type}_extracted_data.txt",
            mime="text/plain"
        )
    else:
        st.write("No recognized ID card type found.")
