# Document Text and Entity Extraction

This repository contains a Streamlit application for extracting text and entities from PDF documents, focusing on Indian ID cards (PAN, Aadhaar, Passport, Driving License, Voter ID). It leverages text extraction, OCR, and entity recognition for both digital and scanned documents.

## Projects Overview

1. **Document Text Extraction**
   - Extracts text from digital PDFs using PyMuPDF and scanned PDFs using PaddleOCR with high-DPI settings.
   - Supports automatic document type detection (e.g., PAN, Aadhaar) based on text patterns.
   - Uses PaddleOCR for its superior accuracy, robustness to complex layouts, and multilingual support, compared to KerasOCR and Tesseract.

2. **Entity Extraction**
   - Employs spaCy for named entity recognition and regex for precise extraction of entities like names, IDs, and dates.
   - Tailored functions for extracting specific fields from PAN, Aadhaar, Passport, Driving License, and Voter ID.
   - Handles variations in document formats using pattern-based matching.

3. **Streamlit Interface**
   - Provides a web-based UI for uploading PDFs, displaying documents, and showing extracted data in JSON format.
   - Includes error handling and logging for robust processing and debugging.

## Dependencies
- Python 3.11.0 or above
- PyMuPDF, spaCy, PaddleOCR, pytesseract, Streamlit
- Fonts (e.g., DejaVuSans)

## Instructions
- Run in Google Colab or locally with Streamlit:
  ```bash
  pip install pymupdf spacy paddleocr pytesseract streamlit
  python -m spacy download en_core_web_sm
  streamlit run app.py
