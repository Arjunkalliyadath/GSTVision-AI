import logging
logging.basicConfig(level=logging.INFO)
import sys
import os

sys.path.append(r'c:\DSA\Intern2\GST2_Converter\backend\fastapi_service')
from services.got_ocr import get_got_ocr

print('Attempting to load GOT-OCR 2.0...')
got = get_got_ocr()
got.load()

if got._loaded:
    print('GOT-OCR loaded successfully!')
else:
    print('Failed to load GOT-OCR.')
