import fitz
import os

pdf_path = 'ref/2510.04871v1.pdf'
doc = fitz.open(pdf_path)
text = ''
for page in doc:
    text += page.get_text()
print(text)
