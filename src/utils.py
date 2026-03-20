"""
utils.py
"""
import re
import logging

def get_logger(name:str) -> logging.Logger:
    logger=logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    text=text.lower()

    # normalizing common units 
    text = re.sub(r'(\d+)\s*gb', r'\1gb', text)
    text = re.sub(r'(\d+)\s*tb', r'\1tb', text)
    text = re.sub(r'(\d+)\s*mp', r'\1mp', text)
    text = re.sub(r'(\d+)\s*ghz', r'\1ghz', text)
    text = re.sub(r'(\d+)\s*mhz', r'\1mhz', text)
    text = re.sub(r'(\d+)\s*mah', r'\1mah', text)
    text = re.sub(r'(\d+)"', r'\1inch', text)
    text = re.sub(r'(\d+)\s*inch', r'\1inch', text)
    text = re.sub(r'(\d+)\s*cm', r'\1cm', text)

    # removing special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # collapse extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text