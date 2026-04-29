import cv2
import re
import numpy as np
from PIL import Image, ImageEnhance
import logging
from sympy import false
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import paddleocr
import easyocr
import pytesseract
from typing import List, Tuple, Dict, Optional
import requests
import base64
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModernEgyptianIDOCR:
    """Advanced Egyptian ID OCR using multiple state-of-the-art models"""
    
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all OCR models"""
        try:
            # 1. TrOCR for high-quality text recognition
            logger.info("Loading TrOCR model...")
            self.processors['trocr'] = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
            self.models['trocr'] = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
            
            # For Arabic text, use multilingual model
            self.processors['trocr_multilingual'] = TrOCRProcessor.from_pretrained('microsoft/trocr-base-multilingual')
            self.models['trocr_multilingual'] = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-multilingual')
            
        except Exception as e:
            logger.warning(f"Failed to load TrOCR: {e}")
        
        try:
            # 2. PaddleOCR for multilingual support
            logger.info("Loading PaddleOCR...")
            self.models['paddle_ar'] = paddleocr.PaddleOCR(use_angle_cls=True, lang='ar', use_gpu=false)
            self.models['paddle_en'] = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', use_gpu=false)
            
        except Exception as e:
            logger.warning(f"Failed to load PaddleOCR: {e}")
        
        try:
            # 3. EasyOCR as fallback
            logger.info("Loading EasyOCR...")
            self.models['easy_ar'] = easyocr.Reader(['ar'], gpu=False)
            self.models['easy_en'] = easyocr.Reader(['en'], gpu=false)
            self.models['easy_mixed'] = easyocr.Reader(['ar', 'en'], gpu=false)
            
        except Exception as e:
            logger.warning(f"Failed to load EasyOCR: {e}")

    def preprocess_image(self, image: np.ndarray, method: str = 'advanced') -> np.ndarray:
        """Advanced image preprocessing for better OCR results"""
        try:
            if method == 'advanced':
                # Convert to PIL for better processing
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                # Enhance image quality
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(2.0)
                
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.5)
                
                # Convert back to OpenCV
                enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                # Additional OpenCV processing
                gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                
                # Noise reduction
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                
                # CLAHE for better contrast
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                final = clahe.apply(denoised)
                
                return final
            
            else:  # basic preprocessing
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                
                # Basic enhancement
                enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
                return enhanced
                
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return image

    def extract_text_trocr(self, image: np.ndarray, bbox: List[int], lang: str = 'en') -> str:
        """Extract text using TrOCR"""
        try:
            x1, y1, x2, y2 = bbox
            if x1 >= x2 or y1 >= y2:
                return ""
            
            # Crop and preprocess
            cropped = image[y1:y2, x1:x2]
            if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                return ""
            
            # Preprocess for TrOCR
            processed = self.preprocess_image(cropped, 'advanced')
            pil_image = Image.fromarray(processed).convert('RGB')
            
            # Choose appropriate model
            model_key = 'trocr_multilingual' if lang == 'ar' else 'trocr'
            
            if model_key in self.models and model_key in self.processors:
                # Generate text
                pixel_values = self.processors[model_key](pil_image, return_tensors="pt").pixel_values
                generated_ids = self.models[model_key].generate(pixel_values)
                generated_text = self.processors[model_key].batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                logger.info(f"TrOCR result: {generated_text}")
                return generated_text.strip()
            
            return ""
            
        except Exception as e:
            logger.warning(f"TrOCR extraction failed: {e}")
            return ""

    def extract_text_paddle(self, image: np.ndarray, bbox: List[int], lang: str = 'en') -> str:
        """Extract text using PaddleOCR"""
        try:
            x1, y1, x2, y2 = bbox
            if x1 >= x2 or y1 >= y2:
                return ""
            
            cropped = image[y1:y2, x1:x2]
            if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                return ""
            
            # Choose appropriate model
            model_key = 'paddle_ar' if lang == 'ar' else 'paddle_en'
            
            if model_key in self.models:
                results = self.models[model_key].ocr(cropped, det=False, cls=False)
                
                if results and results[0]:
                    text = ' '.join([item[0] for item in results[0] if item[1] > 0.5])  # Filter by confidence
                    logger.info(f"PaddleOCR result: {text}")
                    return text.strip()
            
            return ""
            
        except Exception as e:
            logger.warning(f"PaddleOCR extraction failed: {e}")
            return ""

    def extract_text_easyocr(self, image: np.ndarray, bbox: List[int], lang: str = 'en') -> str:
        """Extract text using EasyOCR"""
        try:
            x1, y1, x2, y2 = bbox
            if x1 >= x2 or y1 >= y2:
                return ""
            
            cropped = image[y1:y2, x1:x2]
            if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                return ""
            
            processed = self.preprocess_image(cropped, 'advanced')
            
            # Choose appropriate model
            if lang == 'ar':
                model_key = 'easy_ar'
            elif lang == 'mixed':
                model_key = 'easy_mixed'
            else:
                model_key = 'easy_en'
            
            if model_key in self.models:
                results = self.models[model_key].readtext(processed, detail=0, paragraph=True)
                if results:
                    text = ' '.join(results).strip()
                    logger.info(f"EasyOCR result: {text}")
                    return text
            
            return ""
            
        except Exception as e:
            logger.warning(f"EasyOCR extraction failed: {e}")
            return ""

    def extract_text_ensemble(self, image: np.ndarray, bbox: List[int], lang: str = 'en') -> str:
        """Extract text using ensemble of multiple OCR methods"""
        results = []
        
        # Try all available methods
        methods = [
            ('TrOCR', self.extract_text_trocr),
            ('PaddleOCR', self.extract_text_paddle),
            ('EasyOCR', self.extract_text_easyocr)
        ]
        
        for method_name, method_func in methods:
            try:
                result = method_func(image, bbox, lang)
                if result and len(result.strip()) > 1:
                    results.append((method_name, result.strip()))
            except Exception as e:
                logger.warning(f"{method_name} failed: {e}")
                continue
        
        if not results:
            return ""
        
        # Simple voting: return the most common result, or the first good one
        if len(results) == 1:
            return results[0][1]
        
        # For now, return the first result (you can implement more sophisticated voting)
        logger.info(f"Ensemble results: {results}")
        return results[0][1]

    def detect_egyptian_id_advanced(self, image: np.ndarray) -> str:
        """Advanced Egyptian ID detection using multiple methods"""
        try:
            # Method 1: Try full image OCR with number filtering
            methods = ['paddle_en', 'easy_en', 'easy_mixed']
            
            for method in methods:
                try:
                    if method.startswith('paddle') and method in self.models:
                        results = self.models[method].ocr(image, det=True, cls=False)
                        
                        if results and results[0]:
                            for item in results[0]:
                                text = item[1][0] if len(item[1]) > 0 else ""
                                confidence = item[1][1] if len(item[1]) > 1 else 0
                                
                                if confidence > 0.5:
                                    # Look for 14-digit numbers
                                    numbers = re.findall(r'\d{14}', text)
                                    if numbers:
                                        logger.info(f"Found ID via {method}: {numbers[0]}")
                                        return numbers[0]
                                    
                                    # Look for numbers with separators
                                    cleaned = re.sub(r'[^\d]', '', text)
                                    if len(cleaned) == 14 and cleaned.isdigit():
                                        logger.info(f"Found ID via {method} (cleaned): {cleaned}")
                                        return cleaned
                    
                    elif method.startswith('easy') and method in self.models:
                        results = self.models[method].readtext(image, detail=0)
                        
                        for text in results:
                            # Look for 14-digit numbers
                            numbers = re.findall(r'\d{14}', text)
                            if numbers:
                                logger.info(f"Found ID via {method}: {numbers[0]}")
                                return numbers[0]
                            
                            # Look for numbers with separators
                            cleaned = re.sub(r'[^\d]', '', text)
                            if len(cleaned) == 14 and cleaned.isdigit():
                                logger.info(f"Found ID via {method} (cleaned): {cleaned}")
                                return cleaned
                
                except Exception as e:
                    logger.warning(f"Method {method} failed: {e}")
                    continue
            
            logger.error("All ID detection methods failed")
            return ""
            
        except Exception as e:
            logger.error(f"ID detection failed: {e}")
            return ""

    def decode_egyptian_id_safe(self, id_number: str) -> Dict[str, str]:
        """Safely decode Egyptian ID with comprehensive error handling"""
        try:
            # Input validation
            if not id_number:
                logger.warning("Empty ID number provided")
                return {"Birth Date": "Unknown", "Governorate": "Unknown", "Gender": "Unknown"}
            
            # Clean the ID number
            cleaned_id = re.sub(r'[^\d]', '', str(id_number))
            
            # Validate length
            if len(cleaned_id) != 14:
                logger.warning(f"Invalid ID length: {len(cleaned_id)} (expected 14), ID: {cleaned_id}")
                return {"Birth Date": "Unknown", "Governorate": "Unknown", "Gender": "Unknown"}
            
            # Validate all digits
            if not cleaned_id.isdigit():
                logger.warning(f"ID contains non-digit characters: {cleaned_id}")
                return {"Birth Date": "Unknown", "Governorate": "Unknown", "Gender": "Unknown"}
            
            # Egyptian governorates mapping
            governorates = {
                '01': 'Cairo', '02': 'Alexandria', '03': 'Port Said', '04': 'Suez',
                '11': 'Damietta', '12': 'Dakahlia', '13': 'Ash Sharqia', '14': 'Kaliobeya',
                '15': 'Kafr El-Sheikh', '16': 'Gharbia', '17': 'Monoufia', '18': 'El Beheira',
                '19': 'Ismailia', '21': 'Giza', '22': 'Beni Suef', '23': 'Fayoum',
                '24': 'El Menia', '25': 'Assiut', '26': 'Sohag', '27': 'Qena',
                '28': 'Aswan', '29': 'Luxor', '31': 'Red Sea', '32': 'New Valley',
                '33': 'Matrouh', '34': 'North Sinai', '35': 'South Sinai', '88': 'Foreign'
            }
            
            # Extract components with bounds checking
            try:
                century_digit = int(cleaned_id[0])
                year = int(cleaned_id[1:3])
                month = int(cleaned_id[3:5])
                day = int(cleaned_id[5:7])
                governorate_code = cleaned_id[7:9]
                gender_code = int(cleaned_id[12]) if len(cleaned_id) >= 13 else 0
            except (IndexError, ValueError) as e:
                logger.error(f"Failed to parse ID components: {e}")
                return {"Birth Date": "Unknown", "Governorate": "Unknown", "Gender": "Unknown"}
            
            # Determine full year
            if century_digit == 2:
                full_year = 1900 + year
            elif century_digit == 3:
                full_year = 2000 + year
            else:
                logger.warning(f"Invalid century digit: {century_digit}")
                full_year = 1900 + year  # Default fallback
            
            # Validate date components
            if not (1 <= month <= 12):
                logger.warning(f"Invalid month: {month}")
                month = 1
            
            if not (1 <= day <= 31):
                logger.warning(f"Invalid day: {day}")
                day = 1
            
            # Determine gender
            gender = "Male" if gender_code % 2 != 0 else "Female"
            
            # Get governorate
            governorate = governorates.get(governorate_code, f"Unknown ({governorate_code})")
            
            # Format birth date
            birth_date = f"{full_year:04d}-{month:02d}-{day:02d}"
            
            logger.info(f"Successfully decoded ID: {cleaned_id} -> {birth_date}, {governorate}, {gender}")
            
            return {
                'Birth Date': birth_date,
                'Governorate': governorate,
                'Gender': gender
            }
            
        except Exception as e:
            logger.error(f"ID decoding failed for ID '{id_number}': {e}")
            return {"Birth Date": "Unknown", "Governorate": "Unknown", "Gender": "Unknown"}

# Initialize the modern OCR system
modern_ocr = ModernEgyptianIDOCR()

def process_egyptian_id_modern(image_path: str) -> Tuple[str, ...]:
    """Process Egyptian ID using modern OCR methods"""
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return ("", "", "", "", "", "Unknown", "Unknown", "Unknown")
        
        logger.info(f"Image loaded successfully. Shape: {image.shape}")
        
        # For this example, we'll assume we have bounding boxes from YOLO or similar
        # In practice, you'd run object detection first to get these boxes
        
        # Try to detect the National ID number first
        national_id = modern_ocr.detect_egyptian_id_advanced(image)
        
        # Extract other fields (you'll need to adapt this based on your field detection)
        # This is a placeholder - you'd need to detect field regions first
        first_name = ""
        last_name = ""
        address = ""
        
        # For demonstration, let's assume we have some bounding boxes
        # In practice, these would come from your object detection model
        
        # Decode the national ID
        decoded_info = modern_ocr.decode_egyptian_id_safe(national_id)
        
        return (
            first_name,
            last_name,
            f"{first_name} {last_name}".strip(),
            national_id,
            address,
            decoded_info["Birth Date"],
            decoded_info["Governorate"],
            decoded_info["Gender"]
        )
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return ("", "", "", "", "", "Unknown", "Unknown", "Unknown")

# Example usage
if __name__ == "__main__":
    # Test the modern OCR system
    result = process_egyptian_id_modern("path_to_your_id_image.jpg")
    print("Extracted Information:")
    print(f"First Name: {result[0]}")
    print(f"Last Name: {result[1]}")
    print(f"Full Name: {result[2]}")
    print(f"National ID: {result[3]}")
    print(f"Address: {result[4]}")
    print(f"Birth Date: {result[5]}")
    print(f"Governorate: {result[6]}")
    print(f"Gender: {result[7]}")