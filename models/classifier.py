"""
Food Image Classifier Module (Hybrid Version)
=============================================

This module implements a hybrid classification strategy:
1.  **Primary**: Queries `Kaludi/food-category-classification-v2.0` (Food-101). 
    Excel at prepared dishes (Pizza, Sushi, Apple Pie).
2.  **Fallback**: If Primary confidence is low (< 20%), queries `google/efficientnet-b0` (ImageNet).
    Excels at raw ingredients (Banana, Apple, Milk Carton).

This ensures the user can upload both raw grocery items and cooked meals.
"""

import requests
import time
import json
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, List

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TARGET_CATEGORIES,
    HUGGINGFACE_API_URL_FOOD101,
    HUGGINGFACE_API_URL_IMAGENET,
    get_huggingface_headers,  # Lazy-loaded for Railway compatibility
    DEFAULT_CATEGORY,
    FOOD101_TO_CATEGORY,
)

# =============================================================================
# MAPPING LOGIC
# =============================================================================

# ImageNet Mapping (for raw ingredients)
IMAGENET_KEYWORD_MAPPING = {
    "apple": ["apple", "granny_smith", "pomegranate"],
    "banana": ["banana", "plantain"],
    "bread": ["bread", "loaf", "bagel", "pretzel", "bun", "roll", "bakery", "toast", "croissant"],
    "milk": ["milk", "cream", "pitcher", "jug", "bottle", "carton", "white"],
    "pasta": ["pasta", "spaghetti", "macaroni", "noodle", "lasagna", "ravioli", "carbonara", "vermicelli"],
    "pizza": ["pizza", "flatbread"],
    "burger": ["burger", "cheeseburger", "hamburger", "sandwich"],
    "sushi": ["sushi", "sashimi", "roll", "seafood"],
    "meat": ["meat", "steak", "beef", "pork", "chop", "rib", "roast", "chicken", "turkey", "lamb"],
    "fish": ["fish", "salmon", "tuna", "cod", "trout", "seafood", "bass"],
    "egg": ["egg", "omelet", "omelette"],
    "vegetable": ["vegetable", "salad", "carrot", "broccoli", "cucumber", "tomato", "pepper", "corn", "spinach"],
    "fruit": ["fruit", "berry", "strawberry", "orange", "lemon", "lime", "grape", "melon"],
    "rice": ["rice", "risotto", "grain"],
    "cake": ["cake", "dessert", "muffin", "cupcake", "chocolate"]
}

class FoodClassifier:
    """Hybrid Classifier: Food-101 + ImageNet"""
    
    def __init__(self, device: str = None):
        print("Initializing Hybrid FoodClassifier...")
        print(f"Primary (Food-101): {HUGGINGFACE_API_URL_FOOD101}")
        print(f"Fallback (ImageNet): {HUGGINGFACE_API_URL_IMAGENET}")
        
    def _query(self, url: str, image_bytes: bytes) -> List[Dict]:
        """Generic API query function."""
        max_retries = 3
        headers = get_huggingface_headers()  # Get fresh headers with token at request time
        
        # Debug logging for Railway
        print(f"DEBUG: Headers keys: {list(headers.keys())}")
        if 'Authorization' in headers:
            token_preview = headers['Authorization'][:20] + "..."
            print(f"DEBUG: Auth header present: {token_preview}")
        else:
            print("DEBUG: NO Authorization header - token not found!")
        
        for i in range(max_retries):
            try:
                response = requests.post(url, headers=headers, data=image_bytes)
                if response.status_code == 200:
                    try: return response.json()
                    except json.JSONDecodeError: raise RuntimeError(f"Invalid JSON: {response.text}")
                
                # Handle loading error
                try:
                    err = response.json()
                    if "error" in err and "loading" in err["error"].lower():
                        wait = err.get("estimated_time", 5)
                        print(f"Model loading ({url})... waiting {wait}s")
                        time.sleep(wait)
                        continue
                except: pass
                
                # Return 'None' to indicate failure to the caller, don't raise here yet
                # allowing the hybrid logic to decide if it should try the other model
                print(f"API Warning ({url}): {response.status_code} - {response.text}")
                return None 
                
            except Exception as e:
                print(f"Network error ({url}): {e}")
                time.sleep(1)
        return None

    def classify(self, image: Union[str, Path, bytes]) -> Tuple[str, float, Dict[str, float]]:
        # 1. Prepare Bytes
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f: kb = f.read()
        elif hasattr(image, "read"):
            if hasattr(image, "seek"): image.seek(0)
            kb = image.read()
            if not kb: 
                image.seek(0) 
                kb = image.read()
        elif hasattr(image, "save"):
            from io import BytesIO
            buf = BytesIO()
            image.save(buf, format="JPEG")
            kb = buf.getvalue()
        else: kb = image

        # 2. Try Primary Model (Food-101)
        print("DEBUG: Querying Primary (Food-101)...")
        preds_food = self._query(HUGGINGFACE_API_URL_FOOD101, kb)
        
        best_cat, conf, scores = self._process_food101(preds_food)
        print(f"DEBUG: Primary Result: {best_cat} ({conf:.2f})")
        
        # 3. Hybrid Decision Logic
        # If Food-101 failed (None) OR it's very unsure (< 25%), try ImageNet
        if preds_food is None or conf < 0.25:
            print("DEBUG: Low confidence or failure. Switch to Fallback (ImageNet)...")
            preds_imagenet = self._query(HUGGINGFACE_API_URL_IMAGENET, kb)
            
            if preds_imagenet:
                cat_img, conf_img, scores_img = self._process_imagenet(preds_imagenet)
                print(f"DEBUG: Fallback Result: {cat_img} ({conf_img:.2f})")
                
                # If ImageNet is more confident, use it
                if conf_img > conf:
                    print(f"DEBUG: Using ImageNet result (Confidence {conf_img:.2f} > {conf:.2f})")
                    return cat_img, conf_img, scores_img
        
        # Default to Food-101 result (or default if both failed)
        return best_cat, conf, scores

    def _process_food101(self, preds) -> Tuple[str, float, Dict[str, float]]:
        scores = {c: 0.0 for c in TARGET_CATEGORIES}
        if not preds or not isinstance(preds, list): return DEFAULT_CATEGORY, 0.0, scores
        
        for p in preds:
            lbl = p.get('label', '').lower().replace(' ', '_')
            sc = p.get('score', 0.0)
            target = FOOD101_TO_CATEGORY.get(lbl)
            if target: scores[target] += sc
            
        return self._normalize(scores)

    def _process_imagenet(self, preds) -> Tuple[str, float, Dict[str, float]]:
        scores = {c: 0.0 for c in TARGET_CATEGORIES}
        if not preds or not isinstance(preds, list): return DEFAULT_CATEGORY, 0.0, scores

        for p in preds:
            lbl = p.get('label', '').lower()
            sc = p.get('score', 0.0)
            for target, kws in IMAGENET_KEYWORD_MAPPING.items():
                if any(k in lbl for k in kws):
                    scores[target] += sc
                    break
        
        return self._normalize(scores)

    def _normalize(self, scores):
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k,v in scores.items()}
            best = max(scores, key=scores.get)
            return best, scores[best], scores
        return DEFAULT_CATEGORY, 0.0, scores

# Convenience
_inst = None
def load_food_classifier(device=None):
    global _inst
    if _inst is None: _inst = FoodClassifier(device)
    return _inst

def classify_food(img, clf=None):
    if clf is None: clf = load_food_classifier()
    return clf.classify(img)
