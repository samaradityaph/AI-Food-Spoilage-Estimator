"""
Food Shelf Life Predictor - Flask Web Application
==================================================

A modern web interface for food image classification and shelf life prediction.
Uses Flask backend with REST API and vanilla HTML/CSS/JS frontend.

Features:
- Drag-and-drop image upload
- Automatic food classification with EfficientNet-B0
- Editable food type override
- Dynamic shelf life prediction
- Modern, responsive UI design
- Production-ready security hardening

Usage:
    python app_flask.py
    
Then open: http://localhost:5000
"""

import os
import io
import base64
import logging
from pathlib import Path
from functools import wraps
from collections import defaultdict
import time

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BASE_SHELF_LIFE_REFRIGERATED,
    Q10_COEFFICIENT,
    FROZEN_MULTIPLIER,
)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# Environment detection
IS_PRODUCTION = os.getenv('FLASK_ENV', 'production') == 'production'
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')  # Comma-separated list

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', '60'))  # requests
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', '60'))  # seconds
request_counts = defaultdict(list)  # IP -> list of timestamps

# Maximum upload size (10MB)
MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', str(10 * 1024 * 1024)))

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Configure CORS - restrict in production
if IS_PRODUCTION:
    CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=False)
else:
    CORS(app)  # Allow all in development

# Lazy loaded models
_classifier = None
_predictor = None


# =============================================================================
# SECURITY MIDDLEWARE
# =============================================================================

def get_client_ip():
    """Get client IP, respecting Railway's proxy headers."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    return request.remote_addr or '127.0.0.1'


def rate_limit():
    """Simple in-memory rate limiter."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            client_ip = get_client_ip()
            now = time.time()
            
            # Clean old entries
            request_counts[client_ip] = [
                ts for ts in request_counts[client_ip] 
                if now - ts < RATE_LIMIT_WINDOW
            ]
            
            # Check rate limit
            if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
            
            # Record this request
            request_counts[client_ip].append(now)
            return f(*args, **kwargs)
        return wrapped
    return decorator


@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    # Enable XSS filter
    response.headers['X-XSS-Protection'] = '1; mode=block'
    # Referrer policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # Content Security Policy (allow inline for our simple UI)
    if IS_PRODUCTION:
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self';"
        )
    return response


# =============================================================================
# FOOD-101 CLASS LIST (all 101 categories)
# =============================================================================

FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]

# Default shelf life values for food categories (in days, refrigerated)
# Based on general food science principles
DEFAULT_SHELF_LIFE = {
    # Baked goods (shorter shelf life)
    "apple_pie": 4, "baklava": 14, "bread_pudding": 4, "bruschetta": 2, 
    "cannoli": 3, "carrot_cake": 5, "cheesecake": 5, "chocolate_cake": 5,
    "churros": 2, "cup_cakes": 4, "donuts": 3, "french_toast": 2,
    "garlic_bread": 3, "macarons": 7, "pancakes": 3, "red_velvet_cake": 5,
    "strawberry_shortcake": 3, "tiramisu": 4, "waffles": 3,
    
    # Meat dishes (moderate shelf life when refrigerated)
    "baby_back_ribs": 4, "beef_carpaccio": 2, "beef_tartare": 1, 
    "chicken_curry": 4, "chicken_quesadilla": 3, "chicken_wings": 4,
    "filet_mignon": 3, "grilled_salmon": 3, "hamburger": 3, "hot_dog": 5,
    "lasagna": 5, "peking_duck": 3, "pork_chop": 4, "prime_rib": 4,
    "pulled_pork_sandwich": 4, "steak": 4,
    
    # Seafood (shorter shelf life)
    "ceviche": 2, "crab_cakes": 3, "fish_and_chips": 2, "fried_calamari": 2,
    "lobster_bisque": 3, "lobster_roll_sandwich": 2, "mussels": 2, "oysters": 2,
    "sashimi": 1, "scallops": 2, "shrimp_and_grits": 3, "sushi": 1,
    "tuna_tartare": 1,
    
    # Dairy-based (moderate)
    "cheese_plate": 14, "chocolate_mousse": 3, "creme_brulee": 3,
    "frozen_yogurt": 60, "grilled_cheese_sandwich": 3, "ice_cream": 60,
    "macaroni_and_cheese": 4, "panna_cotta": 4,
    
    # Salads & veggies (short-moderate)
    "beet_salad": 3, "caesar_salad": 2, "caprese_salad": 2, "edamame": 5,
    "greek_salad": 3, "guacamole": 3, "hummus": 7, "seaweed_salad": 4,
    
    # Rice/noodle dishes
    "bibimbap": 3, "fried_rice": 4, "pad_thai": 3, "paella": 3,
    "pho": 3, "ramen": 3, "risotto": 4, "spaghetti_bolognese": 4,
    "spaghetti_carbonara": 3,
    
    # Fried foods (short)
    "falafel": 4, "french_fries": 2, "onion_rings": 2, "samosa": 4,
    "spring_rolls": 3, "takoyaki": 2,
    
    # Soups (moderate)
    "clam_chowder": 4, "french_onion_soup": 4, "hot_and_sour_soup": 4,
    "miso_soup": 3,
    
    # Other
    "beignets": 1, "breakfast_burrito": 3, "club_sandwich": 2,
    "croque_madame": 2, "deviled_eggs": 4, "dumplings": 4,
    "eggs_benedict": 2, "escargots": 2, "foie_gras": 3, "gnocchi": 3,
    "gyoza": 4, "huevos_rancheros": 2, "nachos": 1, "omelette": 3,
    "pizza": 4, "poutine": 2, "ravioli": 4, "tacos": 3,
}

# Fill in any missing with default of 5 days
for food in FOOD101_CLASSES:
    if food not in DEFAULT_SHELF_LIFE:
        DEFAULT_SHELF_LIFE[food] = 5


def get_classifier():
    """Lazy load the food classifier."""
    global _classifier
    if _classifier is None:
        from models.classifier import FoodClassifier
        _classifier = FoodClassifier()
    return _classifier


def get_predictor():
    """Lazy load the shelf life predictor."""
    global _predictor
    if _predictor is None:
        from models.shelf_life_predictor import ShelfLifePredictor
        _predictor = ShelfLifePredictor()
        if not _predictor.load():
            _predictor.train()
    return _predictor


def compute_shelf_life_for_food(
    food_name: str,
    temperature: float,
    humidity: float,
    storage_type: str
) -> float:
    """
    Compute shelf life for any food type using Q10 model.
    
    Args:
        food_name: Name of the food (from Food-101 or custom)
        temperature: Storage temperature in Celsius
        humidity: Relative humidity (30-90%)
        storage_type: 'refrigerated', 'room_temperature', or 'frozen'
    
    Returns:
        Predicted shelf life in days
    """
    # Get base shelf life (default to 5 days if unknown)
    food_key = food_name.lower().replace(" ", "_")
    base_shelf_life = DEFAULT_SHELF_LIFE.get(food_key, 5)
    
    reference_temp = 4.0  # Reference temperature in Celsius
    
    if storage_type == "frozen":
        # Frozen storage extends shelf life significantly
        return base_shelf_life * 8  # ~8x for frozen
    
    # Calculate temperature effect using Q10 model
    temp_diff = temperature - reference_temp
    rate_ratio = Q10_COEFFICIENT ** (temp_diff / 10.0)
    
    # Apply temperature correction
    shelf_life = base_shelf_life / rate_ratio
    
    # Apply humidity correction
    optimal_humidity = 55.0
    humidity_diff = humidity - optimal_humidity
    humidity_factor = 1.0 - (humidity_diff / 100.0) * 0.5
    humidity_factor = max(0.3, min(humidity_factor, 1.2))
    
    shelf_life = shelf_life * humidity_factor
    
    return max(0.5, round(shelf_life, 1))


@app.route('/')
def index():
    """Serve the main page."""
    logger.info(f"Home page accessed from {get_client_ip()}")
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
@rate_limit()
def classify_image():
    """
    Classify an uploaded food image using EfficientNet-B0.
    
    Expects: multipart/form-data with 'image' file
    Returns: JSON with classification results
    """
    global _classifier
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Classify using Hugging Face API (expects bytes)
        if _classifier is not None:
            category, confidence, scores = _classifier.classify(image_bytes)
            
            if category == "unknown":
                return jsonify({
                    'success': True,
                    'detected_food': 'Unknown Food',
                    'confidence': 0.0,
                    'all_scores': {},
                    'note': "Could not identify as Apple, Banana, Bread, or Milk. Please select manually."
                })
            
            return jsonify({
                'success': True,
                'detected_food': category.replace('_', ' ').title(),
                'confidence': round(confidence * 100, 1),
                'all_scores': {k.replace('_', ' ').title(): round(v * 100, 1) for k, v in scores.items()}
            })
        else:
            # Classifier not loaded - fallback
            return jsonify({
                'success': True,
                'detected_food': 'Pizza',
                'confidence': 30.0,
                'all_scores': {},
                'note': 'Model not loaded. Please enter food type manually.'
            })
        
    except Exception as e:
        print(f"Classification error: {e}")
        import traceback
        traceback.print_exc()
        # Return actual error to client (Strict Mode)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
@rate_limit()
def predict_shelf_life():
    """
    Predict shelf life for given food and conditions.
    
    Expects: JSON with food_type, temperature, humidity, storage_type
    Returns: JSON with shelf life prediction
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    food_type = data.get('food_type', 'unknown')
    temperature = float(data.get('temperature', 4))
    humidity = float(data.get('humidity', 60))
    storage_type = data.get('storage_type', 'refrigerated')
    
    try:
        # Compute shelf life
        shelf_life = compute_shelf_life_for_food(
            food_type, temperature, humidity, storage_type
        )
        
        # Determine status
        if shelf_life <= 1:
            status = 'danger'
            status_text = 'Use Immediately!'
        elif shelf_life <= 3:
            status = 'warning'
            status_text = 'Use Soon'
        elif shelf_life <= 7:
            status = 'good'
            status_text = 'Fresh'
        else:
            status = 'excellent'
            status_text = 'Very Fresh'
        
        return jsonify({
            'success': True,
            'shelf_life_days': shelf_life,
            'status': status,
            'status_text': status_text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/foods', methods=['GET'])
def get_food_list():
    """Return list of known food types."""
    foods = [f.replace('_', ' ').title() for f in FOOD101_CLASSES]
    return jsonify({'foods': sorted(foods)})


def preload_models():
    """Pre-load ML models at startup to avoid request-time loading issues."""
    global _classifier
    
    print("Initializing Hugging Face API classifier...")
    try:
        from models.classifier import FoodClassifier
        _classifier = FoodClassifier()
        print("Classifier loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load classifier: {e}")
        print("Image classification will use fallback.")
        _classifier = None


# Pre-load models when imported by Gunicorn
preload_models()

if __name__ == '__main__':
    # Ensure template and static directories exist
    (PROJECT_ROOT / 'templates').mkdir(exist_ok=True)
    (PROJECT_ROOT / 'static').mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Food Shelf Life Predictor - Flask Server")
    print("=" * 60)
    print(f"Environment: {'PRODUCTION' if IS_PRODUCTION else 'DEVELOPMENT'}")
    print(f"Rate Limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds")
    
    # Note: Models are already preloaded at import time
    
    # Get port from environment (Railway sets PORT)
    port = int(os.getenv('PORT', 5000))
    debug = not IS_PRODUCTION
    
    print(f"\nStarting server at http://localhost:{port}")
    print(f"Debug mode: {debug}")
    print("Press Ctrl+C to stop\n")
    
    # NEVER use debug=True in production!
    app.run(debug=debug, host='0.0.0.0', port=port, threaded=True)
