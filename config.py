"""
Configuration Module for Food Shelf Life Predictor
===================================================

This module contains all configuration constants and mappings used throughout
the application. Centralizing these values ensures consistency and makes
the system easily adjustable for different use cases.

Academic Reference:
- Food-101 Dataset: Bossard et al., "Food-101 – Mining Discriminative Components with Random Forests", ECCV 2014
- Shelf life modeling: Based on USDA FoodKeeper data and Q10 temperature coefficients
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directory for datasets
DATA_DIR = PROJECT_ROOT / "data"

# Models directory for saved models
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# =============================================================================
# TARGET FOOD CATEGORIES
# =============================================================================

# The four food categories we support for shelf life prediction
# EXPANDED LIST based on User Request
TARGET_CATEGORIES = [
    "apple", "banana", "bread", "milk", "pasta", 
    "pizza", "burger", "sushi", "meat", "fish", "egg", 
    "vegetable", "fruit", "rice", "cake"
]

# Mapping from our categories to numerical indices for model training
CATEGORY_TO_INDEX = {cat: idx for idx, cat in enumerate(TARGET_CATEGORIES)}
INDEX_TO_CATEGORY = {idx: cat for cat, idx in CATEGORY_TO_INDEX.items()}

# Food-101 contains 101 food categories. We map relevant ones to our expanded targets.
FOOD101_TO_CATEGORY = {
    # Original categories
    "apple_pie": "apple", "banana": "banana",
    "bread_pudding": "bread", "french_toast": "bread", "garlic_bread": "bread",
    "grilled_cheese_sandwich": "bread", "club_sandwich": "bread", 
    "cheese_plate": "milk", "cheesecake": "milk", "ice_cream": "milk", "frozen_yogurt": "milk",
    "spaghetti_bolognese": "pasta", "spaghetti_carbonara": "pasta", "macaroni_and_cheese": "pasta",
    "lasagna": "pasta", "ravioli": "pasta", "gnocchi": "pasta", "pad_thai": "pasta", "ramen": "pasta", "pho": "pasta",

    # NEW MAPPINGS
    "pizza": "pizza",
    "hamburger": "burger", "hot_dog": "burger", # Hotdogs act like processed meat/bread similar to burgers
    "sushi": "sushi", "sashimi": "sushi", "takoyaki": "sushi",
    "steak": "meat", "pork_chop": "meat", "beef_carpaccio": "meat", "baby_back_ribs": "meat", "filet_mignon": "meat", "prime_rib": "meat",
    "fish_and_chips": "fish", "ceviche": "fish", "tuna_tartare": "fish", "grilled_salmon": "fish", "clam_chowder": "fish", "lobster_bisque": "fish", "oysters": "fish",
    "deviled_eggs": "egg", "eggs_benedict": "egg", "omelette": "egg",
    "fried_rice": "rice", "risotto": "rice",
    "carrot_cake": "cake", "chocolate_cake": "cake", "cup_cakes": "cake", "red_velvet_cake": "cake", "strawberry_shortcake": "cake", "tiramisu": "cake",
    "caesar_salad": "vegetable", "greek_salad": "vegetable", "caprese_salad": "vegetable", "beet_salad": "vegetable", "falafel": "vegetable", "edamame": "vegetable", "guacamole": "vegetable",
    "fruit_salad": "fruit",
}

# Base shelf life in days under refrigerated conditions (3-5°C)
BASE_SHELF_LIFE_REFRIGERATED = {
    "apple": 21,      # 3 weeks
    "banana": 5,      # 5-7 days
    "bread": 7,       # 1 week
    "milk": 7,        # 1 week
    "pasta": 4,       # Cooked pasta: 3-5 days
    "pizza": 4,       # Leftover pizza: 3-4 days
    "burger": 3,      # Cooked burger: 3-4 days
    "sushi": 1,       # Sushi: 24 hours max
    "meat": 4,        # Fresh meat: 3-5 days
    "fish": 2,        # Fresh fish: 1-2 days
    "egg": 28,        # Eggs: 3-5 weeks
    "vegetable": 7,   # General veggies: 1 week
    "fruit": 5,       # General cut fruit: 3-5 days
    "rice": 5,        # Cooked rice: 4-6 days
    "cake": 5,        # Cake with frosting: 3-7 days
}

# Frozen storage multiplier (how much longer food lasts when frozen)
FROZEN_MULTIPLIER = {
    "apple": 12,
    "banana": 6,
    "bread": 4,
    "milk": 3,
    "pasta": 8,       # 1-2 months
    "pizza": 8,       # 1-2 months
    "burger": 12,     # 3-4 months
    "sushi": 1,       # Freezing sushi often ruins texture (no extension)
    "meat": 24,       # 6-12 months
    "fish": 12,       # 2-6 months
    "egg": 12,        # Frozen out of shell
    "vegetable": 24,  # 8-12 months
    "fruit": 24,      # 8-12 months
    "rice": 12,       # 6 months
    "cake": 12,       # 4-6 months
}

# Default category if detection fails (low confidence)
DEFAULT_CATEGORY = "unknown"

# EfficientNet-B0 / ViT input specifications
IMAGE_SIZE = 224

# =============================================================================
# STORAGE TYPE CONFIGURATION
# =============================================================================

STORAGE_TYPES = ["refrigerated", "room_temperature", "frozen"]
STORAGE_TO_INDEX = {storage: idx for idx, storage in enumerate(STORAGE_TYPES)}
INDEX_TO_STORAGE = {idx: storage for storage, idx in STORAGE_TO_INDEX.items()}

STORAGE_TEMPERATURE_RANGES = {
    "refrigerated": (3, 5),
    "room_temperature": (20, 25),
    "frozen": (-18, -15),
}

# Q10 coefficient for temperature-shelf life relationship
Q10_COEFFICIENT = 2.5

# =============================================================================
# HUGGING FACE API CONFIGURATION
# =============================================================================

# Hugging Face Inference API URLs
# Primary: Food-101 (best for dishes like "Apple Pie", "Sushi")
# Secondary: ImageNet (best for raw ingredients like "Banana", "Milk Carton")
# Using Router API for both to avoid 410 deprecation errors
HUGGINGFACE_API_URL_FOOD101 = "https://router.huggingface.co/hf-inference/models/Kaludi/food-category-classification-v2.0"
HUGGINGFACE_API_URL_IMAGENET = "https://router.huggingface.co/hf-inference/models/google/efficientnet-b0"

# API Token: Read from environment variable (DO NOT HARDCODE!)
# Set via .env file or docker-compose environment
# NOTE: Using a function for lazy loading - ensures Railway env vars are read at runtime!
def get_huggingface_token():
    """Get Hugging Face API token from environment (lazy load for Railway compatibility)."""
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    # Debug: show what we found
    if token:
        print(f"DEBUG CONFIG: Token found, starts with: {token[:10]}...")
    else:
        print("DEBUG CONFIG: HUGGINGFACE_API_TOKEN not found in environment!")
        print(f"DEBUG CONFIG: Available env vars with 'HUG' or 'TOKEN': {[k for k in os.environ.keys() if 'HUG' in k.upper() or 'TOKEN' in k.upper()]}")
    return token

# For backward compatibility - but prefer get_huggingface_token() in new code
HUGGINGFACE_API_TOKEN = None  # Will be set dynamically

def get_huggingface_headers():
    """Get API headers with current token (lazy load for Railway compatibility)."""
    token = get_huggingface_token()
    headers = {"Content-Type": "application/octet-stream"}  # Router API requires this
    if token and token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"
    else:
        print("WARNING: HUGGINGFACE_API_TOKEN not set in environment!")
    return headers

# For backward compatibility - this is now a function call result
# But callers should use get_huggingface_headers() directly
HUGGINGFACE_HEADERS = {}

# Random Forest hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2

# Saved model filenames
RF_MODEL_FILENAME = "trained_rf_model.joblib"
SCALER_FILENAME = "feature_scaler.joblib"

# =============================================================================
# DATA GENERATION CONFIGURATION
# =============================================================================

# Number of samples to generate per food category
SAMPLES_PER_CATEGORY = 500

# Environmental parameter ranges for data generation
TEMPERATURE_RANGE = (0, 30)      # Celsius
HUMIDITY_RANGE = (30, 90)        # Percentage

# Random seed for reproducibility
DATA_RANDOM_SEED = 42

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Train/test split ratio
TEST_SIZE = 0.2

# Cross-validation folds
CV_FOLDS = 5

# =============================================================================
# STREAMLIT UI CONFIGURATION
# =============================================================================

# Temperature slider configuration
UI_TEMP_MIN = 0
UI_TEMP_MAX = 30
UI_TEMP_DEFAULT = 4

# Humidity slider configuration
UI_HUMIDITY_MIN = 30
UI_HUMIDITY_MAX = 90
UI_HUMIDITY_DEFAULT = 60

# Shelf life color thresholds (days)
SHELF_LIFE_WARNING_THRESHOLD = 3   # Yellow warning below this
SHELF_LIFE_DANGER_THRESHOLD = 1    # Red danger below this
