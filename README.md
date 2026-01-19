# Food Freshness Predictor

> AI-powered food classification and storage life prediction using computer vision.

Upload a photo of any food item, and the system will identify it and tell you how long it will stay fresh based on your storage conditions.

## Features

- **Hybrid AI Classification**: Uses both Food-101 and ImageNet models for maximum accuracy
- **15 Food Categories**: Apple, Banana, Bread, Milk, Pasta, Pizza, Burger, Sushi, Meat, Fish, Egg, Vegetable, Fruit, Rice, Cake
- **Shelf Life Prediction**: RandomForest model predicts remaining days based on temperature, humidity, and storage type
- **Science-Based**: Calculations based on USDA FoodKeeper data and Q10 temperature coefficients
- **Docker Ready**: One-command deployment with `docker-compose up`

## Quick Start

### Option 1: Docker (Recommended)

1. **Clone the repo** and create `.env`:
   ```bash
   cp .env.example .env
   # Edit .env and add your Hugging Face token
   ```

2. **Run with Docker**:
   ```bash
   docker-compose up --build
   ```

3. **Open** [http://localhost:5000](http://localhost:5000)

### Option 2: Local Python

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variable**:
   ```bash
   # Windows PowerShell
   $env:HUGGINGFACE_API_TOKEN = "hf_your_token_here"
   
   # Linux/Mac
   export HUGGINGFACE_API_TOKEN="hf_your_token_here"
   ```

3. **Run the Flask app**:
   ```bash
   python app_flask.py
   ```

4. **Open** [http://localhost:5000](http://localhost:5000)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HUGGINGFACE_API_TOKEN` | Yes | Get free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

## Project Structure

```
food-freshness-predictor/
├── app_flask.py           # Flask web server
├── config.py              # All configuration constants
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container definition
├── docker-compose.yml     # Docker orchestration
├── .env.example           # Environment template
├── .gitignore             # Git exclusions
├── models/
│   ├── classifier.py      # Hybrid HuggingFace API classifier
│   └── shelf_life_predictor.py  # RandomForest regressor
├── utils/
│   └── data_loader.py     # Dataset generation
├── templates/
│   └── index.html         # Web UI
└── data/                  # Generated training data (gitignored)
```

## How It Works

### Food Classification (Hybrid Approach)
1. **Primary**: Queries `Kaludi/food-category-classification-v2.0` (Food-101 model) - Best for prepared dishes
2. **Fallback**: If confidence is low, queries `google/efficientnet-b0` (ImageNet) - Best for raw ingredients
3. **Result**: Returns the highest-confidence match from either model

### Shelf Life Prediction
- **Model**: RandomForestRegressor (scikit-learn)
- **Features**: Food category, temperature, humidity, storage type
- **Science**: Uses Q10 temperature coefficient for accurate decay modeling

## Supported Foods

| Category | Example Items | Refrigerated Shelf Life |
|----------|---------------|-------------------------|
| Apple | Whole apples, apple pie | 21 days |
| Banana | Fresh bananas | 5 days |
| Bread | Loaves, toast, bagels | 7 days |
| Milk | Dairy products | 7 days |
| Pasta | Spaghetti, lasagna, ramen | 4 days |
| Pizza | Leftover pizza | 4 days |
| Burger | Hamburgers, hot dogs | 3 days |
| Sushi | Sushi, sashimi | 1 day |
| Meat | Steak, pork, chicken | 4 days |
| Fish | Salmon, tuna, seafood | 2 days |
| Egg | Eggs, omelettes | 28 days |
| Vegetable | Salads, fresh veggies | 7 days |
| Fruit | Berries, cut fruit | 5 days |
| Rice | Cooked rice, risotto | 5 days |
| Cake | Cakes, desserts | 5 days |

## Tech Stack

- **Backend**: Python, Flask
- **AI**: Hugging Face Inference API (no local GPU required)
- **ML**: scikit-learn (RandomForest)
- **Frontend**: HTML/CSS/JavaScript
- **Deployment**: Docker, docker-compose

## License

MIT License - Free for educational and commercial use.

## References

- Food-101: Bossard et al., "Food-101 – Mining Discriminative Components", ECCV 2014
- Q10 Temperature Coefficient: Labuza, T.P. (1982). "Shelf Life Dating of Foods"
- USDA FoodKeeper Application Data
