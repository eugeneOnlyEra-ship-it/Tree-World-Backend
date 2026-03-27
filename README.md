# Tree World — FastAPI Backend

Soil classification + climate-focused tree recommendations.  
EfficientNet-B0 (fine-tuned) → soil type → Groq AI → tree recommendations.

## Project structure

```
tree-world/
├── main.py                        # FastAPI app entrypoint
├── requirements.txt
├── .env.example                   # copy to .env and fill in keys
├── app/
│   ├── config.py                  # pydantic-settings config
│   ├── dependencies.py            # shared DI: classifier + anthropic client
│   ├── routers/
│   │   └── analyze.py             # POST /api/analyze
│   ├── schemas/
│   │   └── analyze.py             # request / response models
│   └── services/
│       ├── classifier.py          # EfficientNet-B0 inference wrapper
│       └── recommender.py         # Claude AI recommendation service
├── ml/
│   ├── train.py                   # fine-tuning script
│   ├── checkpoints/               # saved model weights go here
│   └── dataset/                   # training images (see below)
└── tests/
    └── test_analyze.py
```

## Quick start

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# 3. Run the API (no model checkpoint needed to test Claude integration)
python main.py
# API docs at http://localhost:8000/docs
```

## Fine-tuning EfficientNet-B0

### 1. Prepare your dataset

Organise images into this folder structure:

```
ml/dataset/
  train/
    clay/       (min 100 images each)
    laterite/
    loam/
    peat/
    sandy/
    silt/
  val/
    clay/       (min 20 images each)
    laterite/
    loam/
    peat/
    sandy/
    silt/
```

**Recommended public datasets:**
- [Mendeley Soil Image Dataset](https://data.mendeley.com/datasets/nb3qyjrpgw/1)
- [Kaggle Soil Classification](https://www.kaggle.com/datasets/prasanshasatpathy/soil-types)
- Collect your own with a phone camera — close-up shots work best

**Tips for good training data:**
- Vary lighting (sunny, overcast, shade)
- Include wet and dry samples of each type
- Aim for 200–500 images per class minimum
- Include images from your target region (e.g. Sub-Saharan Africa)

### 2. Run training

```bash
python ml/train.py
```

Training uses a 2-phase strategy:
1. **Epochs 1–5**: Only the classifier head is trained (backbone frozen)
2. **Epochs 6+**: Full model is fine-tuned with a lower learning rate

Best checkpoint saved to `ml/checkpoints/efficientnet_b0_soil.pth`.

### 3. The API auto-loads the checkpoint on startup

No extra steps — `MODEL_CHECKPOINT_PATH` in `.env` points to it.

## API endpoints

### `POST /api/analyze`

Multipart form request:

| Field | Type | Required |
|-------|------|----------|
| `image` | File (JPEG/PNG/WEBP) | ✓ |
| `region` | string | ✓ |
| `climate_zone` | string | optional |

Example with curl:
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "image=@soil_sample.jpg" \
  -F "region=Malawi, Central Africa" \
  -F "climate_zone=tropical"
```

### `GET /api/health`
Returns `{"status": "ok"}`.

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

## Connecting to React Vite frontend

The frontend should send a `multipart/form-data` POST to `/api/analyze`.
CORS is configured via `ALLOWED_ORIGINS` in `.env` — defaults to `http://localhost:5173`.
