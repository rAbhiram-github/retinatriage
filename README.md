# RetinaTriage 👁️

**AI-Powered Diabetic Retinopathy Screening with Grad-CAM Explainability**

> ⚠️ **Disclaimer:** This is a research/demo tool only — **not a medical device, not for clinical or diagnostic use.**

RetinaTriage is a full-stack web application that classifies fundus (retinal) images into one of five diabetic retinopathy (DR) stages and provides Grad-CAM heatmap overlays showing which regions of the image most influenced the prediction.

---

## DR Stages (APTOS / Standard Classification)

| Stage | Label              | Description                          |
|-------|--------------------|--------------------------------------|
| 0     | No DR              | No signs of diabetic retinopathy     |
| 1     | Mild               | Mild nonproliferative DR             |
| 2     | Moderate           | Moderate nonproliferative DR         |
| 3     | Severe             | Severe nonproliferative DR           |
| 4     | Proliferative DR   | Proliferative diabetic retinopathy   |

---

## Architecture

```
retinatriage/
├── backend/              # FastAPI + PyTorch inference server
│   ├── main.py           # API endpoints (/predict, /health)
│   ├── model.py          # ResNet50 model + preprocessing
│   ├── gradcam.py        # Grad-CAM heatmap generation
│   ├── requirements.txt
│   └── weights/          # Place model.pth here
├── training/             # Model training pipeline
│   ├── train.py          # ResNet50 training on APTOS 2019
│   ├── requirements.txt
│   └── README.md         # Dataset download + training guide
├── frontend/             # Vite + React UI
│   └── ...
└── README.md             # This file
```

---

## Order of Operations

### Option A: Run immediately in untrained/test mode (no weights needed)

The app runs **without trained weights** — it will use ImageNet-pretrained ResNet50 as a fallback and display a clear warning that predictions are meaningless. This is useful for verifying the pipeline end-to-end.

### Option B: Train first, then run

1. **Train the model** following [`training/README.md`](training/README.md) to produce `model.pth`
2. **Place the weights** at `backend/weights/model.pth`
3. **Start the backend** (see below)
4. **Start the frontend** (see below)

---

## Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

- `GET /health` — health check
- `POST /predict` — upload a fundus image, get prediction + Grad-CAM heatmap

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`. Upload a retinal image and click "Analyze".

---

## Training the Model

See [`training/README.md`](training/README.md) for full instructions including:

- Downloading the APTOS 2019 Blindness Detection dataset from Kaggle
- Running training locally or on Google Colab (free GPU)
- Smoke test to verify the pipeline

**Quick smoke test** (verifies pipeline in ~30 seconds, no GPU required):

```bash
cd training
pip install -r requirements.txt
python train.py --data_dir data/ --smoke_test --out ../backend/weights/model.pth
```

**Full training:**

```bash
python train.py --data_dir data/ --epochs 15 --batch_size 32 --lr 1e-4 --out ../backend/weights/model.pth
```

---

## Tech Stack

| Layer    | Technology                              |
|----------|-----------------------------------------|
| Frontend | React (Vite), vanilla CSS               |
| Backend  | FastAPI, Uvicorn                        |
| ML       | PyTorch, torchvision (ResNet50)         |
| XAI      | pytorch-grad-cam (Grad-CAM)            |
| Training | APTOS 2019 dataset, sklearn metrics    |

---

## API Response Format

```json
{
  "stage": 2,
  "stage_label": "Moderate",
  "confidence": 0.87,
  "probabilities": {
    "No DR": 0.02,
    "Mild": 0.05,
    "Moderate": 0.87,
    "Severe": 0.04,
    "Proliferative DR": 0.02
  },
  "heatmap": "<base64 encoded PNG>",
  "untrained_warning": false
}
```

---

## License
MIT © Abhiram Radhakrishnan

This project is for educational and research purposes only.
