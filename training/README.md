# Training the RetinaTriage Model

Train a ResNet50-based diabetic retinopathy classifier on the
[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
dataset (5 classes: No DR → Proliferative DR).

---

## Prerequisites

- **Python 3.8+**
- **NVIDIA GPU recommended** — training on CPU is very slow
- **Kaggle account** — required to download the competition dataset

---

## Step 1: Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

---

## Step 2: Get the APTOS 2019 Dataset from Kaggle

### 2a. Create a Kaggle API token

1. Go to [kaggle.com](https://www.kaggle.com/) → click your profile icon → **Settings**
2. Scroll to the **API** section → click **Create New API Token**
3. This downloads a `kaggle.json` file — keep it safe

### 2b. Accept the competition rules

> **Important:** You **must** visit
> <https://www.kaggle.com/competitions/aptos2019-blindness-detection>
> and click **Join Competition** / accept the rules **before** downloading.
> Otherwise the download will fail with a 403 error.

### 2c. Download & extract

```bash
pip install kaggle

# Linux / macOS
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
kaggle competitions download -c aptos2019-blindness-detection

# Extract into data/
unzip aptos2019-blindness-detection.zip -d data/
```

After extraction your directory should look like:

```
training/
  data/
    train.csv            # columns: id_code, diagnosis (0-4)
    train_images/        # {id_code}.png files
    test.csv
    test_images/
  train.py
  requirements.txt
```

---

## Step 3: Train the Model

### Full training run

```bash
python train.py --data_dir data/ --epochs 15 --batch_size 32 --lr 1e-4 --out ../backend/weights/model.pth
```

### Smoke test (verify the pipeline works end-to-end)

```bash
python train.py --data_dir data/ --smoke_test --out ../backend/weights/model.pth
```

This uses only 100 images for 1 epoch — takes under a minute on any machine.

### All CLI arguments

| Argument       | Default              | Description                                  |
| -------------- | -------------------- | -------------------------------------------- |
| `--data_dir`   | *(required)*         | Path to the `data/` directory                |
| `--epochs`     | `15`                 | Number of training epochs                    |
| `--batch_size` | `32`                 | Batch size                                   |
| `--lr`         | `1e-4`               | Learning rate (Adam)                         |
| `--out`        | `weights/model.pth`  | Where to save the best model weights         |
| `--smoke_test` | off                  | Quick sanity run: 100 images, 1 epoch        |

---

## Step 4: Google Colab (Free GPU)

If you don't have a local GPU, use Google Colab's free T4.
Open a new Colab notebook and paste:

```python
# Step 1: Upload your kaggle.json
from google.colab import files
uploaded = files.upload()  # upload kaggle.json

# Step 2: Setup Kaggle
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Step 3: Download dataset
!pip install kaggle
!kaggle competitions download -c aptos2019-blindness-detection
!unzip -q aptos2019-blindness-detection.zip -d data/

# Step 4: Clone/upload training code
# Upload train.py and requirements.txt, or clone your repo
!pip install -r requirements.txt

# Step 5: Train
!python train.py --data_dir data/ --epochs 15 --batch_size 32 --lr 1e-4 --out model.pth

# Step 6: Download the trained model
files.download('model.pth')
```

---

## Step 5: Deploy the Model

Copy the trained `model.pth` into the backend so the web app can use it:

```bash
cp model.pth ../backend/weights/model.pth
```

The backend's `model.py` loads it with:

```python
model.load_state_dict(torch.load("weights/model.pth"))
```

---

## Expected Results

| Metric                          | Expected Range   |
| ------------------------------- | ---------------- |
| Validation quadratic weighted κ | **0.75 – 0.85**  |
| Validation accuracy             | **70 – 80 %**    |
| Training time (Colab T4)        | ~30 – 45 minutes |

> **Note:** Results vary with random seed and data shuffling.
> The quadratic weighted kappa (QWK) is the primary metric — it accounts
> for the ordinal nature of DR severity grades and is what the original
> Kaggle competition used for scoring.
