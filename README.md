# Real-Time Detection of Diabetic Eye Disorders Using Deep Learning

A Flask-based academic project that accepts an uploaded eye image, applies basic image-processing operations, and uses a TensorFlow/Keras CNN to produce a predicted class from eight eye-condition labels.

> **Important medical disclaimer:** This project is for educational and research demonstration only. The current model has shown approximately **45% validation accuracy** on the supplied dataset and is not clinically validated. Its predictions must not be used to diagnose, treat, or make medical decisions. Consult a qualified eye-care professional for medical concerns.

## Features

- User registration and login using a SQLite database.
- Upload eye images in PNG, JPG, JPEG, or BMP format.
- Basic image processing outputs:
  - Grayscale
  - Canny edge detection
  - Binary thresholding
  - Sharpened image
- CNN inference using a saved Keras model.
- Displays a predicted class and general next-step guidance.
- Flask backend with the existing HTML templates and static assets.

## Technology Stack

- Python
- Flask
- TensorFlow / Keras
- NumPy
- OpenCV
- scikit-learn
- SQLite
- HTML, CSS, and JavaScript (project UI)

## Supported Prediction Labels

The model currently uses this output order:

1. Cataract
2. Glaucoma
3. Mild diabetic retinopathy
4. Moderate diabetic retinopathy
5. Normal
6. Proliferative diabetic retinopathy
7. Diabetic eye disease
8. Severe diabetic retinopathy

The model chooses the class with the highest softmax output. That output is a model score, **not a clinically validated probability or diagnostic confidence**.

## Project Structure

The key files and folders are expected to look like this:

```text
eye_disease_original_ui_project/
├── app.py                    # Flask application and image prediction route
├── train_model.py            # Trains and saves the CNN
├── train_data.npy            # Supplied training dataset (needed for training)
├── requirements.txt          # Python dependencies
├── models/
│   └── eye_disease.keras     # Generated trained model (created by training)
├── templates/                # HTML templates for the existing UI
├── static/
│   ├── uploads/               # Uploaded images (created/used at runtime)
│   └── processed/             # Processed images (created/used at runtime)
└── user_data.db               # SQLite user database (created at runtime)
```

Your local project may contain additional template, CSS, JavaScript, and static files. Keep those files when pushing the complete application.

## Requirements

- Python 3.10–3.12 is recommended, depending on TensorFlow compatibility for your operating system.
- pip
- Git (for version control)
- The supplied `train_data.npy` file if you need to train the model from scratch.

## Setup on Windows (PowerShell)

### 1. Clone the repository

Replace `YOUR_USERNAME/YOUR_REPOSITORY` with your GitHub username and repository name:

```powershell
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

If you are setting up the project from a local folder rather than cloning, open PowerShell in the project directory and skip this step.

### 2. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use this command for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the model (if the model file is not included)

Ensure `train_data.npy` is in the project root, then run:

```powershell
python train_model.py
```

The script saves the model to:

```text
models/eye_disease.keras
```

Training can take time and uses the supplied dataset. The reported validation score is for this dataset split only and does not establish clinical performance.

### 5. Run the application

```powershell
python app.py
```

Open this address in your browser:

```text
http://127.0.0.1:5000
```

Register an account, sign in, and upload a supported image.

## Training Details

`train_model.py` currently:

- Loads `train_data.npy` with NumPy.
- Uses records containing 50 × 50 RGB-channel-shaped arrays and eight-element label vectors.
- Scales image values by dividing by 255.
- Uses an 80/20 stratified train/validation split with `random_state=42`.
- Trains a CNN with three convolution/pooling blocks, a dense layer, dropout, and an eight-class softmax output.
- Uses Adam, sparse categorical cross-entropy, accuracy, and early stopping.
- Saves the resulting model as `models/eye_disease.keras`.

The supplied dataset contains 2,601 records and has imbalanced class counts. The current validation accuracy observed during development was around 45%. Improving and independently validating the model is required before any real-world use.

## Image Upload and Processing

The Flask `/image` route accepts an uploaded image, saves it under `static/uploads/`, and creates processed versions under `static/processed/`. The model resizes the image to 50 × 50 and scales pixel values by 255 before prediction.

Do not upload real patient images, personal information, credentials, or private medical data to a public repository.

## GitHub: Push This Project

If you already created an empty repository on GitHub, open PowerShell in the project folder and run:

```powershell
git init
git add .
git status
git commit -m "Add diabetic eye disorder detection project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your actual repository URL. If `origin` is already configured, check it with `git remote -v` and do not add it a second time. If the repository already contains commits, follow Git's instructions and reconcile the histories rather than force-pushing blindly.

### Before pushing

- Review `git status` and inspect the files included in the commit.
- Do not commit `.venv/`, `__pycache__/`, temporary check scripts, generated uploads, processed user images, or the local SQLite database.
- Check the dataset's license and size before publishing `train_data.npy`. If redistribution is not permitted or the file is too large, leave it out and document how authorized users can obtain it.
- Decide whether to publish `models/eye_disease.keras`. It may be large; if omitted, users must train it using the dataset.
- Never commit passwords, API keys, private user data, or production secrets.
- The current Flask secret key is a placeholder and must be replaced with a secure environment-based secret before deployment.
- Do not deploy with Flask debug mode enabled.

A suitable `.gitignore` should include at least:

```gitignore
.venv/
__pycache__/
*.py[cod]
.env
instance/
user_data.db
static/uploads/*
static/processed/*
models/
check_model.py
check_data.py
check_image_*.jpg
```

If you intend to version the trained model, remove `models/` from `.gitignore` and confirm the file size and repository policies first. If you intend to distribute the dataset, do not ignore it; first verify its license and size. Keep required empty folders using `.gitkeep` files if needed.

## Troubleshooting

### The model file is missing

Run:

```powershell
python train_model.py
```

Confirm that `models/eye_disease.keras` is created.

### Dependency or TensorFlow installation errors

Use a supported Python version and install dependencies inside the activated virtual environment:

```powershell
python --version
python -m pip install -r requirements.txt
```

### Predictions are inaccurate or repeatedly wrong

The current model has limited validation performance and has not been clinically validated. Check dataset labels, class balance, preprocessing consistency, and model evaluation before changing the UI or relying on results.

## Limitations

- The model is an academic prototype, not a medical device.
- The dataset's origin, licensing, and clinical suitability must be independently verified.
- A validation split from the same supplied dataset is not independent clinical validation.
- The model may misclassify images, including images from classes it rarely or never predicts.
- The app's suggested next steps are general informational guidance, not a substitute for professional care.

## License

No license is specified here. Add a license only after confirming that you have the rights to distribute the application code, model, and dataset. The dataset and model may have different licensing terms.
