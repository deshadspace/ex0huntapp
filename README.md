
-----

# 🌌 ex0huntapp: Exoplanet Candidate Classification (Worlds Away Challenge)

This repository contains the full stack application developed for the **NASA Space Apps Challenge 2025** under the **"Worlds Away"** challenge.

`ex0huntapp` is a web-based tool designed to assist scientists and researchers in quickly classifying exoplanet candidates into one of three categories: **CONFIRMED PLANET**, **CANDIDATE**, or **FALSE POSITIVE**. It features a robust Machine Learning (ML) pipeline that utilizes stacked ensemble methods for high-accuracy predictions, built upon foundational planetary and stellar flux data.

## ✨ Features

  * **Custom CSV Upload & Mapping:** Users can upload any CSV dataset containing exoplanetary data and easily map their column headers to the model's required input features.
  * **Stacked Ensemble Prediction:** Utilizes high-performance ML models (XGBoost, Random Forest, CatBoost) in a **stacking architecture** to achieve a highly accurate consensus prediction.
  * **Model Stacking Options:** Provides two distinct meta-model options:
      * **Stack 1:** XGBoost + Random Forest (RFR) predictions fed into a Multi-Layer Perceptron (MLP).
      * **Stack 2:** XGBoost + CatBoost (CATB) predictions fed into an MLP.
  * **Detailed Output:** The application returns a ZIP file containing:
    1.  Individual CSV prediction files for each base model (XGBoost, RFR, CatBoost).
    2.  A final CSV with the stacked model's predicted class and certainty score for each input row, along with the original input data and identifiers.
  * **Transparent Metrics:** Displays the performance metrics (Accuracy, Classification Report) for the selected stack, ensuring transparency in the prediction quality.

-----

## ⚙️ Technical Architecture

### 1\. Backend (Python/Flask)

The backend handles file processing, feature engineering, model inference, and serves the prediction API.

  * **Framework:** **Flask** (`backend/app.py`) with **Flask-CORS** enabled for frontend integration.
  * **Data Processing:** Uses **Pandas** and **NumPy** for CSV handling and **Scikit-learn's `StandardScaler`** for normalization.
  * **Feature Engineering:** Dynamically generates complex polynomial and ratio features (e.g., `prad_div_period`, `insol_x_prad_err1`) from the 15 base features using `itertools.combinations`.
  * **Models:**
      * **Base Models (Level 1):** XGBoost, Random Forest, CatBoost (pre-trained, loaded via `joblib`).
      * **Meta Model (Level 2):** A custom **Multi-Layer Perceptron (MLP)** implemented in **PyTorch** (`torch.nn`). The MLP is trained on the class probability outputs of the Level 1 models.
  * **API Endpoint:** `/predict` (POST) handles the entire pipeline from file upload to ZIP output.

### 2\. Frontend (HTML/CSS/JavaScript)

The user interface provides an easy-to-use portal for prediction.

  * **Structure:** Standard HTML5 (`frontend/index.html`).
  * **Styling:** Custom CSS (`frontend/style.css`) with a space-themed aesthetic (using the "Press Start 2P" font).
  * **Logic:** Pure JavaScript handles file reading, dynamic column mapping generation, asynchronous API calls to the Flask backend, and file download management.

-----

## 💻 Setup and Installation

### Prerequisites

You must have Python 3.8+ and `pip` installed.

### 1\. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd ex0huntapp
```

### 2\. Install Dependencies

Install all required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3\. Run the Backend Server

Start the Flask application. It will run on `http://127.0.0.1:5000/`.

```bash
python backend/app.py
```

### 4\. Access the Frontend

Open the `frontend/index.html` file directly in your web browser.

-----

## 📚 Model Features

The application requires **15 core base features** (and their associated error margins) to run the prediction pipeline. These are:

| Technical Name | Descriptive Name (Displayed in App) | Definition (Tooltip) |
| :--- | :--- | :--- |
| `insol` / `insol_err*` | Insolation Flux (Earth Flux) | Stellar insolation received by the planet, scaled to Earth. |
| `period` / `period_err*` | Orbital Period (days) | The time required to complete one orbit. |
| `prad` / `prad_err*` | Planetary Radius (Earth Radii) | Radius of the planet relative to Earth. |
| `steff` / `steff_err*` | Stellar Temperature (K) | Effective temperature of the host star. |
| `srad` / `srad_err*` | Stellar Radius (Solar Radii) | Radius of the host star relative to the Sun. |

-----

## 📄 License & Credits

This project is licensed under the MIT License - see the `LICENSE` file for details.

Developed for the **NASA Space Apps Challenge 2025**.
