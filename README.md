# Breast Cancer Tumor Classification Project

A machine learning system for classifying breast tumors as benign or malignant using Support Vector Machine (SVM) with 97.4% accuracy.

## Project Overview

This project develops and deploys a machine learning model to assist in breast cancer diagnosis using the Wisconsin Breast Cancer Dataset. The model achieves:

- **97.4% Accuracy** on test data
- **99.6% ROC AUC** score
- **Real-time API** for predictions
- **17 optimized features** from original 30


## Prerequisites
- Python 3.8+
- Anaconda or Miniconda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusesamantha-juteram/tumor-classification.git
cd tumor-classification
```

2. **Create environment**
```bash
conda create -n tumor-env python=3.8
conda activate tumor-env
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Train the Model
```bash
python scripts/model_training_svm.py
```

#### 2. Start the API
```bash
python api.py
```

#### 3. Make Predictions
- **Web Interface**: Go to `http://localhost:8000/docs`
- **Test Script**: `python test_api.py`
- **Direct API Call**: Send POST to `http://localhost:8000/predict`

##  API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model information |
| `/features` | GET | Required features list |

### Example API Usage

**Input Features (17):**
```json
{
  "concave_points_mean": 0.05,
  "area_worst": 800.0,
  "fractal_dimension_worst": 0.08,
  "smoothness_worst": 0.15,
  "symmetry_worst": 0.25,
  "fractal_dimension_mean": 0.06,
  "smoothness_mean": 0.1,
  "compactness_se": 0.02,
  "fractal_dimension_se": 0.003,
  "concave_points_se": 0.01,
  "symmetry_se": 0.015,
  "perimeter_se": 3.0,
  "concavity_se": 0.02,
  "symmetry_mean": 0.18,
  "smoothness_se": 0.005,
  "texture_se": 1.5,
  "texture_mean": 15.0
}
```

**Response:**
```json
{
  "prediction": "Benign",
  "probability": 0.1234,
  "confidence": "High"
}
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 97.4% |
| F1-Score | 96.4% |
| ROC AUC | 99.6% |
| Precision | 97.6% |
| Recall | 95.2% |

### Model Comparison Results

| Model | CV Accuracy | Test Accuracy | ROC AUC |
|-------|-------------|---------------|---------|
| **SVM (Best)** | 97.5% ± 0.7% | 97.4% | 99.6% |
| Logistic Regression | 96.7% ± 0.9% | 95.6% | 99.1% |
| Random Forest | 95.3% ± 2.8% | 95.6% | 98.9% |
| Gradient Boosting | 94.7% ± 3.6% | 95.6% | 98.6% |

## 🧪 Running Tests

```bash
# Test the API
python test_api.py

# Compare all models
python scripts/model_comparison.py

# Run statistical analysis
python scripts/statistical_modeling.py
```

##  Features Used

The model uses 17 carefully selected features (reduced from original 30):

**Mean Values:**
- concave_points_mean, fractal_dimension_mean, smoothness_mean, symmetry_mean, texture_mean

**Worst Values:**
- area_worst, fractal_dimension_worst, smoothness_worst, symmetry_worst

**Standard Error Values:**
- compactness_se, fractal_dimension_se, concave_points_se, symmetry_se, perimeter_se, concavity_se, smoothness_se, texture_se

##  Key Findings

1. **SVM Superiority**: SVM with RBF kernel outperformed all other algorithms
2. **Feature Selection**: VIF-based selection reduced features by 43% without performance loss
3. **Robustness**: Cross-validation shows consistent performance across folds
4. **Clinical Relevance**: High sensitivity minimizes missed cancer cases

##  Development

### Adding New Models

1. Create model training script in `scripts/`
2. Follow the pattern in `model_training_svm.py`
3. Update `model_comparison.py` to include new model
4. Update API model loading logic

### Customizing the API

- Modify `api.py` for additional endpoints
- Update `PatientData` model for different input formats
- Add authentication/logging as needed

## Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
```

##  Disclaimer

This model is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult healthcare professionals for medical decisions.

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Wisconsin Breast Cancer Dataset contributors
- Scikit-learn development team
- FastAPI framework developers

##  Contact

Your Name - [smjuteram@hotmail.com](mailto:smjuteram@hotmail.com)
Project Link: [https://github.com/samantha-juteram/tumor-classification](https://github.com/samantha-juteram/tumor-classification)