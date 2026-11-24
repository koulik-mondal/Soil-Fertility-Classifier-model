# Soil Fertility Classification Using Machine Learning

**Project by:** Koulik Mondal (BWU/BCA/24/374)  
**Programme:** BCA | **Course:** Project Based Learning (PBL)

## 1. Project objective
Build and compare multiclass machine-learning classifiers that predict one of three dataset teaching labels: **Low Fertility**, **Medium Fertility**, or **High Fertility**.

## 2. Important dataset provenance and scientific limitation
This repository uses `data/soil_fertility_classification_1000.csv` (1,000 rows). The dataset documentation (`data/DATASET_README.txt`) states:

- The 1,000 rows are **source-calibrated synthetic teaching data** prepared for BCA PBL; they are not 1,000 copied or authentic laboratory observations.
- The source basis includes AgriNet (Mendeley Data, DOI `10.17632/c7tr9j78td.1`), the Krishnagiri raw soil dataset (Mendeley Data, DOI `10.17632/7bk36mh3tc.1`), and a Dharmapuri soil data article. These sources contain laboratory-tested soil records, but the project CSV is a synthetic teaching dataset calibrated from that basis.
- `Fertility_Class` is a **derived teaching label**. Its score combines N, P, K, organic carbon, pH and EC. Thresholds: Low < 40; Medium 40-69.99; High >= 70.
- Model performance therefore measures how well algorithms learn these derived labels. It must not be presented as proven real-world soil-diagnosis accuracy.

## 3. Dataset attributes
- Identification/provenance: `Sample_ID`, `Source_Calibration`
- Categorical/context: `State`, `District`, `Soil_Type`, `Crop`
- Numerical soil parameters: `pH`, `Electrical_Conductivity_dS_m`, `Organic_Carbon_g_kg`, `Available_N_kg_ha`, `Available_P_kg_ha`, `Available_K_kg_ha`, `Sulphur_mg_kg`, `Zinc_mg_kg`, `Boron_mg_kg`, `Iron_mg_kg`, `Manganese_mg_kg`, `Copper_mg_kg`
- Teaching target construction field: `Fertility_Score`
- Target: `Fertility_Class`

`Fertility_Score` is deliberately excluded from model features because it is used to derive the target class. Including it would cause direct target leakage.

## 4. Workflow
1. Load and inspect dataset, data types, shape and target distribution.
2. Remove exact duplicate rows (if any), audit missing values, verify score-to-label thresholds.
3. Impute numeric missing values with median and categorical with most frequent inside the training pipeline.
4. EDA: class counts, feature distributions, state / soil-type / crop counts, score by class.
5. Correlation analysis among numerical soil properties.
6. Stratified 80/20 train-test split (random state 42).
7. Compare Decision Tree, Random Forest, KNN and SVM (plus 5-fold CV on the training set).
8. Evaluate accuracy, macro precision, macro recall, macro F1, confusion matrices and one-vs-rest ROC curves.
9. Select the model with the highest hold-out macro F1 (accuracy as tie-breaker).
10. Permutation importance on held-out data.
11. Save the trained pipeline and provide a Streamlit prediction interface.

## 5. Project structure
```text
Soil_Fertility_Classification_Koulik_Mondal/
├── app.py                      # Streamlit prediction app
├── requirements.txt
├── README.md
├── PROJECT_OVERVIEW.txt
├── data/
│   ├── soil_fertility_classification_1000.csv
│   └── DATASET_README.txt
├── src/
│   ├── __init__.py
│   └── train.py                # full ML pipeline
├── models/
│   └── best_soil_fertility_model.pkl
├── outputs/
│   ├── model_comparison_results.csv
│   ├── feature_importance_results.csv
│   ├── figures/                # EDA and evaluation charts
│   └── reports/                # metrics, audit and predictions
└── docs/
    ├── PROJECT_REPORT.md
    └── DATA_DICTIONARY.md
```

## 6. Run in VS Code
Open the project folder in VS Code (File > Open Folder), open a terminal (Ctrl+`), then:

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

Open the local Streamlit URL printed in the terminal (usually `http://localhost:8501`).

`python -m src.train` is optional on a fresh copy because trained outputs are already included, and the app retrains automatically if the saved model cannot be loaded. Run it any time you want to regenerate the figures, metrics and model.

## 7. Generated outputs
- `models/best_soil_fertility_model.pkl`
- `outputs/model_comparison_results.csv`, `outputs/feature_importance_results.csv`
- `outputs/figures/` - class distribution, feature distributions, correlation heatmap, model comparison, confusion matrices, ROC and feature importance
- `outputs/reports/` - classification report, test predictions, run summary and data audit metadata

## 8. Reproducibility
The split and estimators use random seed 42. All preprocessing is fitted inside each model pipeline on training data only.

## 9. Responsible interpretation
This is an academic ML demonstration. A model prediction should not be used to decide fertilizer application or crop treatment without current soil laboratory testing and agronomic advice.
