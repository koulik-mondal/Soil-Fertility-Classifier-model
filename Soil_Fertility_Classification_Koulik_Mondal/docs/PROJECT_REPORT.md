# PROJECT REPORT
## Soil Fertility Classification Using Machine Learning

**Submitted by:** Koulik Mondal (BWU/BCA/24/374)  
**Programme:** Bachelor of Computer Applications (BCA)  
**Project category:** Project Based Learning

## Abstract
This project builds a reproducible multiclass machine-learning workflow that classifies soil records into Low, Medium and High Fertility. It covers dataset inspection, cleaning, missing-value handling, exploratory data analysis, correlation analysis, feature interpretation, training and comparison of four algorithms (Decision Tree, Random Forest, KNN, SVM), evaluation, and a Streamlit prediction interface.

**Dataset limitation:** The 1,000-row dataset is source-calibrated synthetic teaching data. Its fertility labels are derived from a teaching score, not independently measured laboratory labels. Results describe performance on this educational dataset and do not establish field diagnostic accuracy.

## 1. Introduction
Soil properties such as pH, electrical conductivity, organic carbon and available nutrients are central to soil assessment. This project applies supervised classification to predict a fertility class from these properties plus location and crop context.

## 2. Objectives
- Understand the dataset attributes and provenance.
- Inspect data quality, duplicates and missing values.
- Perform EDA and visualisations; analyse correlations.
- Train and compare four classifiers using accuracy, macro precision/recall/F1 and confusion matrices.
- Interpret influential features and predict for unseen samples.

## 3. Dataset description and provenance
The CSV has 1,000 records and 20 columns: sample identifier, source calibration, State, District, Soil_Type, Crop, pH, EC, organic carbon, available N/P/K, sulphur, zinc, boron, iron, manganese, copper, Fertility_Score and Fertility_Class (see `DATA_DICTIONARY.md`).

Source basis named in the dataset documentation:
- AgriNet (Mendeley Data, DOI 10.17632/c7tr9j78td.1): 258 laboratory-tested Bareilly (Uttar Pradesh) records.
- Krishnagiri raw soil dataset (Mendeley Data, DOI 10.17632/7bk36mh3tc.1): 125 laboratory-tested samples.
- Dharmapuri soil data article: 125 laboratory-tested samples.

The 1,000 rows are generated synthetic teaching records calibrated on this basis; they are not the original 508 laboratory records.

## 4. Target-label methodology
`Fertility_Class` is derived from `Fertility_Score` (combining N, P, K, organic carbon, pH and EC) with thresholds Low < 40, Medium 40-69.99, High >= 70. The training script checked all 1,000 rows against these thresholds and found **0 mismatches**. The score weights are not disclosed in the dataset documentation and are not invented here. The score is excluded from the inputs because it directly defines the target (target leakage).

## 5. Data preparation
- 1,000 rows x 20 columns; **0 duplicate rows**; **0 missing values**.
- Text columns trimmed; exact duplicates would be removed.
- Inside the pipeline: median imputation + standard scaling for 12 numeric predictors; most-frequent imputation + one-hot encoding for 4 categorical predictors.
- Excluded: `Sample_ID`, `Source_Calibration`, `Fertility_Score`.

## 6. Exploratory data analysis
**Target distribution:** Medium 596 (59.6%), Low 319 (31.9%), High 85 (8.5%): the data are imbalanced, especially for High.

**Numeric summary (mean):** pH 6.97, EC 0.51 dS/m, organic carbon 7.74 g/kg, N 177.41 kg/ha, P 43.03 kg/ha, K 182.11 kg/ha, S 13.16, Zn 0.88, B 0.76, Fe 21.04, Mn 8.92, Cu 0.76 mg/kg.

**Categorical:** 8 states, 18 districts, 7 soil types (Alluvial 304, Loamy 157, Black 148, Red 133, Clay Loam 104, Sandy Loam 78, Lateritic 76) and 8 crops. These describe the synthetic file, not real prevalence.

Figures are in `outputs/figures/`: class distribution, N/P/K, organic carbon, pH, EC histograms, records by state/soil type/crop, fertility score by class.

## 7. Correlation analysis
Pearson correlation was computed across the 12 numeric predictors (`outputs/figures/correlation_heatmap.png`). All pairwise correlations are very weak (the largest absolute value is about 0.06), so the numeric predictors carry largely independent information and there is no multicollinearity concern. Correlation measures only linear association and is not causation.

## 8. Feature selection and importance
Columns were selected by role (identifier, provenance and score removed; 16 predictors kept). Permutation importance (macro F1, held-out test set, 10 repeats) for the selected SVM:

| Feature | Mean importance |
|---|---|
| Available_K_kg_ha | 0.2701 |
| Organic_Carbon_g_kg | 0.2315 |
| Available_N_kg_ha | 0.2247 |
| Available_P_kg_ha | 0.1872 |
| Zinc_mg_kg | 0.0207 |
| Crop | 0.0081 |
| pH | 0.0063 |

Other features are near zero or slightly negative (e.g. EC -0.0216), meaning shuffling them did not hurt this model on this test set. The prominence of K, organic carbon, N and P is expected because the target score is built from these variables; it is not independent evidence of agronomic causation.

## 9. Model development
Stratified 80:20 split (800 train / 200 test, `random_state=42`). Models: Decision Tree (balanced class weights), Random Forest (300 trees, balanced), KNN (k=5), RBF-SVM (C=1, balanced, probability enabled). All preprocessing sits in a scikit-learn Pipeline fitted on training data only.

## 10. Evaluation
Hold-out results (200 test rows) plus 5-fold stratified cross-validation of macro F1 on the training set:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | CV5 Macro F1 (mean ± sd) |
|---|---|---|---|---|---|
| SVM | 0.845 | 0.8424 | 0.8315 | **0.8329** | 0.7748 ± 0.0339 |
| Decision Tree | 0.835 | 0.8311 | 0.7827 | 0.8020 | 0.7292 ± 0.0542 |
| Random Forest | **0.890** | 0.9299 | 0.7535 | 0.8000 | **0.8017 ± 0.0242** |
| KNN | 0.690 | 0.6129 | 0.5645 | 0.5821 | 0.5605 ± 0.0396 |

**Selected SVM, class-wise (test set):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low Fertility | 0.75 | 0.91 | 0.82 | 64 |
| Medium Fertility | 0.91 | 0.82 | 0.86 | 119 |
| High Fertility | 0.87 | 0.76 | 0.81 | 17 |

**Confusion matrix (rows = actual, columns = predicted; Low, Medium, High):**

| | Low | Medium | High |
|---|---|---|---|
| Low | 58 | 6 | 0 |
| Medium | 19 | 98 | 2 |
| High | 0 | 4 | 13 |

One-vs-rest ROC AUC for the SVM: High 0.986, Low 0.964, Medium 0.938.

## 11. Model comparison and selection
The pre-declared rule is highest hold-out macro F1, so **SVM** is selected (0.8329). Random Forest has the highest accuracy (0.890) and the highest cross-validated macro F1 (0.8017 vs 0.7748 for SVM), so the SVM-vs-Random-Forest ranking depends on the split and metric; SVM's advantage is best described as modest. Only 17 High Fertility test rows exist, so class-wise High results are unstable. KNN is clearly weakest.

## 12. Prediction application
`app.py` (Streamlit) takes the 16 inputs, loads `models/best_soil_fertility_model.pkl`, shows the predicted class and class probabilities, and lists the model comparison and top features. It is an educational demo.

## 13. Conclusion
The project implements an end-to-end classification workflow on a 1,000-row educational dataset and compares four algorithms. SVM was selected by hold-out macro F1; K, organic carbon, N and P drive its predictions, consistent with how the labels were constructed. Because the records are synthetic and the labels are derived, the results show how well models reproduce a teaching rule, not real-world soil diagnosis.

## 14. Future scope
- Replace synthetic rows with authentic, documented laboratory records.
- Validate the label definition with soil scientists/agronomists.
- Tune hyper-parameters (grid/random search) and add repeated cross-validation.
- External validation on other regions and seasons.

## 15. How to reproduce
```bash
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```
