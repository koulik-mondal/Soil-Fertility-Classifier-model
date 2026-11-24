from pathlib import Path
import os, json, warnings
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "soil_fertility_classification_1000.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
REPORTS = OUT / "reports"
MODELS = ROOT / "models"
for p in [FIG, REPORTS, MODELS]:
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "Fertility_Class"
CLASS_ORDER = ["Low Fertility", "Medium Fertility", "High Fertility"]
NUMERIC = [
    "pH", "Electrical_Conductivity_dS_m", "Organic_Carbon_g_kg",
    "Available_N_kg_ha", "Available_P_kg_ha", "Available_K_kg_ha",
    "Sulphur_mg_kg", "Zinc_mg_kg", "Boron_mg_kg", "Iron_mg_kg",
    "Manganese_mg_kg", "Copper_mg_kg",
]
CATEGORICAL = ["State", "District", "Soil_Type", "Crop"]
EXCLUDED = ["Sample_ID", "Source_Calibration", "Fertility_Score"]


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(DATA)
    initial_shape = df.shape
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].astype(str).str.strip()
    duplicates = int(df.duplicated().sum())
    missing_before = df.isna().sum().to_dict()
    df = df.drop_duplicates().copy()

    if not set(df[TARGET].unique()).issubset(set(CLASS_ORDER)):
        raise ValueError("Unexpected target labels found. Review the dataset before training.")

    # Verify the documented thresholds (Low <40, Medium 40-69.99, High >=70)
    expected = pd.cut(df["Fertility_Score"], [-np.inf, 40, 70, np.inf],
                      labels=CLASS_ORDER, right=False)
    mismatches = int((expected.astype(str) != df[TARGET].astype(str)).sum())

    X = df[NUMERIC + CATEGORICAL].copy()
    y = df[TARGET].copy()

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                                class_weight="balanced", n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True,
                   random_state=RANDOM_STATE),
    }
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
            "CV5_F1_macro_mean": cv_f1.mean(),
            "CV5_F1_macro_std": cv_f1.std(),
        })
        fitted[name] = pipe
        cm = confusion_matrix(y_test, pred, labels=CLASS_ORDER)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_ORDER).plot(
            ax=ax, cmap="Blues", colorbar=False, xticks_rotation=20)
        ax.set_title(f"Confusion Matrix - {name}")
        savefig("confusion_matrix_" + name.lower().replace(" ", "_") + ".png")

    results = pd.DataFrame(rows).sort_values(["F1_macro", "Accuracy"], ascending=False)
    results.to_csv(OUT / "model_comparison_results.csv", index=False)
    best_name = results.iloc[0]["Model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, MODELS / "best_soil_fertility_model.pkl")

    best_pred = best_model.predict(X_test)
    report = classification_report(y_test, best_pred, labels=CLASS_ORDER, zero_division=0)
    (REPORTS / "classification_report.txt").write_text(
        f"Selected model: {best_name}\n\n{report}", encoding="utf-8")
    pd.DataFrame({"Actual": y_test.values, "Predicted": best_pred}).to_csv(
        REPORTS / "test_predictions.csv", index=False)
    cm_best = confusion_matrix(y_test, best_pred, labels=CLASS_ORDER)
    pd.DataFrame(cm_best, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
        REPORTS / "confusion_matrix_best_model.csv")

    # ---- EDA ----
    df.describe(include="all").to_csv(REPORTS / "descriptive_statistics.csv")
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x=TARGET, order=CLASS_ORDER, color="steelblue")
    plt.title("Fertility Class Distribution"); plt.xticks(rotation=15)
    savefig("fertility_class_distribution.png")

    for col, filename, title in [
        ("Available_K_kg_ha", "available_potassium_distribution.png", "Available Potassium Distribution"),
        ("Organic_Carbon_g_kg", "organic_carbon_distribution.png", "Organic Carbon Distribution"),
        ("pH", "soil_ph_distribution.png", "Soil pH Distribution"),
        ("Electrical_Conductivity_dS_m", "electrical_conductivity_distribution.png", "Electrical Conductivity Distribution"),
        ("Available_N_kg_ha", "available_nitrogen_distribution.png", "Available Nitrogen Distribution"),
        ("Available_P_kg_ha", "available_phosphorus_distribution.png", "Available Phosphorus Distribution"),
    ]:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col], kde=True, color="steelblue")
        plt.title(title)
        savefig(filename)

    plt.figure(figsize=(11, 8))
    corr = df[NUMERIC].corr()
    corr.to_csv(REPORTS / "correlation_matrix.csv")
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Correlation Heatmap - Numerical Soil Features")
    savefig("correlation_heatmap.png")

    for col, filename, title in [
        ("State", "records_by_state.png", "Records by State"),
        ("Soil_Type", "soil_type_distribution.png", "Soil Type Distribution"),
        ("Crop", "crop_distribution.png", "Crop Distribution"),
    ]:
        plt.figure(figsize=(9, 5))
        sns.countplot(data=df, y=col, order=df[col].value_counts().index, color="steelblue")
        plt.title(title)
        savefig(filename)

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET, y="Fertility_Score", order=CLASS_ORDER, color="steelblue")
    plt.title("Derived Fertility Score by Teaching Label")
    savefig("fertility_score_by_class.png")

    plot_df = results.melt(id_vars="Model",
                           value_vars=["Accuracy", "Precision_macro", "Recall_macro", "F1_macro"],
                           var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="deep")
    plt.ylim(0, 1.05)
    plt.title("Model Performance Comparison (Stratified 80/20 Holdout)")
    savefig("model_comparison.png")

    # ---- ROC (one-vs-rest) ----
    try:
        probs = best_model.predict_proba(X_test)
        classes = list(best_model.named_steps["classifier"].classes_)
        y_bin = label_binarize(y_test, classes=classes)
        plt.figure(figsize=(7, 6))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            plt.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Chance")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"One-vs-Rest ROC Curves - {best_name}")
        plt.legend()
        savefig("roc_curves_best_model.png")
    except Exception as exc:
        warnings.warn(f"ROC curve could not be generated: {exc}")

    # ---- Permutation importance ----
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=10,
                                  random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=-1)
    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_Mean": perm.importances_mean,
        "Importance_Std": perm.importances_std,
    }).sort_values("Importance_Mean", ascending=False)
    importance.to_csv(OUT / "feature_importance_results.csv", index=False)
    top = importance.head(15).sort_values("Importance_Mean")
    plt.figure(figsize=(9, 6))
    plt.barh(top["Feature"], top["Importance_Mean"], xerr=top["Importance_Std"])
    plt.xlabel("Decrease in macro F1 after permutation")
    plt.title("Top Feature Importance (Permutation)")
    savefig("top_15_feature_importance.png")

    audit = {
        "dataset_file": "data/soil_fertility_classification_1000.csv",
        "initial_rows": int(initial_shape[0]), "initial_columns": int(initial_shape[1]),
        "duplicate_rows_removed": duplicates,
        "score_vs_label_mismatches": mismatches,
        "missing_values_before_cleaning": {k: int(v) for k, v in missing_before.items()},
        "target_distribution": {k: int(v) for k, v in y.value_counts().to_dict().items()},
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "split": "80% train / 20% test; stratified; random_state=42",
        "features_used": NUMERIC + CATEGORICAL,
        "excluded_from_features": EXCLUDED,
        "selected_model": best_name,
        "selection_rule": "Highest macro F1-score; accuracy used as tie-breaker",
        "dataset_caveat": "1,000 source-calibrated synthetic teaching rows; Fertility_Class is derived from a teaching score, not a laboratory-measured label.",
    }
    (REPORTS / "data_audit_and_run_metadata.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    summary = f"""SOIL FERTILITY CLASSIFICATION - RUN SUMMARY

Dataset: 1,000-instance source-calibrated synthetic teaching dataset
Target: Fertility_Class (Low, Medium, High)
Rows after duplicate removal: {len(df)}
Duplicate rows removed: {duplicates}
Score/label threshold mismatches: {mismatches}
Missing values in supplied dataset: {int(df.isna().sum().sum())}
Training rows: {len(X_train)} | Testing rows: {len(X_test)}
Selected model: {best_name}

Model comparison:
{results.round(4).to_string(index=False)}

Important: Fertility_Score is excluded from model inputs because it is used to derive Fertility_Class.
This avoids direct target leakage. Results measure learning of the dataset's derived teaching labels,
not validation against independent laboratory-measured fertility outcomes.
"""
    (REPORTS / "run_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Artifacts saved under: {ROOT}")


if __name__ == "__main__":
    main()
