# Soil Fertility Classification Using Machine Learning

**Project by:** Saniya Islam  
**Programme:** BCA | **Department:** Computational Sciences  
**Project type:** Project Based Learning (PBL)

## 1. Project objective
Build and compare multiclass machine-learning classifiers that predict one of three dataset teaching labels: **Low Fertility**, **Medium Fertility**, or **High Fertility**.

## 2. Important dataset provenance and scientific limitation
This repository uses the supplied `soil_fertility_classification_1000.csv` (1,000 rows). The accompanying dataset documentation states:

- The 1,000 rows are **source-calibrated synthetic teaching data** prepared for BCA PBL; they are not 1,000 copied or authentic laboratory observations.
- The source basis includes AgriNet (Mendeley Data, DOI `10.17632/c7tr9j78td.1`), Krishnagiri raw soil dataset (Mendeley Data, DOI `10.17632/7bkb36mh3tc.1`), and a Dharmapuri soil data article. These sources contain laboratory-tested soil records, but the project CSV is a synthetic teaching dataset calibrated from that basis.
- `Fertility_Class` is a **derived teaching label**, not an original field-measured laboratory label. Its score combines N, P, K, organic carbon, pH and EC. The documentation gives: Low < 40; Medium 40–69.99; High >= 70.
- Therefore, model performance evaluates how well algorithms learn these derived labels in this educational dataset. It must not be presented as proven real-world soil diagnosis accuracy.

For publication or real agricultural deployment, replace synthetic rows with authentic underlying laboratory records, retain their provenance, and validate the target methodology with domain experts.

## 3. Dataset attributes
- Identification/provenance: `Sample_ID`, `Source_Calibration`
- Categorical/context: `State`, `District`, `Soil_Type`, `Crop`
- Numerical soil parameters: `pH`, `Electrical_Conductivity_dS_m`, `Organic_Carbon_g_kg`, `Available_N_kg_ha`, `Available_P_kg_ha`, `Available_K_kg_ha`, `Sulphur_mg_kg`, `Zinc_mg_kg`, `Boron_mg_kg`, `Iron_mg_kg`, `Manganese_mg_kg`, `Copper_mg_kg`
- Teaching target construction field: `Fertility_Score`
- Target: `Fertility_Class`

`Fertility_Score` is deliberately excluded from model features because it is used to derive the target class. Including it would cause direct target leakage and produce an overly optimistic evaluation.

## 4. Workflow
1. Load and inspect dataset, data types, shape and target distribution.
2. Remove exact duplicate rows (if any) and audit missing values.
3. Impute numerical missing values with median and categorical missing values with most frequent value inside a leakage-safe training pipeline.
4. EDA: distributions of soil properties, class counts, crop/state/soil-type counts and score by class.
5. Correlation analysis among numerical soil properties.
6. Feature selection/interpretation using permutation importance on held-out test data.
7. Stratified 80/20 train-test split (random state 42).
8. Compare Decision Tree, Random Forest, KNN and SVM.
9. Evaluate accuracy, macro precision, macro recall, macro F1 and confusion matrices; create one-vs-rest ROC curves for the selected model.
10. Select the model with the highest macro F1; use accuracy as tie-breaker.
11. Save trained pipeline and provide a Streamlit prediction interface.

## 5. Project structure
```text
Soil_Fertility_Classification_Saniya_Islam/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── soil_fertility_classification_1000.csv
├── src/
│   ├── __init__.py
│   └── train.py
├── models/
│   └── best_soil_fertility_model.pkl   # created after training
├── outputs/
│   ├── figures/                       # EDA and evaluation charts
│   └── reports/                       # metrics, audit and predictions
└── docs/
    └── PROJECT_REPORT.md
```

## 6. Run in VS Code
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

## 7. Generated outputs
Training generates:
- `models/best_soil_fertility_model.pkl`
- `outputs/model_comparison_results.csv`
- `outputs/feature_importance_results.csv`
- `outputs/figures/` — class distribution, feature distributions, correlation heatmap, model comparison, confusion matrices, ROC and feature importance
- `outputs/reports/` — classification report, test predictions, run summary and data audit metadata

## 8. Reproducibility
The split and estimators use random seed 42 where applicable. All preprocessing is fitted inside each model pipeline on training data only. The test set is held out for final comparison.

## 9. Responsible interpretation
This is an academic ML demonstration. A model prediction should not be used to decide fertilizer application or crop treatment without current soil laboratory testing and agronomic advice.
