# PROJECT REPORT
## Soil Fertility Classification Using Machine Learning

**Submitted by:** Saniya Islam  
**Programme:** Bachelor of Computer Applications (BCA)  
**Department:** Computational Sciences  
**Project category:** Project Based Learning

## Abstract
This project demonstrates a reproducible multiclass machine-learning workflow for classifying soil records into Low Fertility, Medium Fertility and High Fertility categories. It includes dataset inspection, cleaning, missing-value handling, exploratory data analysis, visualization, numerical correlation analysis, feature interpretation, model training, evaluation and an interactive prediction interface. Four algorithms are compared: Decision Tree, Random Forest, K-Nearest Neighbors and Support Vector Machine.

**Dataset limitation:** The supplied 1,000-row dataset is source-calibrated synthetic teaching data. Its fertility labels are derived from a teaching score, not independently measured laboratory target labels. The results therefore describe performance on this educational dataset and do not establish field diagnostic accuracy.

## 1. Introduction
Soil properties such as pH, electrical conductivity, organic carbon and available nutrients are important in agricultural soil assessment. Machine learning can be used to explore relationships between these attributes and a defined classification target. This project applies supervised classification to an educational dataset.

## 2. Objectives
- Understand the dataset attributes and provenance.
- Inspect data quality, duplicates and missing values.
- Perform EDA and visualizations.
- Study correlations among numerical variables.
- Train and compare four classification algorithms.
- Evaluate models using accuracy, macro precision, macro recall, macro F1 and confusion matrices.
- Interpret influential input features.
- Demonstrate prediction for an unseen soil record.

## 3. Dataset description and provenance
The provided CSV contains 1,000 records and 20 columns. Attributes include sample identifier, source calibration, state, district, soil type, crop, pH, EC, organic carbon, available N/P/K, sulphur, zinc, boron, iron, manganese, copper, fertility score and fertility class.

The accompanying documentation identifies the source basis as:
- AgriNet (Mendeley Data, DOI: 10.17632/c7tr9j78td.1): 258 laboratory-tested Bareilly, Uttar Pradesh soil records with pH, EC, OC and nutrient measurements.
- Krishnagiri raw soil dataset (Mendeley Data, DOI: 10.17632/7bkb36mh3tc.1): 125 laboratory-tested samples.
- Dharmapuri soil data article: 125 laboratory-tested samples.

The 1,000 rows in this project are generated synthetic teaching records calibrated using this source basis. They are not the original 508 laboratory records and must not be described as such.

## 4. Target-label methodology
The target `Fertility_Class` is derived for teaching. Documentation says the score combines N, P, K, organic carbon, pH and EC, with thresholds:
- Low: score below 40
- Medium: 40 to 69.99
- High: 70 or above

The score is excluded from the model input because it directly defines the target. This avoids direct target leakage. The model is learning patterns associated with a constructed label, not independently verified soil fertility.

## 5. Data preparation
The training script:
- Reads the supplied CSV and checks its structure.
- Removes exact duplicate rows, if present.
- records missing-value counts.
- Imputes numerical variables using median and categorical variables using most-frequent values inside the pipeline.
- Uses one-hot encoding for categorical variables and standard scaling for numerical variables.
- Excludes identifier/provenance columns and the derived score from predictors.

The uploaded dataset audit and exact counts are saved in `outputs/reports/data_audit_and_run_metadata.json` after running the training script.

## 6. Exploratory data analysis
The project creates charts for:
- Target class distribution.
- Available nitrogen, phosphorus and potassium.
- Organic carbon, pH and electrical conductivity.
- Records by state, soil type and crop.
- Fertility score distribution across the teaching classes.
- Correlation heatmap of numerical soil attributes.

These visualizations are saved in `outputs/figures/`.

## 7. Feature selection and interpretation
Permutation importance is calculated using the held-out test set and macro F1 scoring. A feature's importance represents the change in model score when that feature's values are shuffled. It is a model-specific diagnostic, not proof of causation. Results are saved in `outputs/feature_importance_results.csv`.

## 8. Model development
A stratified 80:20 train-test split is used with random state 42. The following models are compared:
1. Decision Tree
2. Random Forest
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (SVM)

All preprocessing is included in a scikit-learn pipeline and is fitted only on training data.

## 9. Evaluation
For each model, the project reports:
- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Confusion matrix

The best model is selected using the highest macro F1-score, with accuracy as a tie-breaker. The numerical results are generated by running `python -m src.train` and stored in `outputs/model_comparison_results.csv`. Do not manually insert unverified metric values into this report.

## 10. Prediction application
The Streamlit application accepts soil parameters and categorical context, then loads the selected trained pipeline to predict one of the three teaching classes. It displays class probabilities where supported. This interface is an educational demonstration only.

## 11. Conclusion
The project implements an end-to-end supervised classification workflow and compares four common algorithms on a stratified holdout set. The selected model is determined by macro F1 and accuracy. The outputs support reproducibility and interpretation of the educational dataset.

The synthetic nature of the records and derived target labels is a central limitation. Independent laboratory observations and expert-validated labels are required before claiming real-world generalization or using predictions for agricultural decisions.

## 12. Future scope
- Replace synthetic rows with authentic, documented laboratory records.
- Validate the label definition with soil scientists/agronomists.
- Collect independent samples from additional regions and seasons.
- Perform external validation and monitor model drift.
- Add agronomist-reviewed recommendations only after domain validation.

## 13. How to reproduce
Follow the installation and execution commands in the repository `README.md`. The script creates the model, metrics, figures and audit files.


## 14. Results from the supplied CSV in this prepared build

The project script was executed on the uploaded CSV. The observed results are:

```text
SOIL FERTILITY CLASSIFICATION — RUN SUMMARY

Dataset: 1,000-instance source-calibrated synthetic teaching dataset
Target: Fertility_Class (Low, Medium, High)
Rows after duplicate removal: 1000
Duplicate rows removed: 0
Missing values in supplied dataset: 0
Training rows: 800 | Testing rows: 200
Selected model: SVM

Model comparison:
        Model  Accuracy  Precision_macro  Recall_macro  F1_macro
          SVM     0.865         0.873594      0.806679  0.831814
Decision Tree     0.835         0.831056      0.782651  0.802041
Random Forest     0.870         0.930917      0.696604  0.745300
          KNN     0.750         0.764401      0.590905  0.624414

Important: Fertility_Score is excluded from model inputs because it is used to derive Fertility_Class.
This avoids direct target leakage. Results measure learning of the dataset's derived teaching labels,
not validation against independent laboratory-measured fertility outcomes.
```

The SVM was selected by macro F1, even though Random Forest had slightly higher accuracy. This selection rule gives more weight to balanced class-wise performance; the minority High Fertility class is much less represented than Medium Fertility. These metrics are still limited to this synthetic teaching dataset.
