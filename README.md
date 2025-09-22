# Bank marketing — classification (Kaggle)

Predict whether a customer subscribes to a term deposit from campaign, demographic and behavioral features. Binary classification; primary metric is ROC AUC (higher is better). Secondary diagnostics: average precision, log loss, calibration.

## Project map
- `1_Bank_EDA.ipynb` - EDA: class balance, feature distributions, missingness, basic drift, UMAP/t-SNE.  
- `2_Bank_ML.ipynb` - Modeling: baselines, compact model families, stratified K-fold CV, OOF metrics, light stacking, focused searches.  
- `3_Bank_Features_Engineering.ipynb` - Features: safe categorical encodings (counts/frequencies without test `y`), numeric transforms, persisted engineered sets and OOF blends.  
- `4_Bank_Final_Analysis.ipynb` - Final: CV vs holdout, native importances + SHAP, calibration, subgroup analysis, model export.  
- `bank_functions.py` - Plotting, metrics, CV and evaluation helpers.  
- `bank_feat_engineer.py` - Reusable feature transformations and pipelines.  
- `bank_models.py` - Model constructors, CV runners and ensembling utilities
- `bank_hyperparameters.py` - Centralised hyperparameter grids.

## Method summary
- **EDA** - Quantify imbalance and missingness; inspect distributions and drift.  
- **Modeling** - Start with interpretable baselines; use a small, diverse set of learners; evaluate via stratified K-fold OOF ROC AUC; prefer simple ensembles (averaging or light stacking on OOF).  
- **Tuning** - Limit to a few high-impact hyperparameters for repeatability.  
- **Feature practice** - Encode and blend without using test targets; keep preprocessing explicit and minimal.  
- **Diagnostics** - Use native importances + SHAP; assess calibration (Brier, calibration curves) and per-group performance.

## Reproducibility & portability
- Notebooks follow a linear narrative - EDA -> experiments -> features -> final analysis.  
- Helper modules are imported by notebooks; avoid duplicating code.  
- Persist OOF blends and engineered artifacts for reproducible reruns.