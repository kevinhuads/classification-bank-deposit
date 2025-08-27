# Podcast listening time prediction (Kaggle)

We predict listening time for a podcast episode given episode and show metadata. This is a regression problem optimized for root mean squared error (RMSE). Unless otherwise stated, units are minutes and lower is better.

## Project map

- `1_Podcast_EDA.ipynb` — Exploratory data analysis: distributions; missingness (MCAR/MAR/MNAR); UMAP/t-SNE; train–test drift.
- `2_Podcast_ML.ipynb` — Modeling: baselines; model families; K-fold CV (RMSE); OOF-only stacking; focused hyperparameter search.
- `3_Podcast_Final_Analysis.ipynb` — Final analysis: CV vs leaderboard; importance (native, SHAP); OOF residuals; calibration; conclusions.
- `podcast_functions.py` — Plotting style and figure helpers; small utilities.
- `podcast_hyperparameters.py` — Centralised grids and search spaces.
- `podcast_models.py` — Model constructors, CV runners, and stacking utilities.

## Method overview

### EDA highlights
- **Task and metric**: regression with RMSE (minutes); lower is better.
- **Distributions and outliers**: profile target/features; document any capping or winsorisation.
- **Features and missingness**: numeric/categorical summaries; missing data analysis with simple independence tests.
- **Structure and shift**: UMAP/t-SNE for coarse structure; basic train–test drift checks due to fixed competition split.

### Modeling approach
- **Start with baselines** :Fit simple models (for example, using episode length) to set a clear reference RMSE.
- **Use a small, diverse set of models** :Combine tree-based models and linear models, plus one or two simple alternatives. Each is included to capture a different kind of pattern.
- **Keep validation simple and consistent**: K-fold cross-validation with a shared seed; metric is RMSE (minutes). Treat very small deltas as noise.
- **Stack safely** : Train a small meta-model on **out-of-fold (OOF)** predictions only. When models perform similarly, average them.
- **Tune only what matters** :Run narrow grids on a few key settings (for example, learning rate, depth, regularisation). Prefer stability over chasing tiny gains.
- **Minimal preprocessing** :Apply consistent scaling where needed, encode categoricals, and keep feature engineering small and motivated by EDA.

### Diagnostics and interpretation
- **Importance**: native importances complemented by **SHAP** to separate true drivers from correlated artefacts.
- **Residuals**: OOF residuals checked for heteroscedasticity and subgroup effects; residuals-vs-prediction and density comparisons.
- **Calibration**: if mean reversion appears (e.g., when a key feature is missing), apply isotonic regression on OOF predictions or stratify models.
- **Generalisation**: compare CV to public/private leaderboards and interpret differences as sampling/shift rather than optimisation artefacts.
