import numpy as np
import pandas as pd
import os
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

####### ---- 

def lgb_default(seed, n_estimator=12000):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimator,
        learning_rate=0.02,
        num_leaves=32,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.7,
        boosting_type="gbdt",
        tree_learner="data",
        max_bin=255,
        n_jobs=-1,
        random_state=seed,
        device="gpu",
        verbosity=0,
    )

def xgb_default(seed , n_estimator = 12000):
    return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=n_estimator,
            learning_rate=0.02,
            grow_policy="lossguide",
            max_leaves=32,
            max_depth=0,
            subsample=0.8,
            colsample_bytree=0.7,
            tree_method="hist",
            device="cuda",
            max_bin=256,
            enable_categorical=True,
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
    )

# Model with Early Stopping Default
def model_default(X_tr, y_tr, X_va, y_va, used_features, 
                  model = "lgb", seed = 3, n_estimator = 12000, rounds = 250):
    if model == "lgb":
        clf = lgb_default(seed=seed, n_estimator = n_estimator)
        clf.fit(
            X_tr[used_features],
            y_tr,
            eval_set=[(X_va[used_features], y_va)],
            callbacks=[lgb.callback.early_stopping(rounds)]
        )
    else:
        clf = xgb_default(seed=seed, n_estimator = n_estimator)
        clf.fit(
            X_tr[used_features],
            y_tr,
            eval_set=[(X_va[used_features], y_va)],
            early_stopping_rounds=rounds,
            callbacks=[xgb.callback.EvaluationMonitor(period=500, show_stdv=False)],
            verbose=False
        )
        
    best_iter = getattr(clf, "best_iteration", None)
    if best_iter is None:
        best_iter = getattr(clf, "best_iteration_", None)
            
    return clf, best_iter


# ---------- Nested KFold Target Encoding  ----------
def kfold_target_encode(train_col, y_train, valid_col, n_splits = 5, seed = 3):
    """
    OOF mean target encoding for training rows and full-fold mapping for validation rows.
    Returns (oof_te_for_train, te_for_valid).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = pd.Series(index=train_col.index, dtype="float32")

    for tr_idx_, te_idx_ in kf.split(train_col):
        tr_idx = train_col.index[tr_idx_]
        te_idx = train_col.index[te_idx_]
        mapping = y_train.loc[tr_idx].groupby(train_col.loc[tr_idx]).mean()
        oof.loc[te_idx] = train_col.loc[te_idx].map(mapping).astype("float32")

    full_mapping = y_train.groupby(train_col).mean()
    val_te = valid_col.map(full_mapping).astype("float32")
    return oof, val_te


def run_feature_set(train_df, test_df, y, base_features, cat_candidates, set_name, use_target_encoding,
                    save_folder, model="xgb", te_inner_splits=5, seed=3, n_folds=5,
                    *, train_more_orig=None, append_orig_times=0):
    """
    Train or reuse cross-validated models with optional target encoding, then persist:
      1) Per-fold models -> {save_folder}/models/{set_name}/fold{i}.joblib
      2) OOF probabilities -> {save_folder}/oof/{set_name}_oof.csv
      3) Test predictions (mean of folds) -> {save_folder}/predictions/predictions_{set_name}.csv
    (docstring truncated — unchanged)
    """

    # --- Paths and skip logic -------------------------------------------------
    models_dir = os.path.join(save_folder, "models", set_name)
    oof_dir = os.path.join(save_folder, "oof")
    preds_dir = os.path.join(save_folder, "predictions")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(oof_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    oof_path = os.path.join(oof_dir, f"{set_name}_oof.csv")
    preds_path = os.path.join(preds_dir, f"predictions_{set_name}.csv")
    model_paths = [os.path.join(models_dir, f"fold{i}.joblib") for i in range(1, n_folds + 1)]

    model_exist = all(os.path.exists(p) for p in model_paths)
    if model_exist:
        all_exist = os.path.exists(oof_path) and os.path.exists(preds_path)
        if all_exist:
            print(f"[{set_name}] Models, OOF and predictions all exist. Skipping entirely.")
            return
        print(f"[{set_name}] All {n_folds} fold models found in {models_dir}. Reusing saved models (no retrain).")

    print(f"\n=== Running setup: {set_name} | features={len(base_features)} | TE={'on' if use_target_encoding else 'off'} | model={model} ===")

    # --- CV setup and TE candidates ------------------------------------------
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(train_df), dtype=float)
    te_candidates = [c for c in base_features if c in cat_candidates] if use_target_encoding else []

    # --- Outer CV loop --------------------------------------------------------
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y), 1):
        model_path = os.path.join(models_dir, f"fold{fold}.joblib")

        # Split data for this fold
        X_tr = train_df.iloc[tr_idx][base_features].copy()
        X_va = train_df.iloc[va_idx][base_features].copy()
        y_tr = y.iloc[tr_idx].copy()
        y_va = y.iloc[va_idx].copy()

        # Optional data augmentation by appending original data
        if train_more_orig is not None and append_orig_times > 0:
            frames = [pd.concat([X_tr, y_tr.rename("y")], axis=1)]
            for _ in range(append_orig_times):
                frames.append(train_more_orig[base_features + ["y"]])
            Xy_tr_aug = pd.concat(frames, axis=0, ignore_index=True)
            y_tr = Xy_tr_aug["y"]
            X_tr = Xy_tr_aug[base_features]
            del Xy_tr_aug

        used_features = list(base_features)
        te_cols_added = []

        # Target encoding (OOF for train, fold-valid for val)
        if use_target_encoding:
            for c in te_candidates:
                oof_te, val_te = kfold_target_encode(
                    train_col=X_tr[c],
                    y_train=y_tr,
                    valid_col=X_va[c],
                    n_splits=te_inner_splits,
                    seed=seed
                )
                te_name = f"TE_{c}"
                X_va[te_name] = val_te.astype("float32")
                if not os.path.exists(model_path):
                    X_tr[te_name] = oof_te.astype("float32")
                te_cols_added.append(te_name)
            used_features += te_cols_added

        # --- Load pre-trained model or train from scratch ---------------------
        if os.path.exists(model_path):
            clf = joblib.load(model_path)
            best_iter = getattr(clf, "best_iteration", None)
            if best_iter is None:
                best_iter = getattr(clf, "best_iteration_", None)

            # XGBoost: use iteration_range, LightGBM: use num_iteration
            if isinstance(clf, xgb.XGBClassifier) and best_iter is not None:
                va_proba = clf.predict_proba(X_va[used_features], iteration_range=(0, int(best_iter) + 1))[:, 1]
            elif (hasattr(clf, "predict_proba") and best_iter is not None and hasattr(clf, "best_iteration_")):
                # assume LightGBM-like wrapper
                va_proba = clf.predict_proba(X_va[used_features], num_iteration=int(best_iter))[:, 1]
            else:
                va_proba = clf.predict_proba(X_va[used_features])[:, 1]

            va_proba = va_proba.astype(float)
            auc = roc_auc_score(y_va, va_proba)
            print(f"[{set_name}] Fold {fold}/{n_folds} (loaded): AUC={auc:.6f} | best_iter={best_iter} | +TE_cols={len(te_cols_added)}")
            oof[va_idx] = va_proba
            continue

            
        clf, best_iter = model_default(model = model, seed = seed)

        # Use the appropriate predict_proba signature per model
        if model == "xgb" and best_iter is not None:
            va_proba = clf.predict_proba(X_va[used_features], iteration_range=(0, int(best_iter) + 1))[:, 1]
        elif model == "lgb" and best_iter is not None:
            va_proba = clf.predict_proba(X_va[used_features], num_iteration=int(best_iter))[:, 1]
        else:
            va_proba = clf.predict_proba(X_va[used_features])[:, 1]
        va_proba = va_proba.astype(float)

        auc = roc_auc_score(y_va, va_proba)
        print(f"[{set_name}] Fold {fold}/{n_folds}: AUC={auc:.6f} | best_iter={best_iter} | +TE_cols={len(te_cols_added)}")

        joblib.dump(clf, model_path, compress=3)
        print(f"[{set_name}] Saved fold {fold} -> {model_path}")

        oof[va_idx] = va_proba

    # --- Persist OOF probabilities -------------------------------------------
    pd.DataFrame({"oof": oof}).to_csv(oof_path, index=False, float_format="%.6f")
    print(f"[{set_name}] Saved OOF -> {oof_path}")

    # --- Build test features (+ mean TE) -------------------------------------
    test_X = test_df[base_features].copy()
    if use_target_encoding:
        tmp = train_df.copy()
        tmp["__y"] = y.values
        global_mean = y.mean()
        for c in te_candidates:
            mapping = tmp.groupby(c)["__y"].mean()
            test_X[f"TE_{c}"] = test_df[c].map(mapping).fillna(global_mean).astype("float32")
    used_features_all = base_features + ([f"TE_{c}" for c in te_candidates] if use_target_encoding else [])

    # --- Predict with each fold model and save aggregated predictions ---------
    preds = pd.DataFrame(index=test_df.index)
    for i, p in enumerate(model_paths, 1):
        clf = joblib.load(p)
        best_iter = getattr(clf, "best_iteration", None)
        if best_iter is None:
            best_iter = getattr(clf, "best_iteration_", None)

        if isinstance(clf, xgb.XGBClassifier) and best_iter is not None:
            proba = clf.predict_proba(test_X[used_features_all], iteration_range=(0, int(best_iter) + 1))[:, 1]
        elif best_iter is not None and hasattr(clf, "best_iteration_"):
            proba = clf.predict_proba(test_X[used_features_all], num_iteration=int(best_iter))[:, 1]
        else:
            proba = clf.predict_proba(test_X[used_features_all])[:, 1]
        preds[f"fold{i}"] = proba.astype(float)

    preds["mean"] = preds.mean(axis=1)
    pd.DataFrame({"id": test_df["id"], "y": preds["mean"]}).to_csv(preds_path, index=False, float_format="%.6f")
    print(f"[{set_name}] Saved test predictions -> {preds_path}")

# For single-shot whole-data fits. Computes TE using all labels. Do not use inside CV (would leak).
def build_full_te_matrix(df_X, y_vec, base_features, cat_candidates):
    """
    Compute full-data mean target-encoding columns TE_{feature} for features in base_features
    that are present in cat_candidates. Returns a tuple (X_with_te, used_features, te_maps, global_mean).
    """
    te_maps = {}
    te_cols = []
    X = df_X[base_features].copy()
    global_mean = float(y_vec.mean())

    for c in base_features:
        if c in cat_candidates:
            mapping = y_vec.groupby(X[c]).mean()
            X[f"TE_{c}"] = X[c].map(mapping).fillna(global_mean).astype("float32")
            te_maps[c] = mapping
            te_cols.append(f"TE_{c}")

    used_features = base_features + te_cols
    return X[used_features], used_features


# Create one stratified hold-out for early stopping when training on the whole dataset; not a CV replacement.
def fit_with_holdout(X, y, used_features=None, model="lgb", n_splits=5, seed=3, rounds=250):
    """
    Fit a tree model using a single stratified hold-out (first fold) for early stopping.
    Returns (clf, used_features, best_iter).
    """
    if used_features is None:
        used_features = list(X.columns)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(iter(skf.split(X, y)))

    X_tr = X.iloc[tr_idx][used_features].copy()
    X_va = X.iloc[va_idx][used_features].copy()
    y_tr = y.iloc[tr_idx].copy()
    y_va = y.iloc[va_idx].copy()

    clf, best_iter = model_default(
        X_tr, y_tr, X_va, y_va, used_features, 
        model = model, seed = seed, rounds = rounds
    )

    return clf, used_features, best_iter