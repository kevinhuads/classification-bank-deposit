# -----------------------------------------------
# hyperparams.py  –  central catalogue of search
# grids for every “run_*” helper in train.py
# -----------------------------------------------

import numpy as np

GRIDS = {

    # ──────────────────────── Linear ────────────────────────
    "lr": {
        "lr__solver":       ["saga"],
        "lr__penalty":      ["l2", "l1", "elasticnet"],
        "lr__C":            [0.01, 0.1, 1, 10, 100],
        "lr__l1_ratio":     [0.0, 0.25, 0.5, 0.75, 1.0],
        "lr__class_weight": [None, "balanced"],
    },

    # ───────────────────── Gradient-boosted trees ───────────
    "lgbm": {
        "lgbm__n_estimators":      [500, 1000, 1500],
        "lgbm__learning_rate":     [0.01, 0.02, 0.05, 0.1, 0.2],
        "lgbm__max_depth":         [1, 3, 5, 7, 9, 11, 13, 15],
        "lgbm__num_leaves":        [3, 10, 50, 100, 200, 500],
        "lgbm__subsample":         [0.5, 0.7, 0.9, 1.0],
        "lgbm__colsample_bytree":  [0.5, 0.7, 0.9, 1.0],
        "lgbm__class_weight":      [None, "balanced"],
    },

    "xgb": {
        "xgb__n_estimators":      [500, 1000, 1500],
        "xgb__learning_rate":     [0.01, 0.02, 0.05, 0.1, 0.2],
        "xgb__max_depth":         [3, 5, 7, 9, 11],
        "xgb__subsample":         [0.5, 0.7, 0.9, 1.0],
        "xgb__colsample_bytree":  [0.5, 0.7, 0.9, 1.0],
        "xgb__gamma":             [0, 0.1, 0.25, 0.5],
        "xgb__min_child_weight":  [1, 5, 10],
        "xgb__scale_pos_weight":  [1, 2, 5, 10],
    },

    "cat": [
        {
            "cat__bootstrap_type":  ["Bayesian"],
            "cat__bagging_temperature": [0.25, 0.5, 0.75, 1, 1.5, 2],
            "cat__iterations":      [1000],
            "cat__learning_rate":   [0.01, 0.02, 0.05, 0.1, 0.2],
            "cat__depth":           [1, 2, 3, 5, 7, 9, 11, 13, 15],
            "cat__l2_leaf_reg":     [1, 3, 5, 7, 9],
            "cat__rsm":             [0.5, 0.7, 0.9, 1.0],
            "cat__scale_pos_weight":[1.0, 2.0, 5.0],
        },
        {
            "cat__bootstrap_type":  ["Bernoulli"],
            "cat__subsample":       [0.5, 0.7, 0.9, 1.0],
            "cat__iterations":      [1000],
            "cat__learning_rate":   [0.01, 0.02, 0.05, 0.1, 0.2],
            "cat__depth":           [1, 2, 3, 5, 7, 9, 11, 13, 15],
            "cat__l2_leaf_reg":     [1, 3, 5, 7, 9],
            "cat__rsm":             [0.5, 0.7, 0.9, 1.0],
            "cat__scale_pos_weight":[1.0, 2.0, 5.0],
        },
    ],

    "hgb": {
        "hgb__learning_rate":     [0.01, 0.05, 0.1],
        "hgb__max_depth":         [None, 3, 5, 7],
        "hgb__max_iter":          [500, 1000],
        "hgb__max_leaf_nodes":    [31, 63, 127],
        "hgb__min_samples_leaf":  [20, 50, 100],
        "hgb__l2_regularization": [0.0, 0.1, 1.0],
    },

    # ─────────────────────── Bagging Forests ────────────────
    "rf": {
        "rf__n_estimators":            [200, 500, 1000],
        "rf__max_depth":               [None, 2, 3, 5, 7, 10, 20, 30, 40],
        "rf__max_leaf_nodes":          [None, 500, 1000],
        "rf__max_features":            ["sqrt", "log2", 0.3, 0.5, 0.7],
        "rf__min_impurity_decrease":   [0.0, 1e-4, 5e-4],
        "rf__min_samples_split":       [2, 5, 10],
        "rf__min_samples_leaf":        [1, 2, 4],
        "rf__class_weight":            [None, "balanced"],
    },

    "et": {
        "et__n_estimators":      [500,1000],
        "et__max_features":      [0.3, 0.5, 0.7, "sqrt"],
        "et__max_depth":         [None, 10, 20, 40],
        "et__min_samples_leaf":  [1, 2, 4],
        "et__min_samples_split": [2, 5, 10],
        "et__bootstrap":         [False, True],
        "et__class_weight":      [None, "balanced"],
    },

    # ────────────────────── Kernel / distance ───────────────
    "knn": {
        "knn__n_neighbors":  np.arange(2, 61),
        "knn__weights":      ["uniform", "distance"],
        "knn__metric":       ["euclidean", "manhattan"],
        "knn__algorithm":    ["auto", "ball_tree", "kd_tree", "brute"],
    },

    "svc": {
        "rbf__gamma":        ["scale", 1e-3, 1e-2, 1e-1, 1],
        "rbf__n_components": [512, 1024, 2048],
        "svc__C":            [0.01, 0.1, 1, 10, 100],
        "svc__loss":         ["squared_hinge"],
    },

    # ───────────── Explainable additive model ───────────────
    "ebm": {
        "ebm__learning_rate":     [0.005, 0.01, 0.05, 0.1],
        "ebm__max_bins":          [64, 128, 256],
        "ebm__max_leaves":        [1, 3, 5],
        "ebm__interactions":      [0, 2, 8, 16],
        "ebm__min_samples_leaf":  [2, 4, 8],
        "ebm__outer_bags":        [4, 8],
    },

    # ─────────────────── Neural Nets (PyTorch) ──────────────
    "nn": {
        "nn__hidden_dim":  [32, 64, 128],
        "nn__epochs":      [50, 100, 200],
        "nn__lr":          [0.001, 0.005, 0.01],
        "nn__weight_decay":[1e-5, 1e-4, 1e-3],
        "nn__batch_size":  [1024, 2048, 4096],
    },

    # ───────────────────── TabNet wrapper ───────────────────
    "tab": {
        "tab__n_d":               [8, 16, 32, 64],
        "tab__n_a":               [8, 16, 32, 64],
        "tab__n_steps":           [3, 4, 5, 7],
        "tab__gamma":             [1.0, 1.3, 1.5, 1.8],
        "tab__lambda_sparse":     [1e-5, 1e-4, 1e-3],
        "tab__optimizer_params":  [{"lr": 0.01}, {"lr": 0.02}, {"lr": 0.05}],
    },
}

tabnet_params = dict(
    tab__max_epochs=150,
    tab__patience=10,
    tab__batch_size=2048,
    tab__virtual_batch_size=256,
)
