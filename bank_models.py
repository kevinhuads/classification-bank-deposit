# --- Standard lib
import math

# --- Third-party
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_halving_search_cv  
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

# --- Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Project
from bank_functions import *  # provides SEED, N_FOLDS, etc.
from bank_hyperparameters import GRIDS, tabnet_params


# ==============================
# Global config
# ==============================
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
N_CANDIDATES = 30
FACTOR = 3
SCORING = "roc_auc"
VERBOSE = 1
LR_ITER = 25000
SVC_ITER = 100000
SVC_TOL = 1e-3
N_JOBS = 8

# ==============================
# Utilities
# ==============================
def search_kw(n_jobs = -1, include_seed = True, 
              halving=False, n_samples = 50000, factor=FACTOR, min_frac=0.10, **override):
    """
    Common kwargs for sklearn *SearchCV objects.
    Adds halving-specific keys when halving=True.
    """
    kw = {"scoring": SCORING, "cv": cv, "verbose": VERBOSE, "n_jobs": n_jobs}
    if include_seed:
        kw["random_state"] = SEED
    if halving:
        kw.update(
            resource="n_samples",
            max_resources=n_samples,
            min_resources=int(n_samples * min_frac),
            factor=factor,
        )
    kw.update(override)
    return kw


def expected_fits(search):
    """
    Estimate total fits for a SearchCV object to drive tqdm progress.
    """
    _cv = search.cv
    if isinstance(search, GridSearchCV):
        grid = search.param_grid
        n_points = int(np.prod([len(v) for v in grid.values()]))
        return n_points * _cv.get_n_splits()

    if isinstance(search, RandomizedSearchCV):
        return search.n_iter * _cv.get_n_splits()

    if isinstance(search, HalvingRandomSearchCV):
        n_rounds = math.floor(math.log(search.n_candidates, search.factor)) + 1
        cand_per_round = [max(1, math.floor(search.n_candidates / (search.factor**r))) for r in range(n_rounds)]
        return _cv.get_n_splits() * sum(cand_per_round)

    return None


def fit_model(search, X, y, **fit_kwargs):
    """
    Fit a SearchCV with a progress bar and persist it with joblib.
    """
    total_fits = expected_fits(search)
    with tqdm_joblib(tqdm(total=total_fits)):
        search.fit(X, y, **fit_kwargs)
    return search


def write_predictions(df, model, features, target, path):
    """
    Write a CSV with id + predicted target.
    """
    output = df[["id"]].copy()
    output[target] = model.predict_proba(df[features])[:,1]
    output.to_csv(path, index=False)


# ==============================
# Preprocessing
# ==============================
def make_preprocessor(X_num, X_cat, 
                      onehot=False, impute=False, sparse_output=True, scaler=False, ordinal=False):
    num_step = ("num", StandardScaler() if scaler else "passthrough", X_num)

    if onehot:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
        ohe = ohe.set_output(transform="default" if sparse_output else "pandas")
        cat_step = ("cat", ohe, X_cat)
        remainder_kw = "passthrough"
    else:
        if ordinal:
            cat_step = ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), X_cat)
        else:
            cat_step = ("cat", "passthrough", X_cat)
        remainder_kw = "drop"

    core = ColumnTransformer(
        transformers=[num_step, cat_step],
        remainder=remainder_kw,
        verbose_feature_names_out=False,
    )

    if not impute:
        return core

    mice_block = ColumnTransformer(
        transformers=[
            ("mice",
             IterativeImputer(estimator=BayesianRidge(), max_iter=100, random_state=SEED).set_output(transform="pandas"),
             X_num),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return Pipeline([
        ("impute", mice_block),
        ("prep", core.set_output(transform="default")),
    ])




# ==============================
# Search builders
# ==============================
def _pipe(preprocessor, step_name, estimator):
    """Build a Pipeline with a preprocessing step and a named estimator step."""
    return Pipeline([("pre", preprocessor), (step_name, estimator)])

def _rand_search(pipe, grid_key, n_iter, n_jobs):
    """Create a RandomizedSearchCV over GRIDS[grid_key] with shared search defaults."""
    return RandomizedSearchCV(
        estimator=pipe,
        param_distributions=GRIDS[grid_key],
        n_iter=n_iter,
        **search_kw(n_jobs=n_jobs),
    )

def _grid_search(pipe, grid_key, n_jobs):
    """Create a GridSearchCV over GRIDS[grid_key] with shared search defaults."""
    return GridSearchCV(
        estimator=pipe,
        param_grid=GRIDS[grid_key],
        **search_kw(n_jobs=n_jobs, include_seed=False),
    )

def _halving_search(pipe, grid_key, n_candidates, X, factor, n_jobs):
    """Create a HalvingRandomSearchCV over GRIDS[grid_key] using sample-based resources."""
    return HalvingRandomSearchCV(
        estimator=pipe,
        param_distributions=GRIDS[grid_key],
        n_candidates=n_candidates,
        **search_kw(n_jobs=n_jobs, halving=True, n_samples=len(X), factor=factor),
    )

# ==============================
# Model runners
# ==============================

# --- Linear
def run_LogReg(X, y, preprocessor, n_candidates=N_CANDIDATES, n_jobs=N_JOBS):
    pipe = _pipe(preprocessor, "lr", LogisticRegression(random_state=SEED, max_iter=LR_ITER))
    search = _halving_search(pipe, "lr", n_candidates=n_candidates, X=X, factor=FACTOR, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search

# --- Gradient Boosting
def run_lightGBM(X, y, preprocessor, n_candidates=N_CANDIDATES, n_jobs=N_JOBS):
    pipe = _pipe(preprocessor, "lgbm", LGBMClassifier(random_state=SEED, verbosity=-1))
    search = _halving_search(pipe, "lgbm", n_candidates=n_candidates, X=X, factor=FACTOR, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search

def run_XGBoost(X, y, preprocessor, n_candidates=N_CANDIDATES, n_jobs=N_JOBS):
    pipe = _pipe(preprocessor, "xgb", XGBClassifier(
        enable_categorical=True,
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
        random_state=SEED,
    ))
    search = _halving_search(pipe, "xgb", n_candidates=n_candidates, X=X, factor=FACTOR, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search


def run_CatBoost(X, y, preprocessor, X_cat, n_candidates=N_CANDIDATES, n_jobs=4):
    pipe = _pipe(preprocessor, "cat", CatBoostClassifier(
        loss_function="Logloss",
        random_state=SEED,
        verbose=False,
        allow_writing_files=False,
        cat_features=X_cat,
    ))
    search = _halving_search(pipe, "cat", n_candidates=n_candidates, X=X, factor=FACTOR, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search


def run_HGB(X, y, preprocessor, n_candidates=N_CANDIDATES, factor=FACTOR, n_jobs=N_JOBS):
    pipe = _pipe(preprocessor, "hgb", HistGradientBoostingClassifier(random_state=SEED))
    search = _halving_search(pipe, "hgb", n_candidates=n_candidates, X=X, factor=factor, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search

# --- Bagging Trees
def run_randomForest(X, y, preprocessor, n_candidates=N_CANDIDATES, n_jobs=N_JOBS):
    pipe = _pipe(preprocessor, "rf", RandomForestClassifier(
        random_state=SEED,
        n_jobs=1,
        oob_score=True,
        bootstrap=True,
    ))
    search = _halving_search(pipe, "rf", n_candidates=n_candidates, X=X, factor=FACTOR, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search


def run_extraTrees(X, y, preprocessor, n_candidates=N_CANDIDATES, factor=FACTOR, n_jobs=N_JOBS):
    pipe = _pipe(preprocessor, "et", ExtraTreesClassifier(random_state=SEED, n_jobs=-1))
    search = _halving_search(pipe, "et", n_candidates=n_candidates, X=X, factor=factor, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search

# --- Kernel Based
def run_KNN(X, y, preprocessor, n_candidates=N_CANDIDATES, factor=FACTOR, n_jobs=N_JOBS):
    pipe = _pipe(preprocessor, "knn", KNeighborsClassifier(n_jobs=n_jobs))
    search = _halving_search(pipe, "knn", n_candidates=n_candidates, X=X, factor=factor, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search


def run_SVC(X, y, preprocessor, n_candidates=N_CANDIDATES, factor=FACTOR, n_jobs=N_JOBS):
    base_pipe = Pipeline([
        ("pre", preprocessor),
        ("rbf", RBFSampler(random_state=SEED)),
        ("svc", LinearSVC(max_iter=SVC_ITER, tol=SVC_TOL, random_state=SEED)),
    ])

    search = _halving_search(base_pipe, "svc", n_candidates=n_candidates, X=X, factor=factor, n_jobs=1)
    fit_model(search, X, y)

    # Calibrate the *best* pipeline to expose predict_proba
    # Use CV to avoid fitting the calibrator on the same data without guarding leakage.
    calibrator = CalibratedClassifierCV(
        estimator=search.best_estimator_,  
        method="sigmoid",                 
        cv=cv                    
    )
    calibrator.fit(X, y)
    # Make downstream code keep working: overwrite best_estimator_ with the calibrated model
    search.best_estimator_ = calibrator
    return search


# --- Explainable additive
def run_EBM(X, y, preprocessor, n_candidates=N_CANDIDATES, factor=FACTOR, n_jobs=N_JOBS):
    pipe = _pipe(preprocessor, "ebm", ExplainableBoostingClassifier(random_state=SEED, n_jobs=n_jobs))
    search = _halving_search(pipe, "ebm", n_candidates=n_candidates, X=X, factor=factor, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search


class TorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_dim=64,
        n_hidden=2,
        dropout=0.10,
        lr=0.005,
        weight_decay=1e-4,
        batch_size=2048,
        epochs=100,
        max_grad_norm=None,
        class_weight=None,          # None | "balanced" | array-like (n_classes,)
        verbose=False,
        random_state=None,
        device="cuda",              # "auto" | "cpu" | "cuda"
    ):
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        self.model_ = None
        self.is_fitted_ = False

    # ---- sklearn plumbing
    def get_params(self, deep=True):
        return {
            "hidden_dim": self.hidden_dim,
            "n_hidden": self.n_hidden,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "max_grad_norm": self.max_grad_norm,
            "class_weight": self.class_weight,
            "verbose": self.verbose,
            "random_state": self.random_state,
            "device": self.device,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __setstate__(self, state):
        self.__dict__.update(state)
        defaults = {
            "n_hidden": 2, "dropout": 0.10, "weight_decay": 1e-4, "batch_size": 2048,
            "max_grad_norm": None, "class_weight": None, "verbose": False,
            "random_state": None, "device": "auto", "model_": None, "is_fitted_": False,
        }
        for k, v in defaults.items():
            if k not in self.__dict__:
                setattr(self, k, v)

    # ---- helpers
    def _resolve_device(self):
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @staticmethod
    def _to_numpy_float32(X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        elif hasattr(X, "to_numpy"):
            X = X.to_numpy()
        return np.asarray(X, dtype=np.float32)

    def _build_mlp(self, n_in, n_out):
        layers, last = [], n_in
        for _ in range(self.n_hidden):
            layers += [nn.Linear(last, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout)]
            last = self.hidden_dim
        layers += [nn.Linear(last, n_out)]
        return nn.Sequential(*layers)

    def _prepare_targets(self, y):
        y_arr = y.values if isinstance(y, pd.Series) else y
        classes = np.unique(y_arr)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_enc = np.asarray([class_to_idx[v] for v in y_arr], dtype=np.int64)
        return classes, y_enc

    def _make_dataloader(self, X_cpu_t, y_cpu_t, device):
        # Keep tensors on CPU for DataLoader; let it pin CPU memory if using CUDA.
        ds = TensorDataset(X_cpu_t, y_cpu_t)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=(device == "cuda"),
        )

    # ---- core API
    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        X, y = self._validate_data(X, y, accept_sparse=True, y_numeric=False, multi_output=False)
        X = self._to_numpy_float32(X)

        self.classes_, y_enc = self._prepare_targets(y)
        n_classes = len(self.classes_)
        n_features = self.n_features_in_
        dev = self._resolve_device()

        self.model_ = self._build_mlp(n_features, n_classes).to(dev)

        # class weights
        if self.class_weight is None:
            weight_t = None
        elif isinstance(self.class_weight, str) and self.class_weight == "balanced":
            counts = np.bincount(y_enc, minlength=n_classes).astype(np.float32)
            inv = 1.0 / np.maximum(counts, 1.0)
            weight_vec = (inv * (counts.sum() / (n_classes * inv.sum()))).astype(np.float32)
            weight_t = torch.tensor(weight_vec, device=dev)
        else:
            weight_vec = np.asarray(self.class_weight, dtype=np.float32)
            if weight_vec.shape[0] != n_classes:
                raise ValueError("class_weight length must equal number of classes")
            weight_t = torch.tensor(weight_vec, device=dev)

        loss_fn = nn.CrossEntropyLoss(weight=weight_t)
        opt = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # CPU tensors for DataLoader; move per batch with non_blocking when CUDA
        X_t_cpu = torch.from_numpy(X)          # CPU
        y_t_cpu = torch.from_numpy(y_enc)      # CPU
        loader = self._make_dataloader(X_t_cpu, y_t_cpu, dev)

        for epoch in range(self.epochs):
            self.model_.train()
            running, n_batches = 0.0, 0
            for xb_cpu, yb_cpu in loader:
                xb = xb_cpu.to(dev, non_blocking=True) if dev == "cuda" else xb_cpu
                yb = yb_cpu.to(dev, non_blocking=True) if dev == "cuda" else yb_cpu

                opt.zero_grad(set_to_none=True)
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.max_grad_norm)
                opt.step()
                running += loss.item()
                n_batches += 1
            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"[epoch {epoch+1}/{self.epochs}] loss={running / max(1, n_batches):.5f}")

        self.is_fitted_ = True
        return self

    def _forward_logits(self, X):
        check_is_fitted(self, attributes=["is_fitted_", "model_"])
        X = self._validate_data(X, accept_sparse=True, reset=False)
        X = self._to_numpy_float32(X)
        dev = self._resolve_device()
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).to(dev)
            logits = self.model_(X_t).detach().cpu().numpy()
        return logits

    def predict(self, X):
        logits = self._forward_logits(X)
        preds = logits.argmax(axis=1)
        return self.classes_[preds]

    def predict_proba(self, X):
        logits = self._forward_logits(X)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return probs

    def decision_function(self, X):
        logits = self._forward_logits(X)
        if logits.ndim == 2 and logits.shape[1] == 2:
            return logits[:, 1]  # positive-class score for binary tasks
        return logits




def run_NeuralNetwork(X, y, preprocessor, n_candidates=N_CANDIDATES, factor=FACTOR, n_jobs=N_JOBS):
    pipe = _pipe(preprocessor, "nn", TorchClassifier(verbose=False, random_state=SEED))
    search = _halving_search(pipe, "nn", n_candidates=n_candidates, X=X, factor=factor, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search


class SkTabNet(ClassifierMixin, BaseEstimator):
    def __init__(self, *, n_d=8, n_a=8, n_steps=3, gamma=1.3,
                 cat_idxs=(), cat_dims=(), cat_emb_dim=1, **kwargs):
        self.n_d, self.n_a, self.n_steps, self.gamma = n_d, n_a, n_steps, gamma
        self.cat_idxs, self.cat_dims, self.cat_emb_dim = tuple(cat_idxs), tuple(cat_dims), cat_emb_dim
        self.kwargs = dict(kwargs)  # ensure it's a plain dict

    def __setstate__(self, state):
        # Backward compatibility for older pickles
        self.__dict__.update(state)
        if "kwargs" not in self.__dict__ or self.kwargs is None:
            self.kwargs = {}

    def get_params(self, deep=True):
        # Use getattr to be robust to missing attributes in older pickles
        return {
            **dict(getattr(self, "kwargs", {})),
            "n_d": getattr(self, "n_d", 8),
            "n_a": getattr(self, "n_a", 8),
            "n_steps": getattr(self, "n_steps", 3),
            "gamma": getattr(self, "gamma", 1.3),
            "cat_idxs": tuple(getattr(self, "cat_idxs", ())),
            "cat_dims": tuple(getattr(self, "cat_dims", ())),
            "cat_emb_dim": getattr(self, "cat_emb_dim", 1),
        }

    def set_params(self, **params):
        # Known top-level params for the wrapper
        known = {"n_d", "n_a", "n_steps", "gamma", "cat_idxs", "cat_dims", "cat_emb_dim"}
        # Allow passing a whole kwargs dict explicitly and merge it
        explicit_kwargs = params.pop("kwargs", None)
        for k in list(params.keys()):
            if k in known:
                setattr(self, k, params.pop(k))
        # Everything else is a TabNetClassifier kwarg
        if not hasattr(self, "kwargs") or self.kwargs is None:
            self.kwargs = {}
        if explicit_kwargs is not None:
            self.kwargs.update(explicit_kwargs)
        self.kwargs.update(params)
        return self

    # --- helpers
    def _to_numpy_float32(self, X):
        if hasattr(X, "toarray"):   # scipy sparse
            X = X.toarray()
        elif hasattr(X, "to_numpy"):  # pandas
            X = X.to_numpy()
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y, **fit_params):
        X, y = self._validate_data(X, y, accept_sparse=False, y_numeric=False, multi_output=False)
        X = self._to_numpy_float32(X)
        y = np.asarray(y).ravel()

        self._model = TabNetClassifier(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps, gamma=self.gamma,
            cat_idxs=list(self.cat_idxs), cat_dims=list(self.cat_dims),
            cat_emb_dim=self.cat_emb_dim, **self.kwargs,
        )
        self._model.fit(X, y, **fit_params)

        # Keep classes_ for downstream logic
        self.classes_ = getattr(self._model, "classes_", np.unique(y))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, reset=False)
        X = self._to_numpy_float32(X)
        y_pred = self._model.predict(X)
        return np.asarray(y_pred).squeeze()

    def predict_proba(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, reset=False)
        X = self._to_numpy_float32(X)
        proba = self._model.predict_proba(X)
        return np.asarray(proba)


def run_tabnet(X, y, X_num, X_cat, preprocessor, n_candidates=N_CANDIDATES, factor=FACTOR, n_jobs=N_JOBS):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = _pipe(preprocessor, "tab", SkTabNet(device_name=device_name, seed=SEED, verbose=0))
    search = _halving_search(pipe, "tab", n_candidates=n_candidates, X=X, factor=factor, n_jobs=n_jobs)

    fit_model(search, X, y, **tabnet_params)
    return search
