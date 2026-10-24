# =============================================================================
# BearingSense — Machine Learning Classifiers & Serialization
# Engineered by : SISMAX (sismax01)
# =============================================================================
"""Machine learning classifiers for bearing fault diagnosis.

Implements four interpretable, self-contained NumPy/SciPy classifiers with a unified
estimator interface (`fit`, `predict`, `predict_proba`):
  1. `PhysicsRuleThresholdClassifier`: Transparent domain baseline using kurtosis and
     envelope fault-frequency harmonic energy ratios (BPFO, BPFI, BSF).
  2. `RegularizedMultinomialLogistic`: L2-regularized multinomial logistic regression
     (softmax linear classifier) optimized via L-BFGS (`scipy.optimize.minimize`) with
     built-in z-score standardization.
  3. `GaussianDiagDiscriminant`: Regularized Gaussian Discriminant / Mahalanobis distance
     probabilistic classifier over standardized features.
  4. `RandomForestBearingClassifier`: Ensemble of CART decision trees trained with
     bootstrap aggregation, random feature subspaces, and Gini impurity splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import optimize, special

from bearingsense.features import FEATURE_NAMES
from bearingsense.simulation import FAULT_CLASSES


class BaseBearingClassifier:
    """Abstract base interface for BearingSense classifiers."""

    name: str = "BaseClassifier"

    def fit(self, X: np.ndarray, y: Sequence[str], feature_names: Optional[List[str]] = None) -> "BaseBearingClassifier":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> List[str]:
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return [self.classes_[int(i)] for i in indices]


class PhysicsRuleThresholdClassifier(BaseBearingClassifier):
    """Domain-heuristic baseline using envelope harmonic ratios and time-domain kurtosis."""

    name = "Physics-Informed Envelope Rule Baseline"

    def __init__(self) -> None:
        self.classes_: List[str] = list(FAULT_CLASSES)
        self.feature_names_: List[str] = list(FEATURE_NAMES)
        self.normal_thresholds_: Dict[str, float] = {}
        self.fault_scales_: Dict[str, Tuple[float, float]] = {}

    def fit(
        self,
        X: np.ndarray,
        y: Sequence[str],
        feature_names: Optional[List[str]] = None,
    ) -> "PhysicsRuleThresholdClassifier":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=str)
        if feature_names is not None:
            self.feature_names_ = list(feature_names)

        unique_cls = sorted(set(y_arr.tolist()), key=lambda c: FAULT_CLASSES.index(c) if c in FAULT_CLASSES else 99)
        self.classes_ = unique_cls

        idx_map = {name: i for i, name in enumerate(self.feature_names_)}
        normal_mask = y_arr == "Normal"
        X_norm = X_arr[normal_mask] if np.any(normal_mask) else X_arr

        for key in ["td_rms", "td_kurtosis", "env_bpfo_energy_ratio", "env_bpfi_energy_ratio", "env_bsf_energy_ratio"]:
            if key in idx_map:
                col = X_norm[:, idx_map[key]]
                self.normal_thresholds_[key] = float(np.percentile(col, 88.0))
            else:
                self.normal_thresholds_[key] = 0.1

        for key in self.feature_names_:
            col = X_arr[:, idx_map[key]]
            med = float(np.median(col))
            iqr = float(np.percentile(col, 75.0) - np.percentile(col, 25.0))
            self.fault_scales_[key] = (med, max(iqr, 1e-6))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        idx = {name: i for i, name in enumerate(self.feature_names_)}

        bpfo = X_arr[:, idx["env_bpfo_energy_ratio"]]
        bpfi = X_arr[:, idx["env_bpfi_energy_ratio"]]
        bsf = X_arr[:, idx["env_bsf_energy_ratio"]]
        sb = X_arr[:, idx["env_ftf_sideband_ratio"]]
        rms = X_arr[:, idx["td_rms"]]

        s_outer = bpfo / max(self.normal_thresholds_.get("env_bpfo_energy_ratio", 0.08), 1e-4)
        s_inner = (0.75 * bpfi + 0.25 * sb) / max(self.normal_thresholds_.get("env_bpfi_energy_ratio", 0.08), 1e-4)
        s_ball = (0.80 * bsf + 0.20 * sb) / max(self.normal_thresholds_.get("env_bsf_energy_ratio", 0.08), 1e-4)

        max_fault_ratio = np.maximum.reduce([s_outer, s_inner, s_ball])
        rms_ratio = rms / max(self.normal_thresholds_.get("td_rms", 0.1), 1e-4)
        s_normal = 2.2 / (0.5 * max_fault_ratio + 0.5 * rms_ratio + 0.25)

        score_map = {
            "Normal": s_normal,
            "Outer_Race_Fault": s_outer,
            "Inner_Race_Fault": s_inner,
            "Ball_Fault": s_ball,
        }
        logits = np.column_stack([score_map.get(c, np.zeros(X_arr.shape[0])) for c in self.classes_])
        return special.softmax(1.8 * logits, axis=1)


class RegularizedMultinomialLogistic(BaseBearingClassifier):
    """Standardized L2-Regularized Multinomial Logistic Regression (Softmax Classifier)."""

    name = "Regularized Multinomial Logistic Regression"

    def __init__(self, l2_penalty: float = 1.0, max_iter: int = 250) -> None:
        self.l2_penalty = float(l2_penalty)
        self.max_iter = int(max_iter)
        self.classes_: List[str] = []
        self.feature_names_: List[str] = list(FEATURE_NAMES)
        self.mean_: np.ndarray = np.array([])
        self.std_: np.ndarray = np.array([])
        self.weights_: np.ndarray = np.array([])
        self.bias_: np.ndarray = np.array([])

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(
        self,
        X: np.ndarray,
        y: Sequence[str],
        feature_names: Optional[List[str]] = None,
    ) -> "RegularizedMultinomialLogistic":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=str)
        if feature_names is not None:
            self.feature_names_ = list(feature_names)

        self.classes_ = sorted(
            set(y_arr.tolist()),
            key=lambda c: FAULT_CLASSES.index(c) if c in FAULT_CLASSES else 99,
        )
        n_samples, n_features = X_arr.shape
        n_classes = len(self.classes_)

        self.mean_ = np.mean(X_arr, axis=0)
        self.std_ = np.std(X_arr, axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        Xs = self._standardize(X_arr)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        Y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
        for i, lbl in enumerate(y_arr):
            Y_onehot[i, class_to_idx[lbl]] = 1.0

        def loss_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            W = params[: n_features * n_classes].reshape(n_features, n_classes)
            b = params[n_features * n_classes :]
            logits = Xs @ W + b
            log_probs = special.log_softmax(logits, axis=1)
            probs = np.exp(log_probs)

            ce_loss = -np.sum(Y_onehot * log_probs) / n_samples
            reg_loss = 0.5 * (self.l2_penalty / n_samples) * np.sum(W * W)
            total_loss = float(ce_loss + reg_loss)

            diff = (probs - Y_onehot) / n_samples
            grad_W = (Xs.T @ diff) + (self.l2_penalty / n_samples) * W
            grad_b = np.sum(diff, axis=0)
            grad = np.concatenate([grad_W.ravel(), grad_b])
            return total_loss, grad

        init_params = np.zeros(n_features * n_classes + n_classes, dtype=np.float64)
        res = optimize.minimize(
            loss_and_grad,
            init_params,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter},
        )
        opt_params = res.x
        self.weights_ = opt_params[: n_features * n_classes].reshape(n_features, n_classes)
        self.bias_ = opt_params[n_features * n_classes :]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        Xs = self._standardize(X_arr)
        logits = Xs @ self.weights_ + self.bias_
        return special.softmax(logits, axis=1)


class GaussianDiagDiscriminant(BaseBearingClassifier):
    """Regularized Gaussian Discriminant Classifier (Quadratic/Diagonal Bayes)."""

    name = "Gaussian Discriminant Classifier"

    def __init__(self, reg_var: float = 0.15) -> None:
        self.reg_var = float(reg_var)
        self.classes_: List[str] = []
        self.feature_names_: List[str] = list(FEATURE_NAMES)
        self.mean_: np.ndarray = np.array([])
        self.std_: np.ndarray = np.array([])
        self.class_means_: np.ndarray = np.array([])
        self.class_vars_: np.ndarray = np.array([])
        self.class_priors_: np.ndarray = np.array([])

    def fit(
        self,
        X: np.ndarray,
        y: Sequence[str],
        feature_names: Optional[List[str]] = None,
    ) -> "GaussianDiagDiscriminant":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=str)
        if feature_names is not None:
            self.feature_names_ = list(feature_names)

        self.classes_ = sorted(
            set(y_arr.tolist()),
            key=lambda c: FAULT_CLASSES.index(c) if c in FAULT_CLASSES else 99,
        )
        self.mean_ = np.mean(X_arr, axis=0)
        self.std_ = np.std(X_arr, axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        Xs = (X_arr - self.mean_) / self.std_

        n_classes = len(self.classes_)
        n_features = Xs.shape[1]
        self.class_means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.class_vars_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.class_priors_ = np.zeros(n_classes, dtype=np.float64)

        for i, cls_name in enumerate(self.classes_):
            mask = y_arr == cls_name
            X_c = Xs[mask]
            self.class_means_[i] = np.mean(X_c, axis=0)
            self.class_vars_[i] = np.var(X_c, axis=0) + self.reg_var
            self.class_priors_[i] = float(np.mean(mask))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        Xs = (X_arr - self.mean_) / self.std_
        n_samples = Xs.shape[0]
        n_classes = len(self.classes_)
        log_likes = np.zeros((n_samples, n_classes), dtype=np.float64)

        for i in range(n_classes):
            diff = Xs - self.class_means_[i]
            var = self.class_vars_[i]
            log_det = 0.5 * np.sum(np.log(2.0 * np.pi * var))
            mahal = 0.5 * np.sum((diff * diff) / var, axis=1)
            log_likes[:, i] = np.log(max(self.class_priors_[i], 1e-9)) - log_det - mahal

        return special.softmax(log_likes, axis=1)


@dataclass
class _TreeNode:
    """Internal node structure for CART decision trees."""

    feature_idx: int = -1
    threshold: float = 0.0
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    probs: Optional[np.ndarray] = None
    impurity_gain: float = 0.0
    n_samples: int = 0


class RandomForestBearingClassifier(BaseBearingClassifier):
    """Ensemble of CART decision trees with bootstrap aggregation and random subspaces."""

    name = "Random Forest Classifier"

    def __init__(
        self,
        n_estimators: int = 60,
        max_depth: int = 6,
        min_samples_split: int = 4,
        max_features: str | int = "sqrt",
        seed: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.seed = int(seed)
        self.classes_: List[str] = []
        self.feature_names_: List[str] = list(FEATURE_NAMES)
        self.trees_: List[_TreeNode] = []
        self.feature_importances_: np.ndarray = np.array([])

    @staticmethod
    def _gini(counts: np.ndarray, total: int) -> float:
        if total <= 0:
            return 0.0
        p = counts / total
        return float(1.0 - np.sum(p * p))

    def _build_tree(
        self,
        X: np.ndarray,
        y_int: np.ndarray,
        n_classes: int,
        depth: int,
        rng: np.random.Generator,
        importance_accum: np.ndarray,
    ) -> _TreeNode:
        n_samples, n_features = X.shape
        counts = np.bincount(y_int, minlength=n_classes).astype(np.float64)
        probs = (counts + 0.05) / (n_samples + 0.05 * n_classes)
        parent_gini = self._gini(counts, n_samples)

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or parent_gini <= 1e-6
        ):
            return _TreeNode(probs=probs, n_samples=n_samples)

        if self.max_features == "sqrt":
            k_feat = max(1, int(np.sqrt(n_features)))
        elif isinstance(self.max_features, int):
            k_feat = min(n_features, max(1, self.max_features))
        else:
            k_feat = n_features

        feat_subset = rng.choice(n_features, size=k_feat, replace=False)

        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0

        for f_idx in feat_subset:
            col = X[:, f_idx]
            f_min, f_max = float(np.min(col)), float(np.max(col))
            if f_max - f_min <= 1e-12:
                continue

            quantiles = np.linspace(0.1, 0.9, num=9)
            thresholds = np.unique(np.quantile(col, quantiles))
            for thresh in thresholds:
                left_mask = col <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                if n_left == 0 or n_right == 0:
                    continue

                left_counts = np.bincount(y_int[left_mask], minlength=n_classes)
                right_counts = counts - left_counts
                gini_left = self._gini(left_counts, n_left)
                gini_right = self._gini(right_counts, n_right)

                weighted_child_gini = (n_left * gini_left + n_right * gini_right) / n_samples
                gain = parent_gini - weighted_child_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f_idx)
                    best_thresh = float(thresh)

        if best_feat == -1 or best_gain <= 1e-7:
            return _TreeNode(probs=probs, n_samples=n_samples)

        importance_accum[best_feat] += best_gain * n_samples
        left_mask = X[:, best_feat] <= best_thresh
        left_node = self._build_tree(
            X[left_mask], y_int[left_mask], n_classes, depth + 1, rng, importance_accum
        )
        right_node = self._build_tree(
            X[~left_mask], y_int[~left_mask], n_classes, depth + 1, rng, importance_accum
        )
        return _TreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_node,
            right=right_node,
            probs=probs,
            impurity_gain=best_gain,
            n_samples=n_samples,
        )

    def fit(
        self,
        X: np.ndarray,
        y: Sequence[str],
        feature_names: Optional[List[str]] = None,
    ) -> "RandomForestBearingClassifier":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=str)
        if feature_names is not None:
            self.feature_names_ = list(feature_names)

        self.classes_ = sorted(
            set(y_arr.tolist()),
            key=lambda c: FAULT_CLASSES.index(c) if c in FAULT_CLASSES else 99,
        )
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([class_to_idx[c] for c in y_arr], dtype=int)

        n_samples, n_features = X_arr.shape
        n_classes = len(self.classes_)
        rng = np.random.default_rng(self.seed)

        self.trees_ = []
        importance_accum = np.zeros(n_features, dtype=np.float64)

        for _ in range(self.n_estimators):
            boot_indices = rng.integers(0, n_samples, size=n_samples)
            tree = self._build_tree(
                X_arr[boot_indices],
                y_int[boot_indices],
                n_classes=n_classes,
                depth=0,
                rng=rng,
                importance_accum=importance_accum,
            )
            self.trees_.append(tree)

        tot_imp = float(np.sum(importance_accum))
        if tot_imp > 0:
            self.feature_importances_ = importance_accum / tot_imp
        else:
            self.feature_importances_ = np.ones(n_features, dtype=np.float64) / n_features

        return self

    def _predict_tree_single(self, node: _TreeNode, x_row: np.ndarray) -> np.ndarray:
        curr = node
        while curr.left is not None and curr.right is not None and curr.feature_idx >= 0:
            if x_row[curr.feature_idx] <= curr.threshold:
                curr = curr.left
            else:
                curr = curr.right
        return curr.probs  # type: ignore[return-value]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        n_samples = X_arr.shape[0]
        n_classes = len(self.classes_)
        all_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

        for tree in self.trees_:
            for i in range(n_samples):
                all_probs[i] += self._predict_tree_single(tree, X_arr[i])

        all_probs /= max(len(self.trees_), 1)
        return all_probs

    def to_dict(self) -> Dict:
        """Serialize trained Random Forest + reference statistics to a JSON-compatible dict."""

        def node_to_dict(n: _TreeNode) -> Dict:
            if n.left is None or n.right is None or n.feature_idx < 0:
                return {"probs": n.probs.tolist() if n.probs is not None else []}
            return {
                "f": n.feature_idx,
                "t": n.threshold,
                "l": node_to_dict(n.left),
                "r": node_to_dict(n.right),
                "probs": n.probs.tolist() if n.probs is not None else [],
            }

        return {
            "model_type": "RandomForestBearingClassifier",
            "author_signature": "SISMAX (sismax01)",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "seed": self.seed,
            "classes": self.classes_,
            "feature_names": self.feature_names_,
            "feature_importances": self.feature_importances_.tolist(),
            "trees": [node_to_dict(t) for t in self.trees_],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RandomForestBearingClassifier":
        """Deserialize a RandomForestBearingClassifier from a dictionary."""

        def dict_to_node(d: Dict) -> _TreeNode:
            probs = np.array(d.get("probs", []), dtype=np.float64)
            if "f" not in d:
                return _TreeNode(probs=probs)
            return _TreeNode(
                feature_idx=int(d["f"]),
                threshold=float(d["t"]),
                left=dict_to_node(d["l"]),
                right=dict_to_node(d["r"]),
                probs=probs,
            )

        model = cls(
            n_estimators=int(data["n_estimators"]),
            max_depth=int(data["max_depth"]),
            min_samples_split=int(data["min_samples_split"]),
            seed=int(data["seed"]),
        )
        model.classes_ = list(data["classes"])
        model.feature_names_ = list(data["feature_names"])
        model.feature_importances_ = np.array(data["feature_importances"], dtype=np.float64)
        model.trees_ = [dict_to_node(t) for t in data["trees"]]
        return model


def save_model_bundle(
    model: RandomForestBearingClassifier,
    train_df_features: np.ndarray,
    train_labels: Sequence[str],
    output_path: str | Path,
    metadata: Optional[Dict] = None,
) -> Path:
    """Save a trained Random Forest model along with class feature statistics for explainability."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    X_arr = np.asarray(train_df_features, dtype=np.float64)
    y_arr = np.asarray(train_labels, dtype=str)
    normal_mask = y_arr == "Normal"
    X_norm = X_arr[normal_mask] if np.any(normal_mask) else X_arr

    bundle = {
        "model": model.to_dict(),
        "background_mean": np.mean(X_arr, axis=0).tolist(),
        "background_std": np.maximum(np.std(X_arr, axis=0), 1e-9).tolist(),
        "normal_baseline_mean": np.mean(X_norm, axis=0).tolist(),
        "normal_baseline_std": np.maximum(np.std(X_norm, axis=0), 1e-9).tolist(),
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return path


def load_model_bundle(bundle_path: str | Path) -> Tuple[RandomForestBearingClassifier, Dict]:
    """Load a serialized model bundle from disk."""
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    model = RandomForestBearingClassifier.from_dict(data["model"])
    return model, data
