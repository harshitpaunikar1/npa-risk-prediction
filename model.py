"""
NPA (Non-Performing Asset) risk prediction model for banking portfolios.
Classifies loans as standard, watchlist, substandard, doubtful, or loss using RBI NPA norms.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


NPA_CLASSES = ["standard", "watchlist", "substandard", "doubtful", "loss"]
NPA_CLASS_WEIGHTS = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}  # higher weight for worse categories


def rbi_npa_classification(days_past_due: int) -> str:
    """Classify a loan according to simplified RBI NPA norms based on days past due."""
    if days_past_due <= 30:
        return "standard"
    elif days_past_due <= 90:
        return "watchlist"
    elif days_past_due <= 365:
        return "substandard"
    elif days_past_due <= 1095:
        return "doubtful"
    else:
        return "loss"


class NPARiskModel:
    """
    Multi-class NPA classification for banking loan portfolios.
    Predicts NPA category and computes provision requirements.
    """

    PROVISION_RATES = {
        "standard": 0.0025,
        "watchlist": 0.10,
        "substandard": 0.15,
        "doubtful": 0.25,
        "loss": 1.00,
    }

    def __init__(self, numeric_features: List[str], categorical_features: List[str],
                 target_col: str = "npa_class"):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.label_encoder = LabelEncoder()
        self.models: Dict[str, Pipeline] = {}
        self.results: List[Dict] = []
        self.best_model_name: Optional[str] = None
        self._best_pipe: Optional[Pipeline] = None

    def _preprocessor(self):
        transformers = []
        if self.numeric_features:
            transformers.append(("num", StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat",
                                  OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                  self.categorical_features))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _estimators(self) -> Dict:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced",
                                                      multi_class="multinomial"),
            "RandomForest": RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                                    random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                             max_depth=4, random_state=42),
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05,
                                                   max_depth=5, random_state=42,
                                                   tree_method="hist", verbosity=0)
        return models

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        feat_cols = self.numeric_features + self.categorical_features
        df = df[feat_cols + [self.target_col]].dropna(subset=[self.target_col])
        for col in self.numeric_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")

        X = df[feat_cols]
        y_str = df[self.target_col].astype(str)
        y = self.label_encoder.fit_transform(y_str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        preprocessor = self._preprocessor()
        self.results = []

        for name, est in self._estimators().items():
            pipe = Pipeline([("preprocessor", preprocessor), ("model", est)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)
            try:
                auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))
            except Exception:
                auc = 0.0
            accuracy = float((y_pred == y_test).mean())
            self.models[name] = pipe
            self.results.append({
                "model": name,
                "accuracy": round(accuracy, 4),
                "auc_ovr": round(auc, 4),
            })

        results_df = pd.DataFrame(self.results).sort_values("auc_ovr", ascending=False).reset_index(drop=True)
        self.best_model_name = results_df.iloc[0]["model"]
        self._best_pipe = self.models[self.best_model_name]
        return results_df

    def predict_class(self, df: pd.DataFrame) -> np.ndarray:
        if self._best_pipe is None:
            raise RuntimeError("Call fit() first.")
        feat_cols = self.numeric_features + self.categorical_features
        preds = self._best_pipe.predict(df[feat_cols])
        return self.label_encoder.inverse_transform(preds)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._best_pipe is None:
            raise RuntimeError("Call fit() first.")
        feat_cols = self.numeric_features + self.categorical_features
        probs = self._best_pipe.predict_proba(df[feat_cols])
        return pd.DataFrame(probs, columns=self.label_encoder.classes_)

    def provision_requirement(self, df: pd.DataFrame,
                               outstanding_col: str = "outstanding_amount") -> pd.DataFrame:
        """Compute required provisions for each loan based on predicted NPA class."""
        df = df.copy()
        predicted = self.predict_class(df)
        df["predicted_npa_class"] = predicted
        df["provision_rate"] = df["predicted_npa_class"].map(self.PROVISION_RATES)
        df["required_provision"] = (df[outstanding_col] * df["provision_rate"]).round(2)
        return df[["predicted_npa_class", "provision_rate", "required_provision"]]

    def portfolio_summary(self, df: pd.DataFrame,
                           outstanding_col: str = "outstanding_amount") -> Dict:
        """Aggregate NPA classification and provision across the loan portfolio."""
        provisions = self.provision_requirement(df, outstanding_col)
        merged = df[[outstanding_col]].copy()
        merged["predicted_npa_class"] = provisions["predicted_npa_class"].values
        merged["required_provision"] = provisions["required_provision"].values
        total_outstanding = float(df[outstanding_col].sum())
        total_provision = float(merged["required_provision"].sum())
        npa_classes = merged[merged["predicted_npa_class"] != "standard"]
        return {
            "total_loans": len(df),
            "total_outstanding": round(total_outstanding, 0),
            "total_provision_required": round(total_provision, 0),
            "provision_coverage_pct": round(total_provision / max(total_outstanding, 1) * 100, 2),
            "npa_loans": int(len(npa_classes)),
            "npa_rate_pct": round(len(npa_classes) / max(len(df), 1) * 100, 2),
            "class_distribution": merged["predicted_npa_class"].value_counts().to_dict(),
        }


if __name__ == "__main__":
    np.random.seed(42)
    n = 3000
    days_past_due = np.random.choice(
        [0, 15, 45, 90, 180, 365, 500, 730, 1200], n,
        p=[0.50, 0.15, 0.12, 0.08, 0.06, 0.04, 0.03, 0.01, 0.01]
    )
    npa_class = [rbi_npa_classification(d) for d in days_past_due]

    df = pd.DataFrame({
        "days_past_due": days_past_due.astype(float),
        "credit_score": np.random.randint(300, 850, n).astype(float),
        "outstanding_amount": np.abs(np.random.lognormal(12, 1.0, n)),
        "loan_to_value_ratio": np.random.uniform(0.3, 1.2, n),
        "num_restructurings": np.random.poisson(0.1, n).astype(float),
        "income": np.random.lognormal(10.5, 0.7, n),
        "loan_type": np.random.choice(["home", "vehicle", "personal", "business"], n),
        "sector": np.random.choice(["retail", "msme", "agriculture", "corporate"], n),
        "npa_class": npa_class,
    })
    print(f"Class distribution:\n{pd.Series(npa_class).value_counts()}")

    model = NPARiskModel(
        numeric_features=["days_past_due", "credit_score", "outstanding_amount",
                          "loan_to_value_ratio", "num_restructurings", "income"],
        categorical_features=["loan_type", "sector"],
    )
    results = model.fit(df)
    print("\nModel comparison:")
    print(results.to_string(index=False))

    print("\nPortfolio summary:")
    summary = model.portfolio_summary(df.head(500), outstanding_col="outstanding_amount")
    for k, v in summary.items():
        print(f"  {k}: {v}")
