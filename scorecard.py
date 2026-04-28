"""
NPA risk scorecard with Weight of Evidence (WoE) and Information Value (IV) analysis.
Generates regulatory-friendly scorecards for loan risk assessment.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class WoEBinner:
    """
    Bins continuous and categorical features using WoE encoding.
    Computes Information Value (IV) per feature.
    """

    def __init__(self, n_bins: int = 10, min_bin_pct: float = 0.05):
        self.n_bins = n_bins
        self.min_bin_pct = min_bin_pct
        self._woe_maps: Dict[str, Dict] = {}
        self._iv_values: Dict[str, float] = {}

    def _compute_woe_iv(self, df: pd.DataFrame, feature: str,
                         target: str) -> Tuple[Dict, float]:
        total_events = df[target].sum()
        total_non_events = len(df) - total_events
        eps = 1e-7
        bins = df.groupby(feature, observed=True)[target].agg(["sum", "count"]).reset_index()
        bins.columns = [feature, "events", "total"]
        bins["non_events"] = bins["total"] - bins["events"]
        bins["pct_events"] = (bins["events"] / max(total_events, 1)).clip(eps)
        bins["pct_non_events"] = (bins["non_events"] / max(total_non_events, 1)).clip(eps)
        bins["woe"] = np.log(bins["pct_events"] / bins["pct_non_events"])
        bins["iv_component"] = (bins["pct_events"] - bins["pct_non_events"]) * bins["woe"]
        woe_map = dict(zip(bins[feature].astype(str), bins["woe"].round(4)))
        iv = round(float(bins["iv_component"].sum()), 4)
        return woe_map, iv

    def fit_transform(self, df: pd.DataFrame, features: List[str],
                       target: str) -> pd.DataFrame:
        result = df.copy()
        for feat in features:
            col = df[feat]
            if pd.api.types.is_numeric_dtype(col):
                try:
                    result[f"{feat}_bin"] = pd.qcut(
                        col, q=self.n_bins, duplicates="drop", labels=False
                    ).astype(str)
                    woe_map, iv = self._compute_woe_iv(result, f"{feat}_bin", target)
                    result[f"{feat}_woe"] = result[f"{feat}_bin"].map(woe_map).fillna(0)
                except Exception:
                    result[f"{feat}_woe"] = 0.0
                    woe_map, iv = {}, 0.0
                self._woe_maps[feat] = woe_map
                self._iv_values[feat] = iv
            else:
                result[f"{feat}_str"] = col.astype(str)
                woe_map, iv = self._compute_woe_iv(result, f"{feat}_str", target)
                result[f"{feat}_woe"] = result[f"{feat}_str"].map(woe_map).fillna(0)
                self._woe_maps[feat] = woe_map
                self._iv_values[feat] = iv
        return result

    def iv_summary(self) -> pd.DataFrame:
        def interpret(iv: float) -> str:
            if iv < 0.02:
                return "useless"
            elif iv < 0.1:
                return "weak"
            elif iv < 0.3:
                return "medium"
            elif iv < 0.5:
                return "strong"
            else:
                return "very strong"

        records = [
            {"feature": f, "iv": v, "predictive_power": interpret(v)}
            for f, v in self._iv_values.items()
        ]
        return pd.DataFrame(records).sort_values("iv", ascending=False).reset_index(drop=True)


class CreditScorecard:
    """
    Logistic regression-based credit scorecard with point-scoring system.
    Converts log-odds to a 300-900 credit score scale.
    """

    def __init__(self, pdo: int = 20, base_score: int = 600, base_odds: float = 50.0):
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self._factor = pdo / np.log(2)
        self._offset = base_score - self._factor * np.log(base_odds)
        self._model = None
        self._feature_cols: List[str] = []

    def fit(self, df: pd.DataFrame, woe_features: List[str], target: str) -> "CreditScorecard":
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise RuntimeError("scikit-learn required for scorecard fitting.")
        self._feature_cols = woe_features
        X = df[woe_features].fillna(0)
        y = df[target]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._model = LogisticRegression(max_iter=500, C=1.0)
        self._model.fit(X_scaled, y)
        self._scaler = scaler
        return self

    def log_odds_to_score(self, log_odds: float) -> int:
        return int(np.clip(self._offset + self._factor * log_odds, 300, 900))

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        X = df[self._feature_cols].fillna(0)
        X_scaled = self._scaler.transform(X)
        probs = self._model.predict_proba(X_scaled)[:, 1]
        eps = 1e-7
        log_odds = np.log((1 - probs + eps) / (probs + eps))
        return np.array([self.log_odds_to_score(lo) for lo in log_odds])

    def score_to_risk_grade(self, score: int) -> str:
        if score >= 800:
            return "AAA"
        elif score >= 750:
            return "AA"
        elif score >= 700:
            return "A"
        elif score >= 650:
            return "BBB"
        elif score >= 600:
            return "BB"
        elif score >= 550:
            return "B"
        elif score >= 500:
            return "CCC"
        else:
            return "D"

    def scorecard_table(self) -> Optional[pd.DataFrame]:
        """Return feature coefficients as a scorecard table with score points."""
        if self._model is None:
            return None
        coefs = self._model.coef_[0]
        points = (coefs * self._factor).round(0).astype(int)
        return pd.DataFrame({
            "feature": self._feature_cols,
            "coefficient": coefs.round(4),
            "score_points": points,
        }).sort_values("score_points", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    np.random.seed(42)
    n = 2000
    credit_score = np.random.randint(300, 850, n)
    income = np.random.lognormal(10.5, 0.7, n)
    loan_to_value = np.random.uniform(0.3, 1.2, n)
    days_past_due = np.random.choice([0, 30, 60, 90, 180], n, p=[0.55, 0.2, 0.1, 0.08, 0.07])
    employment_type = np.random.choice(["salaried", "self_employed", "business"], n)
    p_npa = 1 / (1 + np.exp(0.01 * credit_score - 0.5 * loan_to_value + 0.01 * days_past_due - 5))
    is_npa = (np.random.rand(n) < p_npa).astype(int)

    df = pd.DataFrame({
        "credit_score": credit_score.astype(float),
        "income": income,
        "loan_to_value": loan_to_value,
        "days_past_due": days_past_due.astype(float),
        "employment_type": employment_type,
        "is_npa": is_npa,
    })
    print(f"NPA rate: {is_npa.mean():.2%}")

    binner = WoEBinner(n_bins=5)
    df_woe = binner.fit_transform(
        df, features=["credit_score", "income", "loan_to_value", "days_past_due", "employment_type"],
        target="is_npa"
    )
    print("\nInformation Value summary:")
    print(binner.iv_summary().to_string(index=False))

    woe_features = ["credit_score_woe", "income_woe", "loan_to_value_woe",
                    "days_past_due_woe", "employment_type_woe"]
    scorecard = CreditScorecard(pdo=20, base_score=600, base_odds=50)
    scorecard.fit(df_woe, woe_features=woe_features, target="is_npa")

    scores = scorecard.predict_scores(df_woe.head(10))
    grades = [scorecard.score_to_risk_grade(s) for s in scores]
    print("\nSample scores and risk grades:")
    for score, grade in zip(scores, grades):
        print(f"  Score: {score} | Grade: {grade}")

    print("\nScorecard table:")
    table = scorecard.scorecard_table()
    if table is not None:
        print(table.to_string(index=False))
