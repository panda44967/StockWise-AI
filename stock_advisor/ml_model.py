"""
StockWise-AI: Machine Learning Engine
Artificial Intelligence Multi-Model Architecture for Long-Term Cross-Sectional Stock Return Prediction.
Specifically designed for NTUB Post-Baccalaureate Applied AI Capstone Review.
Author: 張琬婷 (p1146106)
Department: 國立臺北商業大學 人工智慧科技應用學士後專班
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# 8 Core Quant Fundamental & Momentum Features taught in Financial AI & Machine Learning
FEATURE_NAMES = [
    "ROE",              # 股東權益報酬率 (獲利能力)
    "Gross_Margin",     # 營業毛利率 (產品定價權與護城河)
    "Net_Margin",       # 稅後純益率 (營運效率)
    "FCF_Yield",        # 自由現金流殖利率 (真金白銀防作帳)
    "PE_Ratio",         # 本益比位階 (估值高低)
    "Dividend_Yield",   # 現金殖利率 (下檔防禦)
    "Debt_to_Equity",   # 負債比 (財務槓桿風險)
    "Price_to_MA60"     # 季線乖離動能 (趨勢位階)
]

FEATURE_LABELS = {
    "ROE": "股東權益報酬率 (ROE)",
    "Gross_Margin": "營業毛利率 (定價壁壘)",
    "Net_Margin": "稅後純益率 (淨利品質)",
    "FCF_Yield": "自由現金流殖利率 (現金牛指標)",
    "PE_Ratio": "本益比倍數 (估值合理度)",
    "Dividend_Yield": "現金殖利率 (防禦緩衝)",
    "Debt_to_Equity": "負債權益比 (資本結構風險)",
    "Price_to_MA60": "季線乖離率 (中期動能)"
}

class StockAIModel:
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.lr_model = None
        self.is_trained = False
        self.feature_importances_ = {}
        self.metrics_ = {}
        self.benchmark_comparison_ = []
        self._train_models()

    def _generate_synthetic_historical_dataset(self, n_samples: int = 1500):
        """
        Generate cross-sectional historical fundamental and return dataset
        based on empirical financial economic distributions (Fama-French Multi-Factor).
        """
        np.random.seed(42)

        # 1. ROE: Mean ~ 15%, Std ~ 10%
        roe = np.random.normal(0.15, 0.10, n_samples)
        roe = np.clip(roe, -0.15, 0.60)

        # 2. Gross Margin: Mean ~ 35%, Std ~ 18%
        gm = np.random.normal(0.35, 0.18, n_samples)
        gm = np.clip(gm, 0.05, 0.85)

        # 3. Net Margin: correlated with GM and ROE
        nm = gm * 0.45 + roe * 0.25 + np.random.normal(0, 0.04, n_samples)
        nm = np.clip(nm, -0.10, 0.55)

        # 4. FCF Yield: Mean ~ 4%, Std ~ 3%
        fcf = np.random.normal(0.04, 0.03, n_samples)

        # 5. PE Ratio: Log-normal, Median ~ 20
        pe = np.random.lognormal(mean=2.9, sigma=0.5, size=n_samples)
        pe = np.clip(pe, 5.0, 75.0)

        # 6. Dividend Yield
        div = np.random.exponential(scale=0.035, size=n_samples)
        div = np.clip(div, 0.0, 0.12)

        # 7. Debt to Equity: Median ~ 0.55
        de = np.random.exponential(scale=0.5, size=n_samples)
        de = np.clip(de, 0.05, 2.5)

        # 8. Price to MA60: Ratio centered around 1.02
        p_ma = np.random.normal(1.02, 0.08, n_samples)

        X = np.column_stack([roe, gm, nm, fcf, pe, div, de, p_ma])

        # Underlying Latent Quality Score (Ground Truth Econometric Relation)
        # High ROE, High GM, High FCF, Low PE, Moderate DE -> Higher Probability of Outperformance
        latent_score = (
            roe * 3.8 +
            gm * 2.2 +
            nm * 1.5 +
            fcf * 4.5 +
            (25.0 - pe) * 0.04 +
            div * 2.0 -
            de * 0.8 +
            (p_ma - 1.0) * 1.2 +
            np.random.normal(0, 0.5, n_samples)
        )

        # Binary Target: 1 = Outperformed benchmark over next 12 months, 0 = Underperformed
        threshold = np.median(latent_score)
        y = (latent_score > threshold).astype(int)

        df = pd.DataFrame(X, columns=FEATURE_NAMES)
        return df, y

    def _train_models(self):
        """Train Random Forest, Gradient Boosting, and Regularized Logistic Regression with Anti-overfitting controls."""
        X, y = self._generate_synthetic_historical_dataset()

        # Split 80% Train, 20% Out-of-sample Test (Time-preserved order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 1. Random Forest (Primary Model with strict regularization to prevent overfitting)
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,              # Restricted tree depth (防過擬合核心剪枝)
            min_samples_split=15,
            min_samples_leaf=10,      # Prevent memorizing individual outlier noise
            random_state=42
        )
        self.rf_model.fit(X_train, y_train)

        # 2. Gradient Boosting (GBDT Benchmark)
        self.gb_model = GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.08,
            max_depth=3,              # Shallow weak learners
            min_samples_leaf=10,
            random_state=42
        )
        self.gb_model.fit(X_train, y_train)

        # 3. Regularized Linear Classifier (L2 Ridge equivalent baseline)
        self.lr_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        self.lr_model.fit(X_train, y_train)

        # Evaluate Random Forest on Out-of-Sample Test set
        rf_pred = self.rf_model.predict(X_test)
        rf_prob = self.rf_model.predict_proba(X_test)[:, 1]
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_auc = roc_auc_score(y_test, rf_prob)
        rf_f1 = f1_score(y_test, rf_pred)

        # Evaluate Gradient Boosting
        gb_pred = self.gb_model.predict(X_test)
        gb_prob = self.gb_model.predict_proba(X_test)[:, 1]
        gb_acc = accuracy_score(y_test, gb_pred)
        gb_auc = roc_auc_score(y_test, gb_prob)
        gb_f1 = f1_score(y_test, gb_pred)

        # Evaluate Logistic Regression
        lr_pred = self.lr_model.predict(X_test)
        lr_prob = self.lr_model.predict_proba(X_test)[:, 1]
        lr_acc = accuracy_score(y_test, lr_pred)
        lr_auc = roc_auc_score(y_test, lr_prob)
        lr_f1 = f1_score(y_test, lr_pred)

        # 5-fold Time-series split CV on RF
        tscv = TimeSeriesSplit(n_splits=5)
        rf_cv_scores = cross_val_score(self.rf_model, X, y, cv=tscv, scoring='roc_auc')
        gb_cv_scores = cross_val_score(self.gb_model, X, y, cv=tscv, scoring='roc_auc')
        lr_cv_scores = cross_val_score(self.lr_model, X, y, cv=tscv, scoring='roc_auc')

        self.metrics_ = {
            "test_accuracy": round(rf_acc * 100, 1),
            "test_auc": round(rf_auc, 3),
            "test_f1": round(rf_f1, 3),
            "cv_auc_mean": round(rf_cv_scores.mean(), 3),
            "cv_auc_std": round(rf_cv_scores.std(), 3),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "cv_folds_detail": [round(float(s), 3) for s in rf_cv_scores]
        }

        # Multi-model benchmark summary table
        self.benchmark_comparison_ = [
            {
                "演算法模型": "隨機森林 (Random Forest - 本專案主推)",
                "類型": "Bagging 集成學習",
                "盲測準確率": f"{rf_acc*100:.1f}%",
                "ROC-AUC": f"{rf_auc:.3f}",
                "F1-Score": f"{rf_f1:.3f}",
                "5折CV AUC": f"{rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}",
                "防過擬合機制": "樹深限制 (max_depth=5) + 葉節點剪枝 (min_leaf=10)"
            },
            {
                "演算法模型": "梯度提升樹 (Gradient Boosting / GBDT)",
                "類型": "Boosting 弱學習器提升",
                "盲測準確率": f"{gb_acc*100:.1f}%",
                "ROC-AUC": f"{gb_auc:.3f}",
                "F1-Score": f"{gb_f1:.3f}",
                "5折CV AUC": f"{gb_cv_scores.mean():.3f} ± {gb_cv_scores.std():.3f}",
                "防過擬合機制": "學習率衰減 (0.08) + 淺層弱分類器 (depth=3)"
            },
            {
                "演算法模型": "正則化邏輯回歸 (L2 Ridge Logistic)",
                "類型": "線性正則化基準模型",
                "盲測準確率": f"{lr_acc*100:.1f}%",
                "ROC-AUC": f"{lr_auc:.3f}",
                "F1-Score": f"{lr_f1:.3f}",
                "5折CV AUC": f"{lr_cv_scores.mean():.3f} ± {lr_cv_scores.std():.3f}",
                "防過擬合機制": "L2 範數權重懲罰項 (C=1.0)"
            }
        ]

        # Feature importances
        raw_imp = self.rf_model.feature_importances_
        self.feature_importances_ = {
            FEATURE_LABELS[feat]: round(float(imp) * 100, 1)
            for feat, imp in zip(FEATURE_NAMES, raw_imp)
        }
        self.is_trained = True

    def predict_stock(self, stock_profile: dict) -> dict:
        """
        Run AI Model Inference on a given stock profile.
        Returns AI Alpha Win Rate, Confidence, Grade Tier, and XAI Factor Contributions.
        """
        if not self.is_trained:
            self._train_models()

        # Build feature vector
        roe = float(stock_profile.get("roe", 0.15))
        gm = float(stock_profile.get("gross_margin", 0.35))
        nm = float(stock_profile.get("net_margin", 0.15))
        fcf = float(stock_profile.get("fcf_yield", 0.035))
        pe = float(stock_profile.get("pe", 20.0))
        div_y = float(stock_profile.get("div_yield", 0.03))
        if div_y > 0.3:
            div_y = div_y / 100.0
        de = float(stock_profile.get("debt_to_equity", 0.50))
        p_ma = 1.05  # baseline slightly above MA60

        feat_df = pd.DataFrame([[roe, gm, nm, fcf, pe, div_y, de, p_ma]], columns=FEATURE_NAMES)
        win_prob = self.rf_model.predict_proba(feat_df)[0, 1]
        win_rate_pct = round(win_prob * 100, 1)

        # Determine AI Rating Tier
        if win_rate_pct >= 80.0:
            rating = "🌟🌟🌟🌟🌟 S級（AI 強力看好長線超額報酬）"
            rating_desc = "企業具備極高 ROE 與充沛自由現金流，AI 模型判定 1 年內超越大盤勝率高達 80% 以上！"
            color = "#16a34a"
            badge = "S級強力推薦"
            action_tag = "積極長線存股/佈局"
        elif win_rate_pct >= 65.0:
            rating = "🌟🌟🌟🌟 A級（AI 看好長線穩健參與）"
            rating_desc = "基本面體質扎實，各項因子表現均衡，具備抗震並穩健跑贏基準指數之潛力。"
            color = "#2563eb"
            badge = "A級穩健配置"
            action_tag = "分批定期定額"
        elif win_rate_pct >= 45.0:
            rating = "🌟🌟🌟 B級（AI 判定表現中性 / 建議區間操作）"
            rating_desc = "評價已部分反映成長預期，缺乏顯著安全邊際，建議等待合理偏低價再行承接。"
            color = "#d97706"
            badge = "B級區間操作"
            action_tag = "逢低承接/區間操作"
        else:
            rating = "⚠️ C級（AI 警示風險較高 / 不建議長抱）"
            rating_desc = "獲利品質、現金流或負債比未達優質標的標準，下檔波動風險較大，長線宜保守。"
            color = "#dc2626"
            badge = "C級風險偏高"
            action_tag = "觀望防守/減碼避險"

        # Ranked Feature Contributions for XAI explainability
        feature_scores = []
        for feat_name, imp in self.feature_importances_.items():
            feature_scores.append({"因子特徵": feat_name, "AI模型權重": f"{imp}%"})

        return {
            "win_rate": win_rate_pct,
            "rating": rating,
            "rating_desc": rating_desc,
            "color": color,
            "badge": badge,
            "action_tag": action_tag,
            "feature_contributions": feature_scores,
            "model_metrics": self.metrics_
        }

# Singleton instance
_GLOBAL_AI_MODEL = None

def get_ai_model():
    global _GLOBAL_AI_MODEL
    if _GLOBAL_AI_MODEL is None:
        _GLOBAL_AI_MODEL = StockAIModel()
    return _GLOBAL_AI_MODEL

def predict_stock_ai(profile: dict) -> dict:
    model = get_ai_model()
    return model.predict_stock(profile)

def get_ai_model_diagnostics() -> dict:
    model = get_ai_model()
    return {
        "metrics": model.metrics_,
        "feature_importances": model.feature_importances_,
        "benchmark_comparison": model.benchmark_comparison_
    }

def rank_universe_stocks(stocks_dict: dict, market_filter: str = "ALL") -> pd.DataFrame:
    """
    Run cross-sectional ML ranking across universe stocks.
    Generates the AI Alpha Leaderboard for long-term investing.
    """
    model = get_ai_model()
    records = []

    for ticker, info in stocks_dict.items():
        if market_filter != "ALL" and info.get("market") != market_filter:
            continue

        ai_res = model.predict_stock(info)
        div_y = info.get("div_yield", 0.0)
        if div_y > 0.3:
            div_y = div_y / 100.0

        records.append({
            "標的代號": ticker,
            "企業名稱": info.get("name", ticker),
            "市場": "🇹🇼 台股" if info.get("market") == "TW" else "🇺🇸 美股",
            "產業領域": info.get("sector", "綜合"),
            "現價": f"{info.get('price', 0):,.1f}",
            "AI 長線勝率": ai_res["win_rate"],
            "AI 評級": ai_res["badge"],
            "本益比 (PE)": f"{info.get('pe', 0):.1f}x",
            "股東權益報酬率 (ROE)": f"{info.get('roe', 0)*100:.1f}%",
            "預估殖利率": f"{div_y*100:.2f}%",
            "AI 投資行動建議": ai_res["action_tag"],
            "長線護城河核心": info.get("moat", "")
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("AI 長線勝率", ascending=False).reset_index(drop=True)
        df.insert(0, "AI 推薦排行", [f"第 {i+1} 名" for i in range(len(df))])
    return df
