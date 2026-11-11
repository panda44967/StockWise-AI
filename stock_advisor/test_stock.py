"""
StockWise-AI Unit Tests (Built with standard library unittest - meets G2 criteria)
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stock_advisor.engine import (
    get_stock_profile, get_historical_candles, evaluate_investment_status,
    get_high_dividend_stocks, get_industry_recommendations,
    simulate_overfitting_comparison, is_etf_ticker, get_etf_profile,
    get_all_etfs, MARKET_DATA, ETF_DATA
)
from stock_advisor.ml_model import predict_stock_ai, get_ai_model_diagnostics, rank_universe_stocks

class TestStockWise(unittest.TestCase):
    def test_machine_learning_model(self):
        # Test ML diagnostics
        diag = get_ai_model_diagnostics()
        metrics = diag["metrics"]
        self.assertGreater(metrics["test_accuracy"], 65.0)
        self.assertGreater(metrics["test_auc"], 0.75)
        self.assertIn("ROE", str(diag["feature_importances"]))
        self.assertEqual(len(diag["benchmark_comparison"]), 3)

        # Test ML prediction on stock
        dummy_profile = {
            "roe": 0.40, "gross_margin": 0.60, "net_margin": 0.35,
            "fcf_yield": 0.05, "pe": 22.0, "div_yield": 0.02, "debt_to_equity": 0.25
        }
        pred = predict_stock_ai(dummy_profile)
        self.assertGreater(pred["win_rate"], 50.0)
        self.assertIn("S級", pred["rating"])
        self.assertGreater(len(pred["feature_contributions"]), 0)

        # Test ML Cross-Sectional Ranking
        df_rank = rank_universe_stocks(MARKET_DATA, market_filter="ALL")
        self.assertGreater(len(df_rank), 5)
        self.assertIn("AI 長線勝率", df_rank.columns)
        self.assertIn("AI 評級", df_rank.columns)
        self.assertIn("AI 推薦排行", df_rank.columns)

    def test_etf_detection_and_profile(self):
        # Test ETF detection
        self.assertTrue(is_etf_ticker("0050.TW"))
        self.assertTrue(is_etf_ticker("00878.TW"))
        self.assertTrue(is_etf_ticker("VOO"))
        self.assertTrue(is_etf_ticker("QQQ"))
        self.assertFalse(is_etf_ticker("2330.TW"))
        self.assertFalse(is_etf_ticker("AAPL"))

        # Test ETF profile computation
        etf_0050 = get_etf_profile("0050.TW")
        self.assertEqual(etf_0050["market"], "TW")
        self.assertGreater(etf_0050["price"], 0)
        self.assertGreater(etf_0050["nav"], 0)
        self.assertIn("premium_pct", etf_0050)
        self.assertIn("expense_ratio", etf_0050)
        self.assertGreater(len(etf_0050["top_holdings"]), 0)

        # Test US ETF VOO
        voo = get_etf_profile("VOO")
        self.assertEqual(voo["market"], "US")
        self.assertGreater(voo["price"], 0)
        self.assertIn("0.03%", voo["expense_ratio"])

        # Test ETF list fetching
        tw_etfs = get_all_etfs("TW")
        self.assertGreater(len(tw_etfs), 3)
        us_etfs = get_all_etfs("US")
        self.assertGreater(len(us_etfs), 3)

    def test_stock_profile_loading(self):
        # Test TSMC
        profile = get_stock_profile("2330.TW")
        self.assertEqual(profile["market"], "TW")
        self.assertGreater(profile["price"], 0)
        self.assertGreater(profile["eps"], 0)
        self.assertGreater(profile["roe"], 0)
        self.assertIn("moat", profile)

        # Test US stock Apple
        aapl = get_stock_profile("AAPL")
        self.assertEqual(aapl["market"], "US")
        self.assertGreater(aapl["pe"], 0)

    def test_historical_candles(self):
        df = get_historical_candles("2330.TW", period="6mo")
        self.assertGreater(len(df), 20)
        self.assertIn("Open", df.columns)
        self.assertIn("Close", df.columns)
        self.assertIn("MA20", df.columns)
        self.assertIn("MA60", df.columns)

    def test_valuation_status_and_actions(self):
        # Case 1: Undervalued dummy stock
        cheap_stock = {
            "price": 80.0, "cheap_price": 90.0, "fair_value": 100.0, "expensive_price": 120.0,
            "roe": 0.25, "gross_margin": 0.50, "pe": 12.0, "eps": 8.0, "fcf_yield": 0.06
        }
        status_cheap = evaluate_investment_status(cheap_stock)
        self.assertIn("便宜價", status_cheap["val_status"])
        self.assertIn("買進", status_cheap["action"])

        # Case 2: Overvalued dummy stock
        exp_stock = {
            "price": 150.0, "cheap_price": 90.0, "fair_value": 100.0, "expensive_price": 120.0,
            "roe": 0.10, "gross_margin": 0.15, "pe": 45.0, "eps": 3.0, "fcf_yield": 0.01
        }
        status_exp = evaluate_investment_status(exp_stock)
        self.assertIn("昂貴價", status_exp["val_status"])
        self.assertIn("減碼", status_exp["action"])

    def test_high_dividend_screener(self):
        div_list = get_high_dividend_stocks()
        self.assertGreater(len(div_list), 0)
        first_pick = div_list[0]
        self.assertIn("預估殖利率", first_pick)
        self.assertIn("連續配息年數", first_pick)
        self.assertIn("存股評等", first_pick)

    def test_industry_recommendations(self):
        recs_tw = get_industry_recommendations("TW")
        self.assertIn("半導體/先進製程", recs_tw)
        self.assertIn("金融保險/官股金控", recs_tw)

        recs_us = get_industry_recommendations("US")
        self.assertIn("科技/半導體AI", recs_us)
        self.assertIn("生技醫療/製藥", recs_us)

    def test_overfitting_simulation_structure(self):
        df, train_idx = simulate_overfitting_comparison()
        self.assertEqual(len(df), 200)
        self.assertEqual(train_idx, 140)
        self.assertIn("Actual_Price", df.columns)
        self.assertIn("Overfitted_ML_Curve", df.columns)

    def test_security_hardening(self):
        from stock_advisor.security import (
            sanitize_ticker_input, safe_html, check_client_rate_limit,
            get_security_logger, get_system_security_report
        )
        
        # 1. Valid ticker tests
        clean_tw, valid_tw, _ = sanitize_ticker_input("2330")
        self.assertEqual(clean_tw, "2330.TW")
        self.assertTrue(valid_tw)

        clean_us, valid_us, _ = sanitize_ticker_input("aapl ")
        self.assertEqual(clean_us, "AAPL")
        self.assertTrue(valid_us)

        # 2. Malicious injection attempts (XSS, SQLi, Path traversal)
        _, valid_xss, msg_xss = sanitize_ticker_input("<script>alert('xss')</script>")
        self.assertFalse(valid_xss)
        self.assertIn("注入攻擊", msg_xss)

        _, valid_sqli, msg_sqli = sanitize_ticker_input("' UNION SELECT * FROM users--")
        self.assertFalse(valid_sqli)

        _, valid_path, _ = sanitize_ticker_input("../../etc/passwd")
        self.assertFalse(valid_path)

        # 3. HTML escaping
        escaped = safe_html("<img src=x onerror=alert(1)>")
        self.assertNotIn("<img", escaped)
        self.assertIn("&lt;img", escaped)

        # 4. Audit logger & security report
        report = get_system_security_report()
        self.assertEqual(report["status"], "SECURE")
        self.assertEqual(report["zero_pii"], "Compliant (Stateless Local Session, No Persistent User Storage)")
        self.assertEqual(report["api_leak_risk"], "Zero (No Paid / Cloud Secret API Keys in Codebase)")

        logger = get_security_logger()
        logs = logger.get_recent_logs()
        self.assertGreater(len(logs), 0)

if __name__ == "__main__":
    unittest.main()
