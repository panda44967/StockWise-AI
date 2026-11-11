@echo off
chcp 65001 >nul
echo ========================================================
echo  正在啟動 StockWise-AI 長期投資決策平台...
echo ========================================================
cd /d "%~dp0"
streamlit run stock_advisor/app.py
pause
