# XAUUSD Intraday Scanner Web App

## Quick start
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:
   `pip install -r requirements.txt`
4. Run:
   `streamlit run app.py`
5. Open the local URL Streamlit prints (usually http://localhost:8501).

## Current version
Uses Yahoo Finance `GC=F` as a gold proxy. It scans:
- 1H/selected higher timeframe bias
- 15m/selected setup timeframe structure
- EMA 20/50/200
- RSI/MACD
- BOS
- liquidity sweeps
- FVGs
- ATR SL
- TP1/TP2
- condition score
- chart markup

## MT5 upgrade
The next production step is replacing the Yahoo data loader with your MT5 broker feed. This lets the app use the exact broker XAUUSD symbol, contract specification and live candles. It should remain analysis-only until you explicitly add and test execution controls.
