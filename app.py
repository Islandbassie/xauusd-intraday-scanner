import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from metaapi_cloud_sdk import MetaApi

st.set_page_config(
    page_title="XAUUSD Intraday Scanner",
    page_icon="🥇",
    layout="wide",
)

DEFAULT_ACCOUNT = "e72c2ef2-6907-4a3a-90cd-d3846964629d"
DEFAULT_SYMBOL = "XAUUSD"


async def get_mt5(token, account_id, symbol):
    """
    Use MetaApi's account historical-data API for candles and the
    streaming connection for the live XAUUSD quote.

    Historical candles are exposed by MetatraderAccount.get_historical_candles();
    subscribe_to_market_data() belongs to the streaming connection.
    """
    api = MetaApi(token=token)

    streaming = None

    try:
        account = await api.metatrader_account_api.get_account(
            account_id=account_id
        )

        # ---------------------------------------------------------
        # 1) Historical candles
        # ---------------------------------------------------------
        # IMPORTANT: MetaApi exposes historical market data on the
        # MetatraderAccount object itself, not on RpcMetaApiConnection.
        # The current Python SDK documents account.get_historical_candles().
        candles = {}
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for tf in ("1d", "4h", "1h", "15m"):
            candles[tf] = await account.get_historical_candles(
                symbol=symbol,
                timeframe=tf,
                start_time=now,
                limit=250,
            )

        # ---------------------------------------------------------
        # 2) Streaming connection: live quote
        # ---------------------------------------------------------
        streaming = account.get_streaming_connection()
        await streaming.connect()
        await streaming.wait_synchronized()

        # IMPORTANT:
        # subscribe_to_market_data belongs to the streaming connection.
        await streaming.subscribe_to_market_data(symbol=symbol)

        # Give the synchronized terminal state a moment to receive the quote.
        price = None
        for _ in range(10):
            price = streaming.terminal_state.price(symbol=symbol)
            if price:
                break
            await asyncio.sleep(0.5)

        if not price:
            raise RuntimeError(
                f"MetaApi connected successfully, but no live quote was received "
                f"for '{symbol}'. Confirm XAUUSD is visible in MT5 Market Watch."
            )

        return {
            "account": account,
            "price": price,
            "candles": candles,
        }

    finally:
        try:
            if streaming is not None:
                await streaming.close()
        except Exception:
            pass

        try:
            await api.close()
        except Exception:
            pass


def fetch(token, account_id, symbol):
    return asyncio.run(get_mt5(token, account_id, symbol))


def df_from(candles):
    rows = []
    for x in candles:
        rows.append(
            {
                "time": pd.to_datetime(x["time"], utc=True),
                "open": float(x["open"]),
                "high": float(x["high"]),
                "low": float(x["low"]),
                "close": float(x["close"]),
                "volume": float(
                    x.get("tickVolume", x.get("volume", 0))
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )


def indicators(df):
    d = df.copy()

    d["ema20"] = d.close.ewm(span=20, adjust=False).mean()
    d["ema50"] = d.close.ewm(span=50, adjust=False).mean()

    delta = d.close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + gain / loss))

    tr = pd.concat(
        [
            d.high - d.low,
            (d.high - d.close.shift()).abs(),
            (d.low - d.close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    d["atr"] = tr.rolling(14).mean()

    return d


def bias(d):
    x = d.iloc[-1]

    if x.close > x.ema20 > x.ema50:
        return "BULLISH"

    if x.close < x.ema20 < x.ema50:
        return "BEARISH"

    return "NEUTRAL"


def structure(d):
    x = d.tail(30)

    if len(x) < 10:
        return "NEUTRAL"

    hi = x.high.rolling(5, center=True).max().dropna()
    lo = x.low.rolling(5, center=True).min().dropna()

    if len(hi) < 2 or len(lo) < 2:
        return "NEUTRAL"

    if hi.iloc[-1] > hi.iloc[-2] and lo.iloc[-1] > lo.iloc[-2]:
        return "BULLISH"

    if hi.iloc[-1] < hi.iloc[-2] and lo.iloc[-1] < lo.iloc[-2]:
        return "BEARISH"

    return "NEUTRAL"


def fvg(d):
    for i in range(len(d) - 1, 1, -1):
        a = d.iloc[i - 2]
        c = d.iloc[i]

        if c.low > a.high:
            return ("BULLISH", float(a.high), float(c.low))

        if c.high < a.low:
            return ("BEARISH", float(c.high), float(a.low))

    return None


def setup(d15, d1h, d4h, dd):
    """
    Multi-timeframe XAUUSD setup model.

    Directional bias is established from Daily, 4H and 1H.
    The 15M timeframe is the execution/setup timeframe and is the only
    timeframe used to generate the proposed entry, stop and targets.
    """
    d15, d1h, d4h, dd = map(indicators, (d15, d1h, d4h, dd))

    bd = bias(dd)
    b4 = bias(d4h)
    b1 = bias(d1h)
    b15 = bias(d15)
    s15 = structure(d15)

    r15 = float(d15.rsi.iloc[-1]) if pd.notna(d15.rsi.iloc[-1]) else 50

    sell = 0
    buy = 0
    sr = []
    br = []

    # Higher-timeframe directional bias.
    weights = {"Daily": 20, "4H": 25, "1H": 20}
    for name, b, w in (("Daily", bd, weights["Daily"]),
                       ("4H", b4, weights["4H"]),
                       ("1H", b1, weights["1H"])):
        if b == "BEARISH":
            sell += w
            sr.append(f"{name} bearish EMA alignment")
        elif b == "BULLISH":
            buy += w
            br.append(f"{name} bullish EMA alignment")

    # 15M is the only trade/setup timeframe.
    if b15 == "BEARISH":
        sell += 15
        sr.append("15M bearish EMA alignment")
    elif b15 == "BULLISH":
        buy += 15
        br.append("15M bullish EMA alignment")

    if s15 == "BEARISH":
        sell += 15
        sr.append("15M bearish structure")
    elif s15 == "BULLISH":
        buy += 15
        br.append("15M bullish structure")

    if r15 < 45:
        sell += 5
        sr.append("15M momentum below neutral")
    elif r15 > 55:
        buy += 5
        br.append("15M momentum above neutral")

    z = fvg(d15)
    if z and z[0] == "BEARISH":
        sell += 5
        sr.append("Recent bearish 15M FVG")
    elif z and z[0] == "BULLISH":
        buy += 5
        br.append("Recent bullish 15M FVG")

    # Execution gate: 15M direction must agree with the 1H direction,
    # and the 4H must not be directly opposed to the proposed trade.
    buy_gate = b1 == "BULLISH" and b15 in ("BULLISH",) and b4 != "BEARISH"
    sell_gate = b1 == "BEARISH" and b15 in ("BEARISH",) and b4 != "BULLISH"

    if not buy_gate:
        buy = min(buy, 59)
    if not sell_gate:
        sell = min(sell, 59)

    # 15M ATR drives trade planning; no lower timeframe is used.
    atr15 = float(d15.atr.iloc[-1]) if pd.notna(d15.atr.iloc[-1]) else float((d15.high - d15.low).tail(14).mean())
    p = float(d15.close.iloc[-1])

    return {
        "price": p,
        "rsi15": r15,
        "atr15": atr15,
        "bd": bd,
        "b4": b4,
        "b1": b1,
        "b15": b15,
        "s15": s15,
        "buy_gate": buy_gate,
        "sell_gate": sell_gate,
        "sell": min(sell, 100),
        "buy": min(buy, 100),
        "sr": sr,
        "br": br,
        "fvg": z,
    }


def trade_plan(d15, signal, bid, ask):
    """
    Rule-based planning levels.

    These are analytical levels, not guarantees or broker execution
    instructions. The plan uses:
      - current executable side of the quote for the entry reference
      - recent 5M swing structure
      - ATR as a volatility buffer
      - fixed 1R / 2R / 3R targets
    """
    current = ask if signal == "BUY" else bid

    atr = float(d15["atr"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        atr = float((d15["high"] - d15["low"]).tail(14).mean())

    recent = d15.tail(20)

    if signal == "BUY":
        swing_low = float(recent["low"].min())
        # Put SL below recent structure with an ATR buffer.
        sl = swing_low - (0.15 * atr)

        # If structure is too close, use an ATR-based fallback.
        if sl >= current:
            sl = current - atr

        risk = current - sl
        tp1 = current + risk
        tp2 = current + (2 * risk)
        tp3 = current + (3 * risk)

    else:
        swing_high = float(recent["high"].max())
        sl = swing_high + (0.15 * atr)

        if sl <= current:
            sl = current + atr

        risk = sl - current
        tp1 = current - risk
        tp2 = current - (2 * risk)
        tp3 = current - (3 * risk)

    rr1 = 1.0
    rr2 = 2.0
    rr3 = 3.0

    return {
        "signal": signal,
        "entry": current,
        "sl": sl,
        "risk": risk,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rr1": rr1,
        "rr2": rr2,
        "rr3": rr3,
        "atr": atr,
    }


def classify_score(score):
    if score >= 90:
        return "VERY HIGH CONFLUENCE", "🔥"
    if score >= 80:
        return "HIGH CONFLUENCE", "🟢"
    if score >= 70:
        return "TRADE CANDIDATE", "🟢"
    if score >= 60:
        return "MODERATE", "🟡"
    if score >= 45:
        return "DEVELOPING", "⚪"
    return "WAIT", "⚫"


def chart(d, title):
    d = d.tail(120)

    fig = go.Figure(
        [
            go.Candlestick(
                x=d.time,
                open=d.open,
                high=d.high,
                low=d.low,
                close=d.close,
                name="XAUUSD",
            ),
            go.Scatter(
                x=d.time,
                y=d.ema20,
                name="EMA20",
            ),
            go.Scatter(
                x=d.time,
                y=d.ema50,
                name="EMA50",
            ),
        ]
    )

    fig.update_layout(
        title=title,
        height=500,
        xaxis_rangeslider_visible=False,
    )

    return fig


st.title("🥇 XAUUSD Intraday Setup Scanner")
st.caption(
    "BlackBull Markets MT5 market data via MetaApi • "
    "read-only decision support • no automatic order execution"
)

with st.sidebar:
    st.header("Scanner settings")

    symbol = st.text_input(
        "MT5 symbol",
        DEFAULT_SYMBOL,
    ).strip().upper()

    threshold = st.slider(
        "Minimum setup score",
        0,
        100,
        45,
        5,
    )

    st.subheader("Live scanner")

    auto_refresh = st.checkbox(
        "Auto-update scanner",
        value=True,
        help="Refresh the MT5 data automatically while this page is open.",
    )

    refresh_seconds = st.selectbox(
        "Update interval",
        [300],
        index=0,
        disabled=not auto_refresh,
    )

    st.subheader("Risk & cost settings")

    account_size = st.number_input(
        "Account size ($)",
        min_value=100.0,
        value=10000.0,
        step=100.0,
    )

    risk_pct = st.number_input(
        "Risk per trade (%)",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )

    lot_size = st.number_input(
        "Lot size",
        min_value=0.01,
        max_value=100.0,
        value=0.10,
        step=0.01,
        format="%.2f",
    )

    contract_size = st.number_input(
        "XAUUSD contract size (oz/lot)",
        min_value=1.0,
        value=100.0,
        step=1.0,
    )

    # BlackBull ECN Standard: commission defaults to $0.
    commission_per_lot_round_turn = st.number_input(
        "Commission ($/lot, round turn)",
        min_value=0.0,
        value=0.0,
        step=0.1,
    )

    refresh = st.button(
        "🔄 Refresh MT5 data",
        use_container_width=True,
    )

    st.divider()

    st.write("Data source: **BlackBull Markets — MT5**")
    st.write("Server: `BlackbullMarkets-Live`")

try:
    token = st.secrets["METAAPI_TOKEN"]
    account_id = st.secrets.get(
        "METAAPI_ACCOUNT_ID",
        DEFAULT_ACCOUNT,
    )
except Exception:
    st.error(
        "MetaApi secrets are missing. Add METAAPI_TOKEN and "
        "METAAPI_ACCOUNT_ID in Streamlit Cloud → Manage app → "
        "Settings → Secrets."
    )
    st.stop()



def render_live_scanner():
    """Fetch fresh MT5 data and redraw the live scanner."""
    try:
        live = fetch(token, account_id, symbol)
        st.session_state.mt5 = live
        st.session_state.err = None
        st.session_state.last_update = datetime.now(timezone.utc)
    except Exception as e:
        st.session_state.err = str(e)

    if st.session_state.get("err"):
        st.error("MT5/MetaApi data error")
        st.code(st.session_state.err)
        st.info(
            "The live scanner could not refresh this cycle. "
            "Confirm the MetaApi connection and that XAUUSD is visible "
            "in BlackBull MT5 Market Watch."
        )
        return

    raw = st.session_state.mt5
    price = raw["price"]

    dd = indicators(df_from(raw["candles"]["1d"]))
    d4 = indicators(df_from(raw["candles"]["4h"]))
    d1 = indicators(df_from(raw["candles"]["1h"]))
    d15 = indicators(df_from(raw["candles"]["15m"]))

    s = setup(d15, d1, d4, dd)

    bid = float(price["bid"])
    ask = float(price["ask"])
    spread = ask - bid

    # Determine the stronger directional setup.
    if s["buy_gate"] and s["buy"] > s["sell"]:
        primary_signal = "BUY"
        primary_score = s["buy"]
    elif s["sell_gate"] and s["sell"] > s["buy"]:
        primary_signal = "SELL"
        primary_score = s["sell"]
    else:
        primary_signal = "NEUTRAL"
        primary_score = max(s["buy"], s["sell"])

    score_label, score_icon = classify_score(primary_score)

    plan = None
    if primary_signal in ("BUY", "SELL"):
        plan = trade_plan(
            d15,
            primary_signal,
            bid,
            ask,
        )

    spread_cost = spread * contract_size * lot_size
    commission_cost = commission_per_lot_round_turn * lot_size
    round_turn_cost = spread_cost + commission_cost

    risk_amount = account_size * (risk_pct / 100.0)
    risk_distance = (
        risk_amount / (contract_size * lot_size)
        if contract_size > 0 and lot_size > 0
        else 0.0
    )

    a, b, c, d, e = st.columns(5)
    a.metric("XAUUSD Bid", f"{bid:,.2f}")
    b.metric("XAUUSD Ask", f"{ask:,.2f}")
    c.metric("Spread", f"{spread:.2f}")
    d.metric("1H Bias", s["b1"])
    e.metric("15M RSI", f"{s['rsi15']:.1f}")

    server_name = getattr(
        raw["account"],
        "server",
        "BlackbullMarkets-Live",
    )

    update_time = st.session_state.get("last_update")
    update_text = (
        update_time.strftime("%H:%M:%S UTC")
        if update_time
        else "unknown"
    )

    st.success(
        f"🟢 BlackBull MT5 connected • "
        f"Server: {server_name} • "
        f"Live XAUUSD quote received • "
        f"Last update: {update_text}"
    )

    st.divider()

    st.subheader("🧭 Multi-timeframe directional bias")
    b1, b2, b3, b4c = st.columns(4)
    b1.metric("Daily", s["bd"])
    b2.metric("4H", s["b4"])
    b3.metric("1H", s["b1"])
    b4c.metric("15M setup", s["s15"])

    if s["buy_gate"]:
        st.success("🟢 Long bias is aligned: 1H + 15M agree and 4H is not opposing.")
    elif s["sell_gate"]:
        st.error("🔴 Short bias is aligned: 1H + 15M agree and 4H is not opposing.")
    else:
        st.warning("🟡 No clean execution alignment: wait for 15M confirmation or a better higher-timeframe alignment.")

    st.divider()

    st.subheader("🎯 Intraday setup plan")

    if primary_signal == "NEUTRAL":
        st.info(
            "⚪ BUY and SELL scores are equal. No directional setup "
            "is currently preferred."
        )
    else:
        st.markdown(
            f"### {score_icon} {primary_signal} — "
            f"{primary_score}/100"
        )
        st.write(f"**Classification:** {score_label}")

        if primary_score < 70:
            st.info(
                "The setup is below the 70/100 Trade Candidate threshold. "
                "Treat the levels below as analytical reference levels and "
                "wait for additional price-action confirmation."
            )
        elif primary_score < 80:
            st.warning(
                "Trade Candidate: the rule-based conditions are aligned, "
                "but confirmation is still required before considering an entry."
            )
        else:
            st.success(
                "High-confluence rule-based setup. The score measures "
                "alignment of the scanner's conditions; it is not a "
                "probability of winning."
            )

        if plan:
            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "Possible entry",
                f"{plan['entry']:,.2f}",
            )

            p2.metric(
                "Stop loss",
                f"{plan['sl']:,.2f}",
            )

            p3.metric(
                "TP1 / 1R",
                f"{plan['tp1']:,.2f}",
            )

            p4.metric(
                "TP2 / 2R",
                f"{plan['tp2']:,.2f}",
            )

            p5, p6, p7, p8 = st.columns(4)

            p5.metric(
                "TP3 / 3R",
                f"{plan['tp3']:,.2f}",
            )

            p6.metric(
                "Risk distance",
                f"{plan['risk']:,.2f}",
            )

            p7.metric(
                "R:R to TP2",
                "1:2",
            )

            p8.metric(
                "R:R to TP3",
                "1:3",
            )

            st.caption(
                "Entry uses the current executable side of the live quote "
                "(Ask for BUY / Bid for SELL). Stop loss uses recent 15M "
                "structure plus a 15M ATR buffer. TP levels are fixed 1R/2R/3R "
                "planning targets. These are analytical levels, not guaranteed "
                "future prices or automatic orders."
            )

    st.divider()

    st.subheader("💰 BlackBull ECN Standard — estimated trading cost")

    cost1, cost2, cost3, cost4 = st.columns(4)
    cost1.metric("Live spread", f"${spread:.2f}/oz")
    cost2.metric(
        f"Spread cost ({lot_size:.2f} lot)",
        f"${spread_cost:.2f}",
    )
    cost3.metric("Commission", f"${commission_cost:.2f}")
    cost4.metric(
        "Estimated round-turn cost",
        f"${round_turn_cost:.2f}",
    )

    st.caption(
        "ECN Standard commission is set to $0 by default. "
        "The spread is taken directly from the live MT5 Bid/Ask quote. "
        "Swap/overnight financing is not included."
    )

    risk1, risk2, risk3 = st.columns(3)
    risk1.metric(
        "Account risk",
        f"${risk_amount:,.2f}",
        f"{risk_pct:.2f}%",
    )
    risk2.metric("Selected volume", f"{lot_size:.2f} lot")
    risk3.metric("Risk distance", f"${risk_distance:.2f}/oz")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader(f"🔴 SELL setup — {s['sell']}/100")
        st.write(
            "Conditions: "
            + (
                " • ".join(s["sr"])
                if s["sr"]
                else "No strong bearish conditions"
            )
        )

        if s["sell"] < threshold:
            st.info(
                "Score is below your selected threshold. "
                "Analytical setup only — verify live market conditions."
            )
        else:
            st.warning(
                "Score is above your selected threshold. "
                "Analytical setup only — verify live market conditions."
            )

    with right:
        st.subheader(f"🟢 BUY setup — {s['buy']}/100")
        st.write(
            "Conditions: "
            + (
                " • ".join(s["br"])
                if s["br"]
                else "No strong bullish conditions"
            )
        )

        if s["buy"] < threshold:
            st.info(
                "Score is below your selected threshold. "
                "Analytical setup only — verify live market conditions."
            )
        else:
            st.success(
                "Score is above your selected threshold. "
                "Analytical setup only — verify live market conditions."
            )

    st.divider()

    st.subheader("📊 BlackBull XAUUSD charts")
    t1, t2, t3, t4 = st.tabs(["Daily", "4H", "1H", "15M — Trades"])

    with t1:
        st.plotly_chart(chart(dd, "BlackBull XAUUSD — Daily"), use_container_width=True)
    with t2:
        st.plotly_chart(chart(d4, "BlackBull XAUUSD — 4H"), use_container_width=True)
    with t3:
        st.plotly_chart(chart(d1, "BlackBull XAUUSD — 1H"), use_container_width=True)
    with t4:
        st.plotly_chart(chart(d15, "BlackBull XAUUSD — 15M — execution timeframe"), use_container_width=True)

    st.caption(
        "Daily/4H/1H establish directional bias. The 15M is the only "
        "timeframe used for trade setup, entry, SL and TP planning. Scores "
        "are rule-based technical conditions, not guarantees or probabilities. "
        "The app does not place trades. Trading-cost figures are estimates "
        "based on the live spread and selected lot size; verify the broker's "
        "final execution cost."
    )


# Streamlit fragments support automatic reruns without rerunning the whole app.
# This is suitable for a live scanner/monitoring display.
# The scanner defaults to a 5-minute refresh interval to reduce demand.
# Trades are generated only from the 15M setup timeframe; Daily/4H/1H
# provide directional context and confirmation.
run_every = f"{refresh_seconds}s" if auto_refresh else None

@st.fragment(run_every=run_every)
def live_scanner_fragment():
    render_live_scanner()

live_scanner_fragment()
