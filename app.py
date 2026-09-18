import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="XAUUSD Intraday Scanner", page_icon="🥇", layout="wide")

st.title("🥇 XAUUSD Intraday Setup Scanner")
st.caption("Decision-support scanner — no automatic order execution.")

# ----------------------------
# Settings
# ----------------------------
with st.sidebar:
    st.header("Scanner settings")
    symbol = st.text_input("Yahoo symbol", "GC=F")
    period = st.selectbox("Data period", ["5d", "10d", "1mo"], index=1)
    setup_tf = st.selectbox("Setup timeframe", ["5m", "15m", "30m"], index=1)
    htf_tf = st.selectbox("Higher timeframe", ["30m", "1h", "4h"], index=1)
    min_score = st.slider("Minimum setup-condition score", 0, 100, 45)
    risk_pct = st.number_input("Example account risk %", 0.1, 5.0, 1.0, 0.1)
    account = st.number_input("Example account size", 1000.0, 1000000.0, 10000.0, 500.0)

@st.cache_data(ttl=60)
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open","High","Low","Close"]].dropna()

def indicators(df):
    x = df.copy()
    x["EMA20"] = x.Close.ewm(span=20, adjust=False).mean()
    x["EMA50"] = x.Close.ewm(span=50, adjust=False).mean()
    x["EMA200"] = x.Close.ewm(span=200, adjust=False).mean()
    d = x.Close.diff()
    gain, loss = d.clip(lower=0), -d.clip(upper=0)
    ag = gain.ewm(alpha=1/14, adjust=False).mean()
    al = loss.ewm(alpha=1/14, adjust=False).mean()
    x["RSI"] = (100 - 100/(1 + ag/al.replace(0,np.nan))).fillna(50)
    e12 = x.Close.ewm(span=12, adjust=False).mean()
    e26 = x.Close.ewm(span=26, adjust=False).mean()
    x["MACD"] = e12-e26
    x["MACDSignal"] = x.MACD.ewm(span=9, adjust=False).mean()
    prev = x.Close.shift(1)
    tr = pd.concat([(x.High-x.Low), (x.High-prev).abs(), (x.Low-prev).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    return x.dropna()

def bias(df):
    if len(df) < 10: return "UNKNOWN"
    r=df.iloc[-1]
    if r.Close > r.EMA20 > r.EMA50 > r.EMA200: return "STRONG BULLISH"
    if r.Close < r.EMA20 < r.EMA50 < r.EMA200: return "STRONG BEARISH"
    if r.Close > r.EMA50 and r.EMA20 > r.EMA50: return "BULLISH"
    if r.Close < r.EMA50 and r.EMA20 < r.EMA50: return "BEARISH"
    return "RANGING"

def swings(df, n=3):
    highs=[]; lows=[]
    for i in range(n, len(df)-n):
        if df.High.iloc[i] == df.High.iloc[i-n:i+n+1].max(): highs.append(i)
        if df.Low.iloc[i] == df.Low.iloc[i-n:i+n+1].min(): lows.append(i)
    return highs,lows

def structure(df):
    hi,lo=swings(df)
    if len(hi)<2 or len(lo)<2: return "RANGING",None,None
    if df.High.iloc[hi[-1]] > df.High.iloc[hi[-2]] and df.Low.iloc[lo[-1]] > df.Low.iloc[lo[-2]]:
        state="BULLISH"
    elif df.High.iloc[hi[-1]] < df.High.iloc[hi[-2]] and df.Low.iloc[lo[-1]] < df.Low.iloc[lo[-2]]:
        state="BEARISH"
    else: state="RANGING"
    bos=None
    if df.Close.iloc[-1] > df.High.iloc[hi[-2]]: bos="BULLISH BOS"
    elif df.Close.iloc[-1] < df.Low.iloc[lo[-2]]: bos="BEARISH BOS"
    return state,bos,(hi,lo)

def liquidity(df):
    hi,lo=swings(df)
    if not hi or not lo: return None
    atr=float(df.ATR.iloc[-1])
    h=float(df.High.iloc[-1]); l=float(df.Low.iloc[-1]); c=float(df.Close.iloc[-1])
    rh=float(df.High.iloc[hi[-1]]); rl=float(df.Low.iloc[lo[-1]])
    tol=atr*.12
    if h>rh and h>=rh-tol and c<rh: return ("BUY-SIDE SWEEP",rh)
    if l<rl and l<=rl+tol and c>rl: return ("SELL-SIDE SWEEP",rl)
    return None

def fvgs(df):
    out=[]
    for i in range(2,len(df)):
        a,b,c=df.iloc[i-2],df.iloc[i-1],df.iloc[i]
        atr=float(b.ATR)
        if float(c.Low)-float(a.High) > atr*.10:
            out.append(("BULLISH FVG",float(a.High),float(c.Low),df.index[i-2]))
        if float(a.Low)-float(c.High) > atr*.10:
            out.append(("BEARISH FVG",float(c.High),float(a.Low),df.index[i-2]))
    return out

def make_setup(df, direction, score, reasons, kind):
    last=df.iloc[-1]; entry=float(last.Close); atr=float(last.ATR)
    hi,lo=swings(df)
    if direction=="BUY":
        base=float(df.Low.iloc[lo[-1]]) if lo else entry-atr*1.2
        sl=min(base-atr*.20, entry-atr*1.2)
        risk=entry-sl
        tp1=entry+1.5*risk; tp2=entry+2.5*risk
    else:
        base=float(df.High.iloc[hi[-1]]) if hi else entry+atr*1.2
        sl=max(base+atr*.20, entry+atr*1.2)
        risk=sl-entry
        tp1=entry-1.5*risk; tp2=entry-2.5*risk
    if risk<=0:return None
    return dict(direction=direction, kind=kind, score=score, entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                rr1=1.5, rr2=2.5, reasons=reasons)

def scan(htf, mtf):
    hb=bias(htf); mb=bias(mtf); st,bos,_=structure(mtf); sw=liquidity(mtf); gaps=fvgs(mtf)
    mom="BULLISH" if mtf.RSI.iloc[-1]>55 and mtf.MACD.iloc[-1]>mtf.MACDSignal.iloc[-1] else \
        "BEARISH" if mtf.RSI.iloc[-1]<45 and mtf.MACD.iloc[-1]<mtf.MACDSignal.iloc[-1] else "NEUTRAL"
    results=[]
    for direction in ["BUY","SELL"]:
        score=0; reasons=[]
        bull=direction=="BUY"
        if (bull and "BULLISH" in hb) or ((not bull) and "BEARISH" in hb): score+=20; reasons.append("HTF EMA alignment")
        if (bull and "BULLISH" in mb) or ((not bull) and "BEARISH" in mb): score+=20; reasons.append("15m/setup EMA alignment")
        if mom==direction.replace("BUY","BULLISH").replace("SELL","BEARISH"): score+=15; reasons.append("Momentum confirmation")
        if bos==( "BULLISH BOS" if bull else "BEARISH BOS"): score+=20; reasons.append("Break of structure")
        if sw and sw[0]==("SELL-SIDE SWEEP" if bull else "BUY-SIDE SWEEP"): score+=15; reasons.append("Liquidity sweep")
        ftype="BULLISH FVG" if bull else "BEARISH FVG"
        if any(g[0]==ftype for g in gaps[-5:]): score+=10; reasons.append("Recent FVG")
        if score>=min_score:
            s=make_setup(mtf,direction,score,reasons,"STRUCTURE / LIQUIDITY")
            if s: results.append(s)
    return results,dict(htf=hb,mtf=mb,momentum=mom,structure=st,bos=bos or "NONE",
                        liquidity=sw[0] if sw else "NONE",fvg_count=len(gaps))

# ----------------------------
# Load
# ----------------------------
try:
    htf = indicators(load_data(symbol, period, htf_tf))
    mtf = indicators(load_data(symbol, period, setup_tf))
except Exception as e:
    st.error(str(e)); st.stop()

if htf.empty or mtf.empty:
    st.warning("No data available. Check symbol, internet access, or market/data availability.")
    st.stop()

setups,diag=scan(htf,mtf)
last=mtf.iloc[-1]

# Header metrics
c1,c2,c3,c4,c5=st.columns(5)
c1.metric("XAUUSD proxy",f"{float(last.Close):.2f}")
c2.metric("HTF",diag["htf"])
c3.metric("Setup TF",diag["mtf"])
c4.metric("Structure",diag["structure"])
c5.metric("RSI",f"{float(last.RSI):.1f}")

st.divider()

# Setup cards
if not setups:
    st.info("NO TRADE — no setup currently meets the selected condition score.")
else:
    setups=sorted(setups,key=lambda x:x["score"],reverse=True)
    for s in setups:
        st.subheader(f"{'🟢' if s['direction']=='BUY' else '🔴'} {s['direction']} — {s['kind']}  |  {s['score']}/100")
        a,b,c,d,e=st.columns(5)
        a.metric("Entry",f"{s['entry']:.2f}")
        b.metric("SL",f"{s['sl']:.2f}")
        c.metric("TP1",f"{s['tp1']:.2f}")
        d.metric("TP2",f"{s['tp2']:.2f}")
        e.metric("R:R",f"1:{s['rr2']:.1f}")
        risk_money=account*risk_pct/100
        qty=risk_money/abs(s["entry"]-s["sl"])
        st.write("**Conditions:** "+" • ".join(s["reasons"]))
        st.caption(f"Example risk ${risk_money:.2f}; distance-based quantity ≈ {qty:.2f} units. Broker lot sizing varies.")
        st.divider()

# Chart
st.subheader("Live-style chart markup")
plot=mtf.tail(150)
fig,ax=plt.subplots(figsize=(16,7))
ax.plot(plot.index,plot.Close,label="Price")
ax.plot(plot.index,plot.EMA20,label="EMA20")
ax.plot(plot.index,plot.EMA50,label="EMA50")
ax.plot(plot.index,plot.EMA200,label="EMA200")
for s in setups:
    ax.axhline(s["entry"],linestyle="-",linewidth=1.2,label=f"{s['direction']} Entry")
    ax.axhline(s["sl"],linestyle=":",linewidth=1.2,label="SL")
    ax.axhline(s["tp1"],linestyle="--",linewidth=1.0,label="TP1")
    ax.axhline(s["tp2"],linestyle="--",linewidth=1.0,label="TP2")
ax.set_title("XAUUSD setup markup")
ax.grid(True,alpha=.2); ax.legend(ncol=3)
st.pyplot(fig)

# Diagnostics
with st.expander("Scanner diagnostics"):
    st.json(diag)
    st.write("Latest candle:", {k: float(last[k]) for k in ["Open","High","Low","Close","ATR","RSI"]})

st.caption("Data source: Yahoo Finance Gold Futures proxy (GC=F). For broker-accurate XAUUSD, connect the backend to your MT5 terminal.")
