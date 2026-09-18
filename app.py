import asyncio
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from metaapi_cloud_sdk import MetaApi

st.set_page_config(page_title='XAUUSD Intraday Scanner', page_icon='🥇', layout='wide')
DEFAULT_ACCOUNT='e72c2ef2-6907-4a3a-90cd-d3846964629d'
DEFAULT_SYMBOL='XAUUSD'

async def get_mt5(token, account_id, symbol):
    api=MetaApi(token=token)
    try:
        account=await api.metatrader_account_api.get_account(account_id=account_id)
        conn=account.get_rpc_connection()
        await conn.connect()
        await conn.wait_synchronized()

        # MetaApi requires a market-data subscription before a live quote
        # is available through the RPC connection.
        await conn.subscribe_to_market_data(symbol)

        # Confirm that the broker actually exposes this exact symbol.
        symbols = await conn.get_symbols()
        if symbol not in symbols:
            matches = [s for s in symbols if 'XAU' in s.upper() or 'GOLD' in s.upper()]
            detail = ', '.join(matches[:20]) if matches else 'No XAU/GOLD symbols were returned by the broker.'
            raise RuntimeError(
                f"Symbol '{symbol}' is not available on the connected MT5 account. "
                f"Gold-related symbols found: {detail}"
            )

        price=await conn.get_symbol_price(symbol)
        candles={}
        for tf in ('1h','15m','5m'):
            candles[tf]=await conn.get_historical_candles(symbol=symbol,timeframe=tf,start_time=None,limit=250)
        return {'account':account,'price':price,'candles':candles}
    finally:
        try: await api.close()
        except Exception: pass

def fetch(token, account_id, symbol):
    return asyncio.run(get_mt5(token, account_id, symbol))

def df_from(candles):
    df=pd.DataFrame([{'time':pd.to_datetime(x['time'],utc=True),'open':float(x['open']),'high':float(x['high']),'low':float(x['low']),'close':float(x['close']),'volume':float(x.get('tickVolume',x.get('volume',0)))} for x in candles])
    return df.drop_duplicates('time').sort_values('time').reset_index(drop=True)

def indicators(df):
    d=df.copy(); d['ema20']=d.close.ewm(span=20,adjust=False).mean(); d['ema50']=d.close.ewm(span=50,adjust=False).mean()
    delta=d.close.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean().replace(0,np.nan); d['rsi']=100-(100/(1+gain/loss))
    tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1); d['atr']=tr.rolling(14).mean()
    return d

def bias(d):
    x=d.iloc[-1]
    if x.close>x.ema20>x.ema50: return 'BULLISH'
    if x.close<x.ema20<x.ema50: return 'BEARISH'
    return 'NEUTRAL'

def structure(d):
    x=d.tail(30)
    if len(x)<10:return 'NEUTRAL'
    hi=x.high.rolling(5,center=True).max().dropna(); lo=x.low.rolling(5,center=True).min().dropna()
    if len(hi)<2 or len(lo)<2:return 'NEUTRAL'
    if hi.iloc[-1]>hi.iloc[-2] and lo.iloc[-1]>lo.iloc[-2]:return 'BULLISH'
    if hi.iloc[-1]<hi.iloc[-2] and lo.iloc[-1]<lo.iloc[-2]:return 'BEARISH'
    return 'NEUTRAL'

def fvg(d):
    for i in range(len(d)-1,1,-1):
        a=d.iloc[i-2]; c=d.iloc[i]
        if c.low>a.high:return ('BULLISH',float(a.high),float(c.low))
        if c.high<a.low:return ('BEARISH',float(c.high),float(a.low))
    return None

def setup(d5,d15,d1):
    d5,d15,d1=map(indicators,(d5,d15,d1)); b1,b15,s15=bias(d1),bias(d15),structure(d15); r=float(d5.rsi.iloc[-1]) if pd.notna(d5.rsi.iloc[-1]) else 50; atr=float(d5.atr.iloc[-1]) if pd.notna(d5.atr.iloc[-1]) else float((d5.high-d5.low).tail(14).mean()); p=float(d5.close.iloc[-1]); sell=buy=0; sr=[]; br=[]
    if b1=='BEARISH':sell+=25;sr.append('1H bearish EMA alignment')
    elif b1=='BULLISH':buy+=25;br.append('1H bullish EMA alignment')
    if b15=='BEARISH':sell+=20;sr.append('15M bearish EMA alignment')
    elif b15=='BULLISH':buy+=20;br.append('15M bullish EMA alignment')
    if s15=='BEARISH':sell+=20;sr.append('15M bearish structure')
    elif s15=='BULLISH':buy+=20;br.append('15M bullish structure')
    if r<45:sell+=10;sr.append('5M momentum below neutral')
    if r>55:buy+=10;br.append('5M momentum above neutral')
    z=fvg(d15)
    if z and z[0]=='BEARISH':sell+=15;sr.append('Recent bearish FVG')
    if z and z[0]=='BULLISH':buy+=15;br.append('Recent bullish FVG')
    return {'price':p,'rsi':r,'atr':atr,'b1':b1,'b15':b15,'s15':s15,'sell':min(sell,100),'buy':min(buy,100),'sr':sr,'br':br,'fvg':z}

def chart(d,title):
    d=d.tail(120); fig=go.Figure([go.Candlestick(x=d.time,open=d.open,high=d.high,low=d.low,close=d.close,name='XAUUSD'),go.Scatter(x=d.time,y=d.ema20,name='EMA20'),go.Scatter(x=d.time,y=d.ema50,name='EMA50')]); fig.update_layout(title=title,height=500,xaxis_rangeslider_visible=False); return fig

st.title('🥇 XAUUSD Intraday Setup Scanner')
st.caption('BlackBull Markets MT5 market data via MetaApi • read-only decision support • no automatic order execution')
with st.sidebar:
    st.header('Scanner settings'); symbol=st.text_input('MT5 symbol',DEFAULT_SYMBOL).strip().upper(); threshold=st.slider('Minimum setup score',0,100,45,5); refresh=st.button('🔄 Refresh MT5 data',use_container_width=True)
    st.divider(); st.write('Data source: **BlackBull Markets — MT5**'); st.write('Server: `BlackbullMarkets-Live`')
try:
    token=st.secrets['METAAPI_TOKEN']; account_id=st.secrets.get('METAAPI_ACCOUNT_ID',DEFAULT_ACCOUNT)
except Exception:
    st.error('MetaApi secrets are missing. Add METAAPI_TOKEN and METAAPI_ACCOUNT_ID in Streamlit Cloud → Manage app → Settings → Secrets.'); st.stop()
if refresh or 'mt5' not in st.session_state:
    with st.spinner('Connecting to BlackBull MT5 and loading 5M / 15M / 1H candles...'):
        try: st.session_state.mt5=fetch(token,account_id,symbol); st.session_state.err=None; st.session_state.refreshed=datetime.now(timezone.utc)
        except Exception as e: st.session_state.err=str(e)
if st.session_state.get('err'):
    st.error('MT5/MetaApi data error'); st.code(st.session_state.err); st.info('Check that MetaApi shows the account as CONNECTED and DEPLOYED, and that the broker symbol is exactly XAUUSD.'); st.stop()
raw=st.session_state.mt5; price=raw['price']; d5=indicators(df_from(raw['candles']['5m'])); d15=indicators(df_from(raw['candles']['15m'])); d1=indicators(df_from(raw['candles']['1h'])); s=setup(d5,d15,d1); bid=float(price['bid']); ask=float(price['ask'])
a,b,c,d,e=st.columns(5); a.metric('XAUUSD Bid',f'{bid:,.2f}'); b.metric('XAUUSD Ask',f'{ask:,.2f}'); c.metric('1H Bias',s['b1']); d.metric('15M Structure',s['s15']); e.metric('5M RSI',f'{s["rsi"]:.1f}')
st.success(f"🟢 BlackBull MT5 connected • Server: {getattr(raw['account'],'server','BlackbullMarkets-Live')} • Broker quote: {price.get('brokerTime',price.get('time',''))}")
st.divider(); l,r=st.columns(2)
with l:
    st.subheader(f'🔴 SELL setup — {s["sell"]}/100'); st.write('Conditions: '+(' • '.join(s['sr']) if s['sr'] else 'No strong bearish conditions')); st.info('Analytical setup only — verify live market conditions before acting.') if s['sell']<threshold else st.warning('Score is above your selected threshold.')
with r:
    st.subheader(f'🟢 BUY setup — {s["buy"]}/100'); st.write('Conditions: '+(' • '.join(s['br']) if s['br'] else 'No strong bullish conditions')); st.info('Analytical setup only — verify live market conditions before acting.') if s['buy']<threshold else st.success('Score is above your selected threshold.')
st.divider(); st.subheader('📊 BlackBull XAUUSD charts'); t1,t2,t3=st.tabs(['5M','15M','1H'])
with t1: st.plotly_chart(chart(d5,'BlackBull XAUUSD — 5M'),use_container_width=True)
with t2: st.plotly_chart(chart(d15,'BlackBull XAUUSD — 15M'),use_container_width=True)
with t3: st.plotly_chart(chart(d1,'BlackBull XAUUSD — 1H'),use_container_width=True)
st.caption('Scores are rule-based technical conditions, not guarantees or automatic trade instructions. The app does not place trades.')
