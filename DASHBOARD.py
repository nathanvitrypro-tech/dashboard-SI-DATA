import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf
import numpy as np

# =========================================================
# 1. CONFIGURATION & STYLE (HEDGE FUND THEME)
# =========================================================
st.set_page_config(layout="wide", page_title="CAC 40 Pro Dash", page_icon="🇫🇷")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
        color: #e0e0e0;
    }
    
    .stApp {
        background: rgb(10,10,25);
        background: linear-gradient(170deg, rgba(10,10,25,1) 0%, rgba(20,5,40,1) 60%, rgba(0,0,0,1) 100%);
        background-attachment: fixed;
    }
    
    .block-container { 
        padding-top: 4rem !important; 
        padding-bottom: 2rem; 
    }
    
    /* --- TOOLTIPS & CARDS --- */
    .tooltip-icon {
        display: inline-flex;
        justify-content: center;
        align-items: center;
        width: 14px;
        height: 14px;
        background-color: rgba(255,255,255,0.2);
        color: #e0e0e0;
        border-radius: 50%;
        font-size: 9px;
        margin-left: 6px;
        cursor: help;
        vertical-align: middle;
        position: relative;
        top: -1px;
    }
    .tooltip-icon:hover { background-color: rgba(255,255,255,0.8); color: #000; }

    .kpi-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%;
    }
    
    .kpi-title {
        color: #BBBBBB !important; font-size: 0.75rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1px; display: inline-block;
    }
    
    .kpi-value {
        font-size: 1.5rem; font-weight: 800; color: #FFFFFF;
        background: linear-gradient(90deg, #00f2c3, #0099ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    
    .kpi-detail { font-size: 0.8rem; margin-top: 2px; }
    .kpi-delta-pos { color: #00f2c3; font-weight: bold; }
    .kpi-delta-neg { color: #ff0055; font-weight: bold; }
    .kpi-delta-neutral { color: #888; }
    
    .scorecard {
        background-color: rgba(255,255,255,0.05); padding: 12px;
        border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 8px;
    }
    .score-title {
        color: #FFFFFF !important; font-size: 0.75em; font-weight: 800;
        text-transform: uppercase; letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# --- LISTE DES ACTIONS CAC 40 ---
tickers = {
    "LVMH": "MC.PA", "TOTALENERGIES": "TTE.PA", "L'OREAL": "OR.PA", "AIRBUS": "AIR.PA",
    "SCHNEIDER": "SU.PA", "AIR LIQUIDE": "AI.PA", "BNP PARIBAS": "BNP.PA", 
    "SOCIETE GENERALE": "GLE.PA", "AXA": "CS.PA", "VINCI": "DG.PA", 
    "SAFRAN": "SAF.PA", "HERMES": "RMS.PA", "KERING": "KER.PA", 
    "SANOFI": "SAN.PA", "ORANGE": "ORA.PA", "RENAULT": "RNO.PA", 
    "CAPGEMINI": "CAP.PA", "STMICRO": "STMPA.PA", "MICHELIN": "ML.PA", "DANONE": "BN.PA"
}

# =========================================================
# 2. CALCULS & DONNÉES (ROBUSTE)
# =========================================================
def calculate_advanced_stats(data, index_data):
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + gain/loss))
    
    # Moyennes Mobiles
    data['SMA50'] = data['Close'].rolling(50).mean()
    data['SMA200'] = data['Close'].rolling(200).mean()
    
    # Corrélation (Sécurisée)
    corr = 0
    if index_data is not None and not index_data.empty:
        try:
            # On aligne les dates
            common_idx = data.index.intersection(index_data.index)
            if len(common_idx) > 10:
                corr = data.loc[common_idx]['Close'].pct_change().corr(index_data.loc[common_idx]['Close'].pct_change())
        except:
            corr = 0 
    
    # Win Rate
    green_days = len(data[data['Close'] > data['Open']])
    total_days = len(data)
    win_rate = (green_days / total_days) * 100 if total_days > 0 else 0
    
    # Volume Trend
    vol_short = data['Volume'].tail(5).mean()
    vol_long = data['Volume'].tail(20).mean()
    vol_trend = ((vol_short - vol_long) / vol_long) * 100 if vol_long > 0 else 0
    
    # Ratio Sharpe
    returns = data['Close'].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    
    return data, {
        "corr_index": corr if not pd.isna(corr) else 0,
        "win_rate": win_rate, 
        "vol_trend": vol_trend,
        "sharpe": sharpe, 
        "volatility_day": returns.std() * 100
    }

@st.cache_data(ttl=3600)
def get_data(symbol, period="1y"):
    try:
        # 1. On récupère l'action (CRITIQUE)
        tk = yf.Ticker(symbol)
        data = tk.history(period=period)
        
        if data.empty: return None, None
        
        # 2. On essaie de récupérer le CAC 40 (OPTIONNEL)
        try:
            index_data = yf.Ticker("^FCHI").history(period=period)
        except:
            index_data = None # Si ça plante, on continue sans
        
        # 3. Calculs
        data, stats = calculate_advanced_stats(data, index_data)
        
        info = tk.fast_info
        # Sécurité sur le prix en direct
        stats['last'] = info.last_price if info.last_price else data['Close'].iloc[-1]
        stats['prev'] = info.previous_close if info.previous_close else data['Close'].iloc[-2]
        
        return data, stats
        
    except: return None, None

@st.cache_data(ttl=3600)
def get_global_market():
    global_data = []
    for n, s in tickers.items():
        try:
            t = yf.Ticker(s)
            hist = t.history(period="1mo") 
            fi = t.fast_info
            
            if fi.last_price is not None and not hist.empty:
                var = ((fi.last_price - fi.previous_close)/fi.previous_close)*100
                volatility = hist['Close'].pct_change().std() * 100 
                
                global_data.append({
                    "Action": n, "Symbole": s, 
                    "Prix": fi.last_price, "Variation": var, 
                    "Volume": fi.last_volume, "Volatilité": volatility
                })
        except: continue
    return pd.DataFrame(global_data)

# =========================================================
# 3. INTERFACE
# =========================================================
st.sidebar.title("🇫🇷 CAC 40 Pro")
page = st.sidebar.radio("Navigation", ["Vue Globale 🌍", "Deep Dive 🔍"])
if st.sidebar.button("🔄 Refresh"): st.cache_data.clear(); st.rerun()

c_up = "#00f2c3"
c_down = "#ff0055"

# --- PAGE 1 : VUE GLOBALE ---
if page == "Vue Globale 🌍":
    st.title("🌍 Vue d'ensemble (Paris)")
    with st.spinner("Analyse du CAC 40..."):
        df = get_global_market()
    
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        best = df.loc[df['Variation'].idxmax()]
        worst = df.loc[df['Variation'].idxmin()]
        c1.metric("TOP PERF", best['Action'], f"{best['Variation']:.2f}%")
        c2.metric("FLOP PERF", worst['Action'], f"{worst['Variation']:.2f}%")
        c3.metric("LEADER VOLUME", df.sort_values('Volume', ascending=False).iloc[0]['Action'])
        st.divider()
        
        c_chart, c_list = st.columns([2, 1])
        with c_chart:
            st.subheader("🎯 Matrice Risque / Gain")
            st.caption("ℹ️ HAUT = Gains | GAUCHE = Stable | DROITE = Risqué")
            
            fig = px.scatter(
                df, x="Volatilité", y="Variation", 
                size="Volume", color="Variation", 
                text="Symbole", hover_name="Action",
                color_continuous_scale="RdYlGn",
                labels={"Volatilité": "RISQUE (Volatilité)", "Variation": "GAIN (24h)"}
            )
            
            fig.update_traces(
                textposition='top center', 
                marker=dict(line=dict(width=1, color='white'), opacity=0.9),
                textfont=dict(color='white', size=11)
            )
            fig.update_layout(
                height=450, 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(255,255,255,0.05)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)
            )
            st.plotly_chart(fig, use_container_width=True)

        with c_list:
            st.subheader("Cotations")
            st.dataframe(
                df[['Action', 'Prix', 'Variation']].style.format({"Prix": "{:.2f}€", "Variation": "{:+.2f}%"})
                .background_gradient(subset=['Variation'], cmap="RdYlGn", vmin=-3, vmax=3), 
                height=450, 
                use_container_width=True
            )

# --- PAGE 2 : DEEP DIVE ---
elif page == "Deep Dive 🔍":
    st.sidebar.markdown("---")
    sel = st.sidebar.selectbox("Action :", list(tickers.keys()))
    sym = tickers[sel]
    
    time_frame = st.sidebar.radio("Période :", ["6 Mois", "1 An", "5 Ans"], index=1, horizontal=True)
    p_map = {"6 Mois": "6mo", "1 An": "1y", "5 Ans": "5y"}
    
    with st.spinner(f"Analyse de {sel}..."):
        hist, stats = get_data(sym, p_map[time_frame])
        
    if hist is None: 
        st.error("Erreur de données (Yahoo API). L'action semble inaccessible.")
        st.stop()

    # --- KPI HEADER ---
    var = ((stats['last'] - stats['prev'])/stats['prev'])*100
    k1, k2, k3, k4 = st.columns(4)
    
    def kpi_card(title, value, detail, detail_color_class, help_text):
        return f"""
        <div class="kpi-card">
            <div style="display: flex; justify-content: center; align-items: center;">
                <span class="kpi-title">{title}</span>
                <span class="tooltip-icon" title="{help_text}">?</span>
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-detail {detail_color_class}">{detail}</div>
        </div>
        """
    
    var_class = "kpi-delta-pos" if var > 0 else "kpi-delta-neg"
    k1.markdown(kpi_card("PRIX ACTUEL", f"{stats['last']:.2f} €", f"{var:+.2f}%", var_class, "Dernier cours coté à Paris."), unsafe_allow_html=True)
    
    sharpe = stats['sharpe']
    s_cls = "kpi-delta-pos" if sharpe > 1 else ("kpi-delta-neutral" if sharpe > 0 else "kpi-delta-neg")
    s_txt = "Excellent" if sharpe > 2 else ("Bon" if sharpe > 1 else "Risqué")
    k2.markdown(kpi_card("SHARPE", f"{sharpe:.2f}", s_txt, s_cls, "Rentabilité ajustée au risque. >1 est solide."), unsafe_allow_html=True)
    
    corr = stats['corr_index']
    c_cls = "kpi-delta-pos" if corr > 0.8 else "kpi-delta-neutral"
    c_txt = "Suit le CAC" if corr > 0.8 else "Indép."
    k3.markdown(kpi_card("CORR. CAC40", f"{corr:.2f}", c_txt, c_cls, "1 = Bouge exactement comme l'indice CAC 40."), unsafe_allow_html=True)
    
    rsi = hist['RSI'].iloc[-1]
    r_cls = "kpi-delta-neg" if rsi > 70 else ("kpi-delta-pos" if rsi < 30 else "kpi-delta-neutral")
    r_txt = "Surchauffe" if rsi > 70 else ("Opportunité" if rsi < 30 else "Neutre")
    k4.markdown(kpi_card("RSI", f"{rsi:.1f}", r_txt, r_cls, "Indicateur de tension (0-100)."), unsafe_allow_html=True)
    
    st.divider()

    col_score, col_chart = st.columns([1, 3])
    with col_score:
        st.subheader("📊 Scorecard")
        st.markdown(f"""
        <div class="scorecard">
            <span class="score-title">VOLATILITÉ <span class="tooltip-icon" title="Écart type journalier">?</span></span>
            <span style="font-size:1.3em; font-weight:bold;">{stats['volatility_day']:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        wr_color = c_up if stats['win_rate'] > 50 else c_down
        st.markdown(f"""
        <div class="scorecard">
            <span class="score-title">WIN RATE <span class="tooltip-icon" title="% de séances en hausse">?</span></span>
            <span style="font-size:1.3em; font-weight:bold; color:{wr_color}">{stats['win_rate']:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        vol_txt = "Hausse" if stats['vol_trend'] > 0 else "Baisse"
        vol_col = c_up if stats['vol_trend'] > 0 else c_down
        st.markdown(f"""
        <div class="scorecard">
            <span class="score-title">VOL TREND <span class="tooltip-icon" title="Volume récent vs moyen terme">?</span></span>
            <span style="font-size:1.3em; font-weight:bold; color:{vol_col}">{stats['vol_trend']:+.1f}%</span><br>
            <span style="font-size:0.7em; color:#ccc;">{vol_txt}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="scorecard">
            <span class="score-title">RANGE <span class="tooltip-icon" title="Plus haut / Plus bas période">?</span></span>
            <span style="font-size:0.8em; color:#888;">H:</span> <span style="color:{c_up}">{hist['High'].max():.2f}€</span><br>
            <span style="font-size:0.8em; color:#888;">L:</span> <span style="color:{c_down}">{hist['Low'].min():.2f}€</span>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        st.subheader(f"Graphique Technique ({time_frame})")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Prix"))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color="orange", width=1), name="SMA 50"))
        if len(hist) > 200:
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color="#00f2c3", width=2), name="SMA 200"))
        fig.update_layout(height=450, margin=dict(t=10, b=10, l=0, r=0), 
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis_rangeslider_visible=False, yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                          legend=dict(orientation="h", y=1.02))
        
        st.plotly_chart(fig, use_container_width=True)
