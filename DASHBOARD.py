import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf
import numpy as np

# =========================================================
# 1. CONFIGURATION ET STYLE
# =========================================================
st.set_page_config(layout="wide", page_title="Market Dashboard Ultimate")

st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }

    /* Style des cartes */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    h5 { color: #555; font-weight: 600; margin-bottom: 15px; }

    /* Astuce pour éviter que les metrics ne soient coupées */
    [data-testid="stMetricValue"] {
        font-size: 24px;
    }
    </style>
""", unsafe_allow_html=True)

tickers = {
    "LVMH": "MC.PA", "TOTAL": "TTE.PA", "L'OREAL": "OR.PA", "AIRBUS": "AIR.PA",
    "SCHNEIDER": "SU.PA", "AIR LIQUIDE": "AI.PA", "BNP PARIBAS": "BNP.PA",
    "AXA": "CS.PA", "VINCI": "DG.PA", "SAFRAN": "SAF.PA", "HERMES": "RMS.PA",
    "KERING": "KER.PA", "SANOFI": "SAN.PA", "ESSILOR": "EL.PA", "ORANGE": "ORA.PA",
    "RENAULT": "RNO.PA", "CAPGEMINI": "CAP.PA", "STMICRO": "STMPA.PA"
}


# =========================================================
# 2. FONCTIONS DATA (CACHE)
# =========================================================

@st.cache_data(ttl=3600)
def get_global_data():
    global_data = []
    for name, sym in tickers.items():
        try:
            t = yf.Ticker(sym)
            fi = t.fast_info
            last = fi.last_price
            prev = fi.previous_close
            var = ((last - prev) / prev) * 100 if prev else 0
            global_data.append({
                "Entreprise": name, "Symbole": sym, "Prix": last,
                "Variation %": var, "Market Cap": fi.market_cap, "Volume": fi.last_volume
            })
        except:
            continue
    return pd.DataFrame(global_data)


@st.cache_data(ttl=3600)
def get_multi_history(tickers_dict):
    symbols = list(tickers_dict.values())
    return yf.download(symbols, period="1y", progress=False)['Close']


@st.cache_data(ttl=3600)
def get_detail_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        fi = stock.fast_info
        info = {
            "last": fi.last_price, "prev": fi.previous_close,
            "open": fi.open, "mcap": fi.market_cap,
            "vol_avg": fi.three_month_average_volume
        }
        return hist, info
    except:
        return None, None


@st.cache_data(ttl=3600)
def get_index_data(symbol):
    try:
        return yf.Ticker(symbol).history(period="1mo")['Close']
    except:
        return None


# =========================================================
# 3. NAVIGATION
# =========================================================
st.sidebar.title("📱 Navigation")
page = st.sidebar.radio("Aller vers :", ["Vue Globale 🌍", "Vue Détaillée 🔍"])

if st.sidebar.button("🔄 Actualiser tout"):
    st.cache_data.clear()
    st.rerun()

# =========================================================
# PAGE 1 : VUE GLOBALE
# =========================================================
if page == "Vue Globale 🌍":
    st.title("🌍 Vue d'ensemble du CAC 40")

    with st.spinner("Analyse du marché en cours..."):
        df_global = get_global_data()
        df_history_all = get_multi_history(tickers)

    best_perf = df_global.loc[df_global['Variation %'].idxmax()]
    worst_perf = df_global.loc[df_global['Variation %'].idxmin()]
    total_cap = df_global['Market Cap'].sum() / 1e9

    col1, col2, col3 = st.columns(3)
    col1.metric("Top Performance 🚀", f"{best_perf['Entreprise']}", f"{best_perf['Variation %']:.2f} %")
    col2.metric("Moins bonne Perf 📉", f"{worst_perf['Entreprise']}", f"{worst_perf['Variation %']:.2f} %")
    col3.metric("Valorisation Totale", f"{total_cap:.2f} Mds €")

    st.divider()

    st.subheader("📈 Comparateur de Performance (Base 100)")
    selected_tickers = st.multiselect("Comparer :", list(tickers.keys()), default=["LVMH", "TOTAL", "AIRBUS"])

    if selected_tickers:
        fig_comp = go.Figure()
        for name in selected_tickers:
            sym = tickers[name]
            if sym in df_history_all.columns:
                series = df_history_all[sym].dropna()
                if not series.empty:
                    first_price = series.iloc[0]
                    normalized_series = ((series - first_price) / first_price) * 100
                    fig_comp.add_trace(go.Scatter(x=series.index, y=normalized_series, mode='lines', name=name))
        fig_comp.update_layout(hovermode="x unified", margin=dict(t=10, b=0, l=0, r=0), height=400,
                               yaxis_title="Variation (%)", paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False),
                               yaxis=dict(showgrid=True, gridcolor='#eee'))
        st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("📊 Tableau des Prix")
        st.dataframe(df_global.style.format({"Prix": "{:.2f} €", "Variation %": "{:+.2f} %", "Market Cap": "{:,.0f}"})
                     .background_gradient(subset=["Variation %"], cmap="RdYlGn", vmin=-3, vmax=3),
                     use_container_width=True, height=600,
                     column_config={"Volume": st.column_config.ProgressColumn("Volume", format="%d", min_value=0,
                                                                              max_value=int(df_global['Volume'].max())),
                                    "Market Cap": st.column_config.NumberColumn("Market Cap", format="%.2e €")})
    with c2:
        st.subheader("🗺️ Carte (Market Cap)")
        fig_tree = px.treemap(df_global, path=['Entreprise'], values='Market Cap', color='Variation %',
                              color_continuous_scale=['#e74c3c', '#ecf0f1', '#2ecc71'], color_continuous_midpoint=0,
                              custom_data=['Prix', 'Variation %'])
        fig_tree.update_traces(textposition="middle center", texttemplate="%{label}<br>%{customdata[1]:.2f}%",
                               hovertemplate='<b>%{label}</b><br>Prix: %{customdata[0]:.2f}€<br>Var: %{customdata[1]:.2f}%')
        fig_tree.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=600)
        st.plotly_chart(fig_tree, use_container_width=True)

# =========================================================
# PAGE 2 : VUE DÉTAILLÉE (CORRIGÉE POUR MARKET CAP)
# =========================================================
elif page == "Vue Détaillée 🔍":

    st.sidebar.markdown("---")
    st.sidebar.subheader("Sélection Focus")
    selected_name = st.sidebar.selectbox("Choisir une entreprise :", list(tickers.keys()))
    symbol = tickers[selected_name]

    with st.spinner(f"Chargement des données de {selected_name}..."):
        hist, info = get_detail_data(symbol)
        cac40 = get_index_data("^FCHI")
        sp500 = get_index_data("^GSPC")
        bitcoin = get_index_data("BTC-EUR")

    if hist is None:
        st.error("Erreur technique ou données indisponibles.")
        st.stop()


    # --- Fonctions Graphiques ---
    def plot_donut_volume(vol_today, vol_avg):
        values = [vol_today, max(0, vol_avg - vol_today)]
        colors = ['#2ecc71', '#ecf0f1']
        fig = go.Figure(
            data=[go.Pie(labels=['Vol Day', 'Rest'], values=values, hole=.6, marker=dict(colors=colors), sort=False)])
        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=250, showlegend=False)
        vol_str = f"{vol_today / 1e6:.1f}M" if vol_today > 1e6 else f"{vol_today / 1e3:.0f}K"
        fig.add_annotation(text=f"<b>Volume</b><br>{vol_str}", x=0.5, y=0.5, font_size=16, showarrow=False)
        return fig


    def plot_performance_bars(hist):
        last = hist['Close'].iloc[-1]

        def get_var(days):
            if len(hist) > days: return ((last - hist['Close'].iloc[-days]) / hist['Close'].iloc[-days]) * 100
            return 0

        perfs = [{'Label': '1 Sem', 'V': get_var(5)}, {'Label': '1 Mois', 'V': get_var(20)},
                 {'Label': '6 Mois', 'V': get_var(120)}, {'Label': '1 An', 'V': get_var(250)}]
        colors = ['#2ecc71' if p['V'] >= 0 else '#e74c3c' for p in perfs]
        fig = go.Figure(
            go.Bar(x=[p['V'] for p in perfs], y=[p['Label'] for p in perfs], orientation='h', marker_color=colors,
                   text=[f"{p['V']:+.1f}%" for p in perfs], textposition='auto'))
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250, xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig


    def plot_candlestick_real(df):
        df['MA20'] = df['Close'].rolling(window=20).mean()
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='Moy 20j'))
        fig.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=300, xaxis_rangeslider_visible=False,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
        return fig


    def plot_sparkline_real(series, color):
        if series is None: return go.Figure()
        fig = go.Figure(go.Scatter(y=series.values, mode='lines', line=dict(color=color, width=2)))
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=50, xaxis=dict(visible=False),
                          yaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig


    # --- DASHBOARD LAYOUT ---
    st.title(f"📊 Analyse Focus : {selected_name}")

    col_left, col_mid, col_right = st.columns([1, 1.5, 1.5], gap="medium")

    # --- Colonne Gauche ---
    with col_left:
        with st.container():
            st.write("##### Activité (Volume)")
            st.plotly_chart(plot_donut_volume(hist['Volume'].iloc[-1], info['vol_avg']), use_container_width=True,
                            config={'displayModeBar': False})
        with st.container():
            st.write("##### Derniers Jours")
            last_days = hist.tail(5).iloc[::-1].copy()
            last_days['Variation'] = last_days['Close'].pct_change(-1) * 100
            df_show = last_days[['Close', 'Variation']].copy()
            df_show['Close'] = df_show['Close'].map('{:.2f} €'.format)
            df_show['Variation'] = df_show['Variation'].map('{:+.2f} %'.format)
            st.table(df_show)

    # --- Colonne Milieu (MODIFIÉE POUR MARKET CAP) ---
    with col_mid:
        with st.container():
            # 1. On met les Indicateurs Clés SUR UNE LIGNE SÉPARÉE pour avoir de la place
            st.write("##### Indicateurs Clés")
            kpi1, kpi2, kpi3 = st.columns(3)

            var_day = ((info['last'] - info['prev']) / info['prev']) * 100
            kpi1.metric("Prix", f"{info['last']:.2f}€")
            kpi2.metric("Var Jour", f"{var_day:+.2f}%", delta=f"{var_day:+.2f}%")
            kpi3.metric("Market Cap", f"{info['mcap'] / 1e9:.1f} B€")  # Plus de coupure ici !

            st.divider()

            # 2. Le graphique de performance prend toute la largeur maintenant
            st.write("##### Performances Historiques")
            st.plotly_chart(plot_performance_bars(hist), use_container_width=True, config={'displayModeBar': False})

        with st.container():
            st.write(f"##### Courbe : {selected_name}")
            fig_line = go.Figure(go.Scatter(x=hist.index, y=hist['Close'], fill='tozeroy', line=dict(color='#3498db'),
                                            fillcolor='rgba(52, 152, 219, 0.1)'))
            fig_line.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=200, paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False),
                                   yaxis=dict(showgrid=True, gridcolor='#eee'))
            st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})

    # --- Colonne Droite ---
    with col_right:
        with st.container():
            st.write("##### Analyse Technique")
            st.plotly_chart(plot_candlestick_real(hist), use_container_width=True, config={'displayModeBar': False})
        with st.container():
            st.write("##### Comparaison Marchés")
            for name, data_idx in [("CAC 40", cac40), ("S&P 500", sp500), ("Bitcoin", bitcoin)]:
                c_spark = st.columns([1, 3])
                c_spark[0].write(f"**{name}**")
                if data_idx is not None:
                    col = '#2ecc71' if data_idx.iloc[-1] > data_idx.iloc[0] else '#e74c3c'
                    c_spark[1].plotly_chart(plot_sparkline_real(data_idx, col), use_container_width=True,
                                            config={'displayModeBar': False})

                st.divider()
