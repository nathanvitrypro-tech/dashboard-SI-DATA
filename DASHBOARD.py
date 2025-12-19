import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf
import numpy as np

# =========================================================
# 1. CONFIGURATION ET STYLE MODERN (GLASSMORPHISM & NEON)
# =========================================================
st.set_page_config(layout="wide", page_title="Market Dashboard Ultimate", page_icon="⚡")

st.markdown("""
    <style>
    /* Import police moderne */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
        color: #e0e0e0; /* Texte clair par défaut */
    }

    /* FOND GÉNÉRAL - DÉGRADÉ SOMBRE MODERNE */
    .stApp {
        background: rgb(10,10,30);
        background: linear-gradient(160deg, rgba(10,10,30,1) 0%, rgba(25,15,45,1) 50%, rgba(10,30,50,1) 100%);
        background-attachment: fixed;
    }
    
    /* BARRE LATÉRALE - STYLE NEON */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8); /* Semi-transparent */
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    section[data-testid="stSidebar"] h1, h2, h3, p, div, span { color: #ffffff !important; }
    
    /* CARTES GLASSMORPHISM (L'effet "Verre Dépoli") */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: rgba(255, 255, 255, 0.05); /* Très transparent */
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1); /* Bordure subtile */
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); /* Ombre portée */
        backdrop-filter: blur(12px); /* Le flou magique */
        margin-bottom: 20px;
    }
    
    /* TITRES AVEC EFFET NEON SUBTIL */
    h1, h2, h3 { 
        color: #ffffff; 
        font-weight: 800; 
        text-shadow: 0 0 10px rgba(0, 153, 255, 0.3);
    }
    h5 { 
        color: #00f2c3; /* Cyan électrique */
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 1.5px; 
        font-size: 0.85rem;
    }
    
    /* CHIFFRES EN GROS (MÉTRIQUES) AVEC DÉGRADÉ */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(45deg, #00f2c3, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    [data-testid="stMetricLabel"] { color: #b0b0b0; font-size: 1rem; }
    [data-testid="stMetricDelta"] svg { color: #00f2c3 !important; } /* Flèche verte en cyan */
    
    /* PERSONNALISATION DES BOUTONS RADIO ET SELECTBOX */
    div[role="radiogroup"] label > div:first-child {
        background-color: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
    }
    div[data-baseweb="select"] > div {
         background-color: rgba(255, 255, 255, 0.05);
         color: white;
         border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* TABLEAUX STYLE SOMBRE */
    [data-testid="stDataFrame"] { box-shadow: none; }
    [data-testid="stDataFrame"] div[class*="css"] {
        background-color: transparent !important;
        color: #e0e0e0 !important;
    }
    </style>
""", unsafe_allow_html=True)

tickers = {
    "LVMH": "MC.PA", "TOTAL": "TTE.PA", "L'OREAL": "OR.PA", "AIRBUS": "AIR.PA",
    "SCHNEIDER": "SU.PA", "AIR LIQUIDE": "AI.PA", "BNP PARIBAS": "BNP.PA", 
    "SOCIETE GENERALE": "GLE.PA", "VEOLIA": "VIE.PA",
    "AXA": "CS.PA", "VINCI": "DG.PA", "SAFRAN": "SAF.PA", "HERMES": "RMS.PA", 
    "KERING": "KER.PA", "SANOFI": "SAN.PA", "ESSILOR": "EL.PA", "ORANGE": "ORA.PA",
    "RENAULT": "RNO.PA", "CAPGEMINI": "CAP.PA", "STMICRO": "STMPA.PA"
}

# =========================================================
# 2. FONCTIONS DE RÉCUPÉRATION (CACHE)
# =========================================================
# (Pas de changement ici, le cache fonctionne)
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
        except: continue
    return pd.DataFrame(global_data)

@st.cache_data(ttl=3600)
def get_multi_history(tickers_dict, period="1y"):
    symbols = list(tickers_dict.values())
    data = yf.download(symbols, period=period, progress=False)['Close']
    return data

@st.cache_data(ttl=3600)
def get_detail_data(symbol, period="1y"):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        inf = stock.info
        dividend_yield = inf.get('dividendYield', 0)
        trailing_pe = inf.get('trailingPE', 0)
        fi = stock.fast_info
        info_dict = {"last": fi.last_price, "prev": fi.previous_close, "mcap": fi.market_cap, "dividend": dividend_yield, "per": trailing_pe}
        return hist, info_dict
    except: return None, None

@st.cache_data(ttl=3600)
def get_historical_data(symbol, period="1y"):
    try: return yf.Ticker(symbol).history(period=period)['Close']
    except: return None

@st.cache_data(ttl=3600)
def get_index_data(symbol):
    try: return yf.Ticker(symbol).history(period="1mo")['Close']
    except: return None

# =========================================================
# 3. NAVIGATION
# =========================================================
st.sidebar.markdown("## ⚡ Navigation")
page = st.sidebar.radio("Menu :", ["Vue Globale 🌍", "Vue Détaillée 🔍"], label_visibility="collapsed")
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Actualiser les données"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.success("Design Modern v2.0\nMode Sombre Actif")

# =========================================================
# CONFIGURATION COULEURS MODERNES
# =========================================================
neon_cyan = "#00f2c3"
neon_pink = "#ff0055"
neon_blue = "#0099ff"
dark_bg = "rgba(0,0,0,0)" # Fond transparent pour les graphs
grid_color = "rgba(255,255,255,0.1)"
text_color = "#ffffff"

# =========================================================
# PAGE 1 : VUE GLOBALE
# =========================================================
if page == "Vue Globale 🌍":
    st.title("Vue d'ensemble du CAC 40 🇫🇷")
    
    with st.spinner("Chargement du marché..."):
        df_global = get_global_data()
        
    best_perf = df_global.loc[df_global['Variation %'].idxmax()]
    worst_perf = df_global.loc[df_global['Variation %'].idxmin()]
    total_cap = df_global['Market Cap'].sum() / 1e9
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Top Performance 🚀", f"{best_perf['Entreprise']}", f"{best_perf['Variation %']:.2f} %", delta=f"{best_perf['Variation %']:.2f} %")
    col2.metric("Moins bonne Perf 📉", f"{worst_perf['Entreprise']}", f"{worst_perf['Variation %']:.2f} %", delta=f"{worst_perf['Variation %']:.2f} %")
    col3.metric("Valorisation Totale", f"{total_cap:.2f} Mds €")
    
    st.divider()

    st.subheader("📈 Comparateur de Performance (Base 100)")
    col_conf1, col_conf2 = st.columns([1, 2])
    with col_conf1:
        time_period_global = st.radio("Période :", ["1 Mois", "3 Mois", "6 Mois", "1 An", "5 Ans", "10 Ans"], index=3, horizontal=True)
        period_map_global = {"1 Mois": "1mo", "3 Mois": "3mo", "6 Mois": "6mo", "1 An": "1y", "5 Ans": "5y", "10 Ans": "10y"}
        selected_yahoo_period_global = period_map_global[time_period_global]
    with col_conf2:
        selected_tickers = st.multiselect("Comparer :", list(tickers.keys()), default=["LVMH", "TOTAL", "AIRBUS"])
    
    df_history_dynamic = get_multi_history(tickers, period=selected_yahoo_period_global)
    
    if selected_tickers:
        fig_comp = go.Figure()
        colors_cycle = [neon_cyan, neon_blue, "#ffcc00", neon_pink, "#ad00ff"]
        for i, name in enumerate(selected_tickers):
            sym = tickers[name]
            if sym in df_history_dynamic.columns:
                series = df_history_dynamic[sym].dropna()
                if not series.empty:
                    first_price = series.iloc[0]
                    normalized_series = ((series - first_price) / first_price) * 100
                    # Style Ligne Neon
                    fig_comp.add_trace(go.Scatter(
                        x=series.index, y=normalized_series, mode='lines', name=name, 
                        line=dict(width=3, color=colors_cycle[i % len(colors_cycle)]),
                        hovertemplate='%{y:.2f}%'
                    ))
        
        fig_comp.update_layout(
            template="plotly_dark", # Thème sombre natif
            hovermode="x unified", margin=dict(t=10, b=0, l=0, r=0), height=450,
            yaxis_title="Performance (%)", paper_bgcolor=dark_bg, plot_bgcolor=dark_bg,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor=grid_color),
            legend=dict(orientation="h", y=1.02, xanchor="right", x=1, font=dict(color=text_color)),
            font=dict(color=text_color)
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Sélectionnez au moins une entreprise.")
    
    st.divider()
    
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("📊 Tableau des Prix")
        # Tableau stylisé sombre avec dégradé moderne (Cyan à Magenta)
        st.dataframe(df_global.style.format({"Prix": "{:.2f} €", "Variation %": "{:+.2f} %", "Market Cap": "{:,.0f}"})
                     .background_gradient(subset=["Variation %"], cmap="cool_r", vmin=-3, vmax=3), # cool_r = dégradé cyan/magenta
                     use_container_width=True, height=600,
                     column_config={"Volume": st.column_config.ProgressColumn("Volume", format="%d", min_value=0, max_value=int(df_global['Volume'].max())),
                                    "Market Cap": st.column_config.NumberColumn("Market Cap", format="%.2e €")})
    with c2:
        st.subheader("🗺️ Carte du Marché")
        # Treemap avec palette moderne
        fig_tree = px.treemap(df_global, path=['Entreprise'], values='Market Cap', color='Variation %',
                              color_continuous_scale=['#ff0055', '#1a1a2e', '#00f2c3'], color_continuous_midpoint=0, # Rose -> Sombre -> Cyan
                              custom_data=['Prix', 'Variation %'])
        fig_tree.update_traces(textposition="middle center", texttemplate="<b>%{label}</b><br>%{customdata[1]:.2f}%",
                               hovertemplate='<b>%{label}</b><br>Prix: %{customdata[0]:.2f}€<br>Var: %{customdata[1]:.2f}%',
                               marker=dict(line=dict(color=dark_bg, width=2)))
        fig_tree.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=600, paper_bgcolor=dark_bg, font=dict(color=text_color),
                               coloraxis_colorbar=dict(title="Var %", tickfont=dict(color=text_color)))
        st.plotly_chart(fig_tree, use_container_width=True)

# =========================================================
# PAGE 2 : VUE DÉTAILLÉE
# =========================================================
elif page == "Vue Détaillée 🔍":
    
    st.sidebar.markdown("### Sélection Focus")
    selected_name = st.sidebar.selectbox("Entreprise :", list(tickers.keys()), label_visibility="collapsed")
    symbol = tickers[selected_name]
    st.sidebar.markdown("### Période d'analyse")
    time_period_detail = st.sidebar.radio("Durée :", ["1 Mois", "3 Mois", "6 Mois", "1 An", "2 Ans", "5 Ans", "10 Ans"], index=3, label_visibility="collapsed")
    period_map_detail = {"1 Mois": "1mo", "3 Mois": "3mo", "6 Mois": "6mo", "1 An": "1y", "2 Ans": "2y", "5 Ans": "5y", "10 Ans": "10y"}
    selected_yahoo_period_detail = period_map_detail[time_period_detail]

    with st.spinner(f"Chargement des données ({time_period_detail}) de {selected_name}..."):
        hist, info = get_detail_data(symbol, period=selected_yahoo_period_detail)
        cac40_hist_period = get_historical_data("^FCHI", period=selected_yahoo_period_detail)
        cac40 = get_index_data("^FCHI")
        sp500 = get_index_data("^GSPC")
        bitcoin = get_index_data("BTC-EUR")

    if hist is None or hist.empty:
        st.error("Données indisponibles.").stop()

    # --- FONCTIONS GRAPHIQUES NEON ---
    def plot_dividend_gauge(yield_val):
        val = yield_val * 100 if yield_val and yield_val < 0.5 else (yield_val if yield_val else 0)
        # Jauge style "Arc reactor"
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = val, title = {'text': "Rendement Dividende", 'font': {'color': text_color}},
            number = {'suffix': "%", 'font': {'size': 30, 'color': neon_cyan}},
            gauge = {'axis': {'range': [None, 8], 'tickcolor': text_color}, 
                     'bar': {'color': neon_cyan, 'thickness': 1}, # Barre fine cyan
                     'bgcolor': "rgba(255,255,255,0.05)", 'borderwidth': 0,
                     'steps': [{'range': [0, 8], 'color': 'rgba(0, 242, 195, 0.1)'}]} # Fond brillant subtil
        ))
        fig.update_layout(margin=dict(t=40, b=10, l=30, r=30), height=200, paper_bgcolor=dark_bg, font={'color': text_color})
        return fig

    def plot_performance_bars(hist):
        last = hist['Close'].iloc[-1]
        def get_var(days):
            if len(hist) > days: return ((last - hist['Close'].iloc[-days]) / hist['Close'].iloc[-days]) * 100
            return 0
        perfs = [{'Label': '1 S', 'V': get_var(5)}, {'Label': '1 M', 'V': get_var(20)}, 
                 {'Label': '3 M', 'V': get_var(60)}, {'Label': '6 M', 'V': get_var(120)}]
        # Couleurs Neon Cyan vs Pink
        colors = [neon_cyan if p['V'] >= 0 else neon_pink for p in perfs]
        
        fig = go.Figure(go.Bar(
            x=[p['V'] for p in perfs], y=[p['Label'] for p in perfs], 
            orientation='h', marker_color=colors, 
            text=[f"{p['V']:+.1f}%" for p in perfs], textposition='auto', 
            name="Perf", textfont=dict(color='white')
        ))
        fig.update_layout(
            template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0), height=250, 
            xaxis=dict(showgrid=False, zerolinecolor=grid_color), yaxis=dict(showgrid=False, tickfont=dict(color=text_color)),
            paper_bgcolor=dark_bg, plot_bgcolor=dark_bg, showlegend=False, font={'color': text_color}
        )
        return fig

    def plot_price_vs_benchmark(stock_series, benchmark_series, stock_name, benchmark_name="CAC 40"):
        df = pd.concat([stock_series, benchmark_series], axis=1, join='inner')
        df.columns = ['Stock', 'Benchmark']
        df = (df / df.iloc[0]) * 100
        
        fig = go.Figure()
        # Area chart avec dégradé sous la courbe
        fig.add_trace(go.Scatter(x=df.index, y=df['Stock'], mode='lines', name=stock_name, line=dict(color=neon_blue, width=3), fill='tozeroy', fillcolor='rgba(0, 153, 255, 0.1)'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Benchmark'], mode='lines', name=benchmark_name, line=dict(color='#b0b0b0', width=2, dash='dot')))
        
        fig.update_layout(
            template="plotly_dark", title=dict(text=f"Performance vs {benchmark_name} (Base 100)", font=dict(color=text_color)),
            margin=dict(t=40, b=0, l=0, r=0), height=250,
            paper_bgcolor=dark_bg, plot_bgcolor=dark_bg,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor=grid_color),
            showlegend=True, legend=dict(orientation="h", y=1.15, font=dict(color=text_color)), font={'color': text_color}
        )
        return fig

    def plot_candlestick_real(df):
        window = 50 if len(df) > 200 else (20 if len(df) > 50 else 5)
        df['MA'] = df['Close'].rolling(window=window).mean()
        fig = go.Figure()
        # Bougies Neon (Vert Cyan / Rouge Pink)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Prix',
                                     increasing_line_color=neon_cyan, decreasing_line_color=neon_pink))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA'], line=dict(color='#ffcc00', width=2), name=f'Moy. {window}j'))
        
        fig.update_layout(
            template="plotly_dark", margin=dict(t=10, b=20, l=0, r=0), height=300, 
            xaxis_rangeslider_visible=False, paper_bgcolor=dark_bg, 
            plot_bgcolor=dark_bg, showlegend=True, 
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center", font=dict(color=text_color)),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor=grid_color), font={'color': text_color}
        )
        return fig
    
    def plot_sparkline_real(series, color):
        if series is None: return go.Figure()
        # Sparkline avec lueur
        fig = go.Figure(go.Scatter(y=series.values, mode='lines', fill='tozeroy', line=dict(color=color, width=2), fillcolor=color.replace(')', ', 0.2)').replace('rgb', 'rgba')))
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=50, xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor=dark_bg, plot_bgcolor=dark_bg, showlegend=False)
        return fig

    # --- MISE EN PAGE ---
    st.title(f"📊 Analyse : {selected_name}")

    col_left, col_mid, col_right = st.columns([1, 1.5, 1.5], gap="medium")

    with col_left:
        with st.container():
            st.write("##### Rendement & Valorisation")
            st.plotly_chart(plot_dividend_gauge(info['dividend']), use_container_width=True, config={'displayModeBar': False})
            st.divider()
            per_val = info['per']
            per_str = f"{per_val:.1f}x" if per_val and per_val > 0 else "N/A"
            st.metric("PER (Cours/Bénéfice)", per_str, delta_color="off")

        with st.container():
            st.write("##### Derniers Jours")
            days_to_show = min(5, len(hist))
            last_days = hist.tail(days_to_show).iloc[::-1].copy()
            last_days['Variation'] = last_days['Close'].pct_change(-1) * 100
            df_show = last_days[['Close', 'Variation']].copy()
            df_show['Close'] = df_show['Close'].map('{:.2f} €'.format)
            df_show['Variation'] = df_show['Variation'].map('{:+.2f} %'.format)
            # Petit hack pour colorer le tableau en mode sombre
            st.dataframe(df_show.style.applymap(lambda x: f'color: {neon_cyan}' if '+' in x else (f'color: {neon_pink}' if '-' in x else ''), subset=['Variation']), use_container_width=True)

    with col_mid:
        with st.container():
            st.write("##### Indicateurs Clés")
            kpi1, kpi2, kpi3 = st.columns(3)
            var_day = ((info['last'] - info['prev']) / info['prev']) * 100
            kpi1.metric("Prix", f"{info['last']:.2f}€", delta_color="off")
            kpi2.metric("Var Jour", f"{var_day:+.2f}%", delta=f"{var_day:+.2f}%")
            kpi3.metric("Capitalisation", f"{info['mcap']/1e9:.1f} B€", delta_color="off")
            st.divider()
            st.write("##### Performances Historiques")
            st.plotly_chart(plot_performance_bars(hist), use_container_width=True, config={'displayModeBar': False})

        with st.container():
            if cac40_hist_period is not None and not cac40_hist_period.empty:
                fig_vs_bench = plot_price_vs_benchmark(hist['Close'], cac40_hist_period, selected_name)
                st.plotly_chart(fig_vs_bench, use_container_width=True, config={'displayModeBar': False})
            else:
                st.warning("Benchmark indisponible.")

    with col_right:
        with st.container():
            st.write(f"##### Analyse Technique ({time_period_detail})")
            st.plotly_chart(plot_candlestick_real(hist), use_container_width=True, config={'displayModeBar': False})
        with st.container():
            st.write("##### Comparaison Marchés (1 mois)")
            for name, data_idx in [("CAC 40", cac40), ("S&P 500", sp500), ("Bitcoin", bitcoin)]:
                c_spark = st.columns([1, 3])
                c_spark[0].write(f"**{name}**")
                if data_idx is not None:
                    col = neon_cyan if data_idx.iloc[-1] > data_idx.iloc[0] else neon_pink
                    c_spark[1].plotly_chart(plot_sparkline_real(data_idx, col), use_container_width=True, config={'displayModeBar': False})
                st.divider()
