import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf
import numpy as np

# =========================================================
# 1. CONFIGURATION ET STYLE (CSS)
# =========================================================
st.set_page_config(layout="wide", page_title="Market Dashboard Ultimate")

st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    h5 { color: #555; font-weight: 600; margin-bottom: 15px; }
    [data-testid="stMetricValue"] { font-size: 24px; }
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
        
        # --- AJOUT: DONNÉES FINANCIÈRES (BILANS) ---
        financials = stock.financials
        # -------------------------------------------

        data_points = {
            "dividend": inf.get('dividendYield', 0),
            "per": inf.get('trailingPE', 0),
            "targetMeanPrice": inf.get('targetMeanPrice', 0),
            "recommendationKey": inf.get('recommendationKey', 'N/A'),
            "profitMargins": inf.get('profitMargins', 0), 
            "beta": inf.get('beta', 0), 
            "debtToEquity": inf.get('debtToEquity', 0),
            "sector": inf.get('sector', 'N/A')
        }

        fi = stock.fast_info
        
        info_dict = {
            "last": fi.last_price, 
            "prev": fi.previous_close,
            "mcap": fi.market_cap,
            **data_points
        }
        return hist, info_dict, financials
    except Exception as e:
        return None, None, None

@st.cache_data(ttl=3600)
def get_historical_data(symbol, period="1y"):
    try: return yf.Ticker(symbol).history(period=period)['Close']
    except: return None

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
        
    best_perf = df_global.loc[df_global['Variation %'].idxmax()]
    worst_perf = df_global.loc[df_global['Variation %'].idxmin()]
    total_cap = df_global['Market Cap'].sum() / 1e9
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Top Performance 🚀", f"{best_perf['Entreprise']}", f"{best_perf['Variation %']:.2f} %")
    col2.metric("Moins bonne Perf 📉", f"{worst_perf['Entreprise']}", f"{worst_perf['Variation %']:.2f} %")
    col3.metric("Valorisation Totale", f"{total_cap:.2f} Mds €")
    
    st.divider()

    st.subheader("📈 Comparateur de Performance (Base 100)")
    col_conf1, col_conf2 = st.columns([1, 2])
    
    with col_conf1:
        time_period_global = st.radio("Période Globale :", ["1 Mois", "3 Mois", "6 Mois", "1 An", "5 Ans", "10 Ans"], index=3, horizontal=True)
        period_map_global = {"1 Mois": "1mo", "3 Mois": "3mo", "6 Mois": "6mo", "1 An": "1y", "5 Ans": "5y", "10 Ans": "10y"}
        selected_yahoo_period_global = period_map_global[time_period_global]

    with col_conf2:
        selected_tickers = st.multiselect("Comparer :", list(tickers.keys()), default=["LVMH", "TOTAL", "AIRBUS"])
    
    df_history_dynamic = get_multi_history(tickers, period=selected_yahoo_period_global)
    
    if selected_tickers:
        fig_comp = go.Figure()
        for name in selected_tickers:
            sym = tickers[name]
            if sym in df_history_dynamic.columns:
                series = df_history_dynamic[sym].dropna()
                if not series.empty:
                    first_price = series.iloc[0]
                    normalized_series = ((series - first_price) / first_price) * 100
                    fig_comp.add_trace(go.Scatter(x=series.index, y=normalized_series, mode='lines', name=name, hovertemplate='%{y:.2f}%'))
        fig_comp.update_layout(hovermode="x unified", margin=dict(t=10, b=0, l=0, r=0), height=450,
                               yaxis_title="Performance (%)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#eee'),
                               legend=dict(orientation="h", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Sélectionnez au moins une entreprise.")
    
    st.divider()
    
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("📊 Tableau des Prix")
        st.dataframe(df_global.style.format({"Prix": "{:.2f} €", "Variation %": "{:+.2f} %", "Market Cap": "{:,.0f}"})
                     .background_gradient(subset=["Variation %"], cmap="RdYlGn", vmin=-3, vmax=3),
                     use_container_width=True, height=600,
                     column_config={"Volume": st.column_config.ProgressColumn("Volume", format="%d", min_value=0, max_value=int(df_global['Volume'].max())),
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
# PAGE 2 : VUE DÉTAILLÉE
# =========================================================
elif page == "Vue Détaillée 🔍":
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sélection Focus")
    selected_name = st.sidebar.selectbox("Choisir une entreprise :", list(tickers.keys()))
    symbol = tickers[selected_name]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Période d'analyse")
    time_period_detail = st.sidebar.radio("Choisir la durée :", ["1 Mois", "3 Mois", "6 Mois", "1 An", "2 Ans", "5 Ans", "10 Ans"], index=3)
    period_map_detail = {"1 Mois": "1mo", "3 Mois": "3mo", "6 Mois": "6mo", "1 An": "1y", "2 Ans": "2y", "5 Ans": "5y", "10 Ans": "10y"}
    selected_yahoo_period_detail = period_map_detail[time_period_detail]

    with st.spinner(f"Chargement des données ({time_period_detail}) de {selected_name}..."):
        # Modification de l'unpacking ici (3 valeurs retournées maintenant)
        hist, info, financials = get_detail_data(symbol, period=selected_yahoo_period_detail)
        cac40_hist_period = get_historical_data("^FCHI", period=selected_yahoo_period_detail)

    if hist is None or hist.empty:
        st.error("Données indisponibles.")
        st.stop()

    # --- FONCTIONS GRAPHIQUES ---
    def plot_dividend_gauge(yield_val):
        if yield_val is None: val = 0
        else: val = yield_val * 100 if yield_val < 0.5 else yield_val
            
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = val, title = {'text': "Rendement Dividende"},
            number = {'suffix': "%", 'font': {'size': 26}},
            gauge = {'axis': {'range': [None, 8]}, 'bar': {'color': "#2ecc71"},
                     'steps': [{'range': [0, 2], 'color': '#ecf0f1'}, {'range': [2, 5], 'color': '#d5f5e3'}, {'range': [5, 8], 'color': '#abebc6'}]}
        ))
        fig.update_layout(margin=dict(t=30, b=10, l=30, r=30), height=200, paper_bgcolor='rgba(0,0,0,0)')
        return fig

    def plot_performance_bars(hist):
        last = hist['Close'].iloc[-1]
        def get_var(days):
            if len(hist) > days: return ((last - hist['Close'].iloc[-days]) / hist['Close'].iloc[-days]) * 100
            return 0
        perfs = [{'Label': '1 Sem', 'V': get_var(5)}, {'Label': '1 Mois', 'V': get_var(20)}, 
                 {'Label': '3 Mois', 'V': get_var(60)}, {'Label': '6 Mois', 'V': get_var(120)}]
        colors = ['#2ecc71' if p['V'] >= 0 else '#e74c3c' for p in perfs]
        
        fig = go.Figure(go.Bar(
            x=[p['V'] for p in perfs], y=[p['Label'] for p in perfs], 
            orientation='h', marker_color=colors, 
            text=[f"{p['V']:+.1f}%" for p in perfs], textposition='auto', 
            name="Performance (%)"
        ))
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0), height=250, 
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), 
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
            showlegend=True, legend=dict(orientation="h", y=-0.1)
        )
        return fig

    def plot_price_vs_benchmark(stock_series, benchmark_series, stock_name, benchmark_name="CAC 40"):
        df = pd.concat([stock_series, benchmark_series], axis=1, join='inner')
        df.columns = ['Stock', 'Benchmark']
        df = (df / df.iloc[0]) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Stock'], mode='lines', name=stock_name, line=dict(color='#3498db', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Benchmark'], mode='lines', name=benchmark_name, line=dict(color='#95a5a6', width=2, dash='dot')))
        
        fig.update_layout(
            title=f"Performance relative vs {benchmark_name} (Base 100)",
            margin=dict(t=40, b=0, l=0, r=0), height=250,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#eee', title='Performance (Base 100)'),
            showlegend=True, legend=dict(orientation="h", y=1.1)
        )
        return fig

    def plot_candlestick_real(df):
        window = 50 if len(df) > 200 else (20 if len(df) > 50 else 5)
        df['MA'] = df['Close'].rolling(window=window).mean()
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Prix (OHLC)'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA'], line=dict(color='orange', width=1), name=f'Moyenne {window}j'))
        
        fig.update_layout(
            margin=dict(t=10, b=20, l=0, r=0), height=300, 
            xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', showlegend=True, 
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center")
        )
        return fig
    
    def plot_price_vs_target_bar(current, target):
        if not target or target == 0: return go.Figure()
        upside = ((target - current) / current) * 100
        color_target = "#2ecc71" if target > current else "#e74c3c"
        
        x_vals = ["Prix Actuel", "Objectif Analystes"]
        y_vals = [current, target]
        colors = ["#3498db", color_target]
        
        fig = go.Figure(go.Bar(
            x=y_vals, y=x_vals, orientation='h',
            marker_color=colors, text=[f"{current:.2f}€", f"{target:.2f}€"],
            textposition='auto'
        ))
        fig.update_layout(
            title=dict(text=f"Potentiel: {upside:+.2f}%", font=dict(color=color_target, size=18)),
            margin=dict(t=40, b=0, l=0, r=0), height=150,
            xaxis=dict(showgrid=False, visible=False), yaxis=dict(showgrid=False),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False
        )
        return fig

    # --- NOUVEAU : GRAPHIQUE FINANCIER (REVENUS/PROFITS) ---
    def plot_financial_growth(financials):
        if financials is None or financials.empty:
            return go.Figure()
        
        try:
            fin_T = financials.T
            fin_T = fin_T.sort_index().tail(4) # Dernières 4 années
            dates = fin_T.index.strftime('%Y')
            
            # Recherche flexible des clés
            rev_key = next((k for k in ['Total Revenue', 'TotalRevenue', 'Revenue'] if k in financials.index), None)
            inc_key = next((k for k in ['Net Income', 'NetIncome', 'Net Income Common Stockholders'] if k in financials.index), None)
            
            if not rev_key or not inc_key: return go.Figure()

            revenue = fin_T[rev_key]
            income = fin_T[inc_key]
        except: return go.Figure()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=dates, y=revenue, name="Chiffre d'Affaires", marker_color='#3498db'))
        fig.add_trace(go.Bar(x=dates, y=income, name="Bénéfice Net", marker_color='#2ecc71'))

        fig.update_layout(
            title=dict(text="Croissance (CA vs Bénéfices)", font=dict(size=14, color="#555")),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#eee', tickformat=".2s"),
            margin=dict(t=40, b=20, l=10, r=10), height=200,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", y=-0.2), barmode='group'
        )
        return fig

    # --- MISE EN PAGE DÉTAILLÉE ---
    st.title(f"📊 Analyse Focus : {selected_name}")

    col_left, col_mid, col_right = st.columns([1, 1.5, 1.5], gap="medium")

    with col_left:
        with st.container():
            st.write("##### Rendement & Valorisation")
            st.plotly_chart(plot_dividend_gauge(info['dividend']), use_container_width=True, config={'displayModeBar': False})
            st.divider()
            per_val = info['per']
            per_str = f"{per_val:.1f}x" if per_val and per_val > 0 else "N/A"
            st.metric("PER (Ratio Cours/Bénéfice)", per_str, help="Un PER de 15 est la moyenne historique.")

        with st.container():
            st.write("##### 🎯 Objectif Analystes")
            if info['targetMeanPrice'] and info['targetMeanPrice'] > 0:
                st.plotly_chart(plot_price_vs_target_bar(info['last'], info['targetMeanPrice']), use_container_width=True, config={'displayModeBar': False})
                st.caption(f"Consensus : **{info.get('recommendationKey', 'N/A').upper()}**")
            else:
                st.warning("Pas d'objectif de cours disponible.")

    with col_mid:
        with st.container():
            st.write("##### Indicateurs Clés")
            kpi1, kpi2, kpi3 = st.columns(3)
            var_day = ((info['last'] - info['prev']) / info['prev']) * 100
            kpi1.metric("Prix", f"{info['last']:.2f}€")
            kpi2.metric("Var Jour", f"{var_day:+.2f}%", delta=f"{var_day:+.2f}%")
            kpi3.metric("Market Cap", f"{info['mcap']/1e9:.1f} B€")
            st.divider()
            st.write("##### Performances Historiques")
            st.plotly_chart(plot_performance_bars(hist), use_container_width=True, config={'displayModeBar': False})

        with st.container():
            if cac40_hist_period is not None and not cac40_hist_period.empty:
                fig_vs_bench = plot_price_vs_benchmark(hist['Close'], cac40_hist_period, selected_name)
                st.plotly_chart(fig_vs_bench, use_container_width=True, config={'displayModeBar': False})
            else:
                st.warning("Données du benchmark indisponibles.")

    with col_right:
        with st.container():
            st.write(f"##### Analyse Technique ({time_period_detail})")
            st.plotly_chart(plot_candlestick_real(hist), use_container_width=True, config={'displayModeBar': False})
        
        with st.container():
            st.write("##### 💎 Fondamentaux & Santé")
            f1, f2, f3 = st.columns(3)
            margin = info.get('profitMargins', 0)
            f1.metric("Marge Nette", f"{margin*100:.1f}%" if margin else "N/A", help="Rentabilité nette.")
            beta = info.get('beta', 0)
            f2.metric("Bêta", f"{beta:.2f}" if beta else "N/A", help="Volatilité (1 = moyenne).")
            debt = info.get('debtToEquity', 0)
            f3.metric("Dette", f"{debt:.0f}%" if debt else "N/A")
            
            st.divider()
            
            # --- REMPLACEMENT ICI : GRAPHIQUE FINANCIER ---
            st.caption(f"🏢 Secteur : **{info.get('sector', 'N/A')}**")
            # Appel de la nouvelle fonction financière
            st.plotly_chart(plot_financial_growth(financials), use_container_width=True, config={'displayModeBar': False})
            # ----------------------------------------------
