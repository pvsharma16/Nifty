import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Nifty 50 Stock Clustering", layout="wide")
st.title("📊 Nifty 50 Stock Clustering Dashboard")

# Updated Nifty 50 ticker list (verified on Yahoo Finance)
nifty50_tickers = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
    'LT.NS', 'SBIN.NS', 'HINDUNILVR.NS', 'ITC.NS', 'BHARTIARTL.NS',
    'ASIANPAINT.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'SUNPHARMA.NS',
    'MARUTI.NS', 'WIPRO.NS', 'HCLTECH.NS', 'NESTLEIND.NS', 'TECHM.NS',
    'ULTRACEMCO.NS', 'TITAN.NS', 'NTPC.NS', 'ONGC.NS', 'JSWSTEEL.NS',
    'COALINDIA.NS', 'POWERGRID.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
    'ADANIENT.NS', 'ADANIPORTS.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS',
    'CIPLA.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'INDUSINDBK.NS',
    'TATAMOTORS.NS', 'TATASTEEL.NS', 'DRREDDY.NS', 'SBILIFE.NS',
    'M&M.NS', 'BAJAJ-AUTO.NS', 'HINDALCO.NS', 'APOLLOHOSP.NS',
    'DABUR.NS', 'PIDILITIND.NS', 'HAVELLS.NS', 'ICICIGI.NS',
    'CHOLAFIN.NS', 'TRENT.NS'
]

# Sidebar controls
st.sidebar.header("Settings")
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=8, value=4)
date_range = st.sidebar.date_input("Select Date Range", ["2024-01-01", "2025-06-01"])

# Download data with diagnostics
with st.spinner("Fetching data from Yahoo Finance..."):
    try:
        raw_data = yf.download(
            tickers=nifty50_tickers,
            start=date_range[0],
            end=date_range[1],
            group_by='ticker',
            auto_adjust=True,
            threads=True
        )
        # Extract adjusted close prices manually
        adj_close = pd.DataFrame({
            ticker: raw_data[ticker]['Close']
            for ticker in nifty50_tickers
            if ticker in raw_data and 'Close' in raw_data[ticker]
        })
        st.write("Sample of fetched Adjusted Close data:", adj_close.head(3))
    except Exception as e:
        st.error(f"❌ Error while downloading data: {e}")
        st.stop()

# Check if any data was fetched
if adj_close.empty:
    st.error("No data could be retrieved for the selected date range. Try a broader range or different tickers.")
    st.stop()

# Log fetched and missing tickers
valid_tickers = adj_close.columns.tolist()
missing = [t for t in nifty50_tickers if t not in valid_tickers]
if missing:
    st.warning(f"⚠️ Could not fetch data for: {', '.join(missing)}")
if valid_tickers:
    st.success(f"✅ Data successfully fetched for: {', '.join([t.replace('.NS', '') for t in valid_tickers])}")

# Calculate returns and clean
returns = adj_close.pct_change()
returns_clean = returns.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
X = returns_clean.T

# Show how many valid stocks are available
st.info(f"✅ {X.shape[0]} out of {len(nifty50_tickers)} stocks have usable return data.")

# Check if enough data is available
if X.shape[0] <= n_clusters:
    st.warning(
        f"Only {X.shape[0]} clean stocks available, which is less than or equal to the number of clusters ({n_clusters})."
    )
    st.info("Try reducing the number of clusters or extending the date range.")
    st.stop()

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Visualization
fig, ax = plt.subplots(figsize=(12, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set2', s=100, ax=ax)

# Annotate points
for i, ticker in enumerate(X.index):
    ax.text(X_pca[i, 0]+0.01, X_pca[i, 1], ticker.replace('.NS', ''), fontsize=8)

ax.set_title("Nifty 50 Stock Clusters Based on Daily Returns")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.grid(True)
st.pyplot(fig)

# Show raw data option
if st.sidebar.checkbox("Show Raw Returns Data"):
    st.subheader("Daily Returns")
    st.dataframe(returns_clean.T.round(4))
