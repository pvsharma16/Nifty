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

# Nifty 50 symbols (Yahoo Finance format)
nifty50_tickers = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
    'LT.NS', 'SBIN.NS', 'HINDUNILVR.NS', 'ITC.NS', 'BHARTIARTL.NS',
    'ASIANPAINT.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'SUNPHARMA.NS',
    'MARUTI.NS', 'WIPRO.NS', 'HCLTECH.NS', 'NESTLEIND.NS', 'TECHM.NS',
    'ULTRACEMCO.NS', 'TITAN.NS', 'NTPC.NS', 'ONGC.NS', 'JSWSTEEL.NS',
    'COALINDIA.NS', 'POWERGRID.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
    'ADANIENT.NS', 'ADANIPORTS.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS',
    'BRITANNIA.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'EICHERMOT.NS', 'BPCL.NS',
    'GRASIM.NS', 'INDUSINDBK.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS',
    'DRREDDY.NS', 'UPL.NS', 'SHREECEM.NS', 'SBILIFE.NS', 'M&M.NS',
    'BAJAJ-AUTO.NS', 'HINDALCO.NS', 'ICICIPRULI.NS', 'APOLLOHOSP.NS'
]

# Sidebar controls
st.sidebar.header("Settings")
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=8, value=4)
date_range = st.sidebar.date_input("Select Date Range", ["2024-06-01", "2025-06-01"])

# Download data
data = yf.download(nifty50_tickers, start=date_range[0], end=date_range[1])
returns = data.pct_change().dropna()

# Preprocessing
X = returns.T
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
    st.dataframe(returns.T.round(4))
