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
    'RELIANCE.NS', 'TCS.NS'
]

# Sidebar controls
st.sidebar.header("Settings")
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=8, value=4)
date_range = st.sidebar.date_input("Select Date Range", ["2025-05-10", "2025-06-18"])

# Download data
raw_data = yf.download(nifty50_tickers, start=date_range[0], end=date_range[1])

# Check if 'Adj Close' exists
#if 'Adj Close' not in raw_data.columns:
#    st.error("Failed to retrieve valid stock data. Please try again or adjust the date range.")
#    st.stop()

data = raw_data #['Adj Close']

# Log fetched and missing tickers
valid_tickers = data.columns.tolist()
missing = [t for t in nifty50_tickers if t not in valid_tickers]

if missing:
    st.warning(f"Warning: Could not fetch data for these tickers: {', '.join(missing)}")
#if valid_tickers:
#    st.success(f"✅ Data successfully fetched for: {', '.join([t.replace('.NS', '') for t in valid_tickers])}")

# Calculate returns and clean
returns = data.pct_change()
returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any').dropna(axis=0, how='any')
X = returns_clean.T

# Check if enough data is available
if X.shape[0] < 3:
    st.error("Not enough clean stock data to cluster. Try a longer date range or fewer clusters.")
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
