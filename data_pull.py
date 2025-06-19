import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import io

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
        adj_close = pd.DataFrame({
            ticker: raw_data[ticker]['Close']
            for ticker in nifty50_tickers
            if ticker in raw_data and 'Close' in raw_data[ticker]
        })
        st.write("Sample of fetched Adjusted Close data:", adj_close.head(3))
    except Exception as e:
        st.error(f"❌ Error while downloading data: {e}")
        st.stop()

if adj_close.empty:
    st.error("No data could be retrieved for the selected date range. Try a broader range or different tickers.")
    st.stop()

valid_tickers = adj_close.columns.tolist()
missing = [t for t in nifty50_tickers if t not in valid_tickers]
if missing:
    st.warning(f"⚠️ Could not fetch data for: {', '.join(missing)}")
if valid_tickers:
    st.success(f"✅ Data successfully fetched for: {', '.join([t.replace('.NS', '') for t in valid_tickers])}")

returns = adj_close.pct_change()
returns_clean = returns.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
X = returns_clean.T

st.info(f"✅ {X.shape[0]} out of {len(nifty50_tickers)} stocks have usable return data.")

if X.shape[0] <= n_clusters:
    st.warning(
        f"Only {X.shape[0]} clean stocks available, which is less than or equal to the number of clusters ({n_clusters})."
    )
    st.info("Try reducing the number of clusters or extending the date range.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots(figsize=(12, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set2', s=100, ax=ax)

for i, ticker in enumerate(X.index):
    ax.text(X_pca[i, 0]+0.01, X_pca[i, 1], ticker.replace('.NS', ''), fontsize=8)

ax.set_title("Nifty 50 Stock Clusters Based on Daily Returns")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.grid(True)
st.pyplot(fig)

# Cluster-level mean return and volatility
st.subheader("📈 Cluster Averages: Return & Volatility")
X_df = pd.DataFrame(X_scaled, index=X.index)
X_df['Cluster'] = labels
X_df['Mean Return'] = X_df.index.map(lambda t: returns_clean[t].mean())
X_df['Volatility'] = X_df.index.map(lambda t: returns_clean[t].std())

summary = X_df.groupby('Cluster')[['Mean Return', 'Volatility']].mean().round(4)
st.dataframe(summary)

# Scatter plot of cluster means
fig2, ax2 = plt.subplots()
ax2.scatter(summary['Volatility'], summary['Mean Return'], c=summary.index, cmap='Set2', s=120)
for i in summary.index:
    ax2.text(summary.loc[i, 'Volatility']+0.0005, summary.loc[i, 'Mean Return'], f"Cluster {i}", fontsize=9)
ax2.set_xlabel("Volatility")
ax2.set_ylabel("Mean Return")
ax2.set_title("📊 Cluster Centers: Mean Return vs. Volatility")
ax2.grid(True)
st.pyplot(fig2)

# CSV download
csv_buffer = io.StringIO()
X_df.reset_index().rename(columns={'index': 'Ticker'}).to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 Download Clustered Stock Data as CSV",
    data=csv_buffer.getvalue(),
    file_name="nifty50_clusters.csv",
    mime="text/csv"
)

# Show raw data option
if st.sidebar.checkbox("Show Raw Returns Data"):
    st.subheader("Daily Returns")
    st.dataframe(returns_clean.T.round(4))
