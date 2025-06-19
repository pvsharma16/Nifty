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
st.set_page_config(page_title="NSE Stock Clustering", layout="wide")
st.title("📊 NSE Stock Clustering Dashboard")

# Load stock metadata
metadata_df = pd.read_csv("nse_fo_sample_template.csv")

# Sidebar controls
st.sidebar.header("Settings")
stock_set = st.sidebar.selectbox("Stock Universe", ["Nifty 50", "F&O Stocks"])
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=8, value=4)
date_range = st.sidebar.date_input("Select Date Range", ["2024-01-01", "2025-06-01"])
mode = st.sidebar.selectbox("Cluster by", ["Overall returns", "Day-of-week patterns"])

# Optional filters
selected_sector = st.sidebar.multiselect(
    "Filter by Sector", options=metadata_df['Sector'].unique(), default=metadata_df['Sector'].unique()
)
selected_cap = st.sidebar.multiselect(
    "Filter by Market Cap", options=metadata_df['MarketCap'].unique(), default=metadata_df['MarketCap'].unique()
)

# Apply filters
filtered_df = metadata_df[
    metadata_df['Sector'].isin(selected_sector) & metadata_df['MarketCap'].isin(selected_cap)
]

# Select tickers based on stock set
if stock_set == "Nifty 50":
    tickers = filtered_df['Ticker'].tolist()[:50]
else:
    tickers = filtered_df['Ticker'].tolist()

# Download data
with st.spinner("Fetching data from Yahoo Finance..."):
    try:
        raw_data = yf.download(
            tickers=tickers,
            start=date_range[0],
            end=date_range[1],
            group_by='ticker',
            auto_adjust=True,
            threads=True
        )
        adj_close = pd.DataFrame({
            ticker: raw_data[ticker]['Close']
            for ticker in tickers
            if ticker in raw_data and 'Close' in raw_data[ticker]
        })
    except Exception as e:
        st.error(f"❌ Error while downloading data: {e}")
        st.stop()

if adj_close.empty:
    st.error("No data could be retrieved for the selected date range. Try a broader range or different tickers.")
    st.stop()

returns = adj_close.pct_change()
returns_clean = returns.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

if mode == "Day-of-week patterns":
    returns_clean['Day'] = returns_clean.index.day_name()
    weekday_avg = returns_clean.groupby('Day').mean().T
    weekday_avg = weekday_avg[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']]
    X = weekday_avg
else:
    X = returns_clean.T

st.info(f"✅ {X.shape[0]} out of {len(tickers)} stocks have usable return data.")

if X.shape[0] <= n_clusters:
    st.warning(f"Only {X.shape[0]} clean stocks available, which is less than or equal to the number of clusters ({n_clusters}).")
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

ax.set_title("NSE Stock Clusters")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.grid(True)
st.pyplot(fig)

# Cluster-level summary
st.subheader("📈 Cluster Averages: Return, Volatility & Count")
X_df = pd.DataFrame(X_scaled, index=X.index)
X_df['Cluster'] = labels
X_df['Mean Return'] = X_df.index.map(lambda t: returns_clean[t].mean())
X_df['Volatility'] = X_df.index.map(lambda t: returns_clean[t].std())

summary = X_df.groupby('Cluster').agg({
    'Mean Return': 'mean',
    'Volatility': 'mean',
    'Cluster': 'count'
}).rename(columns={'Cluster': 'Count'}).round(4)

# Highlight best cluster
best_cluster = summary['Mean Return'].idxmax()

def highlight_best(s):
    is_best = s.name == best_cluster
    return ['background-color: #c6f6d5; font-weight: bold' if is_best else '' for _ in s]

summary_style = summary.style.apply(highlight_best, axis=1)
st.dataframe(summary_style)

# Scatter plot of cluster means
fig2, ax2 = plt.subplots()
ax2.scatter(summary['Volatility'], summary['Mean Return'], c=summary.index, cmap='Set2', s=120)
for i in summary.index:
    label = f"🏆 Cluster {i}" if i == best_cluster else f"Cluster {i}"
    ax2.text(summary.loc[i, 'Volatility']+0.0005, summary.loc[i, 'Mean Return'], label, fontsize=9)
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
    file_name="nse_clusters.csv",
    mime="text/csv"
)

# Show raw data option
if st.sidebar.checkbox("Show Raw Returns Data"):
    st.subheader("Daily Returns")
    st.dataframe(returns_clean.T.round(4))
