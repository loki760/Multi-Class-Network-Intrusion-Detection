import polars as pl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import time

print("Starting clean sample extraction and dimensionality reduction...")
start_t = time.time()

# 1. Load schema as Float64 for numeric to prevent nulls
schema = {}
with open("dataset/02-20-2018.csv") as f:
    headers = f.readline().strip().split(",")
for h in headers:
    if h in ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"]:
        schema[h] = pl.String
    else:
        schema[h] = pl.Float64

lazy_df = pl.scan_csv("dataset/02-20-2018.csv", schema=schema, ignore_errors=True)

# 2. Extract balanced sample: 25,000 DDoS and 25,000 Benign (50,000 total flows for smooth rendering)
# plus a 10,000 natural ratio sample
print("Sampling 25k DDoS flows...")
ddos_df = lazy_df.filter(pl.col("Label") == "DDoS attacks-LOIC-HTTP").limit(25000).collect()

print("Sampling 25k Benign flows...")
benign_df = lazy_df.filter(pl.col("Label") == "Benign").limit(25000).collect()

sample_df = pl.concat([ddos_df, benign_df])
print(f"Sample collected: {sample_df.shape}")

# 3. Handle infinities and NaNs
# In CICIDS, Flow Byts/s and Flow Pkts/s can be inf or nan
numeric_cols = [c for c, dt in zip(sample_df.columns, sample_df.dtypes) if dt == pl.Float64]

# Drop zero variance columns
zero_var_cols = []
for c in numeric_cols:
    std = sample_df[c].std()
    if std is None or std == 0 or np.isnan(std):
        zero_var_cols.append(c)

print(f"Zero variance columns to exclude ({len(zero_var_cols)}):", zero_var_cols)
active_numeric_cols = [c for c in numeric_cols if c not in zero_var_cols]

# Convert to pandas for sklearn operations
df_pd = sample_df.to_pandas()

# Replace infs with max finite or NaN, then impute with median
for c in active_numeric_cols:
    df_pd[c] = df_pd[c].replace([np.inf, -np.inf], np.nan)
    median_val = df_pd[c].median()
    df_pd[c] = df_pd[c].fillna(median_val if not np.isnan(median_val) else 0.0)

# 4. Standardize for Clustering & PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pd[active_numeric_cols])

print("Computing PCA (3 components)...")
pca = PCA(n_components=3, random_state=42)
pca_coords = pca.fit_transform(X_scaled)
df_pd["PCA1"] = pca_coords[:, 0]
df_pd["PCA2"] = pca_coords[:, 1]
df_pd["PCA3"] = pca_coords[:, 2]

explained_var = pca.explained_variance_ratio_
print(f"PCA Explained Variance: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}, PC3={explained_var[2]:.4f}")

# 5. Compute t-SNE on a 5,000 point subset for ultra-detailed non-linear cluster visualization
print("Computing t-SNE on 5k subset...")
tsne_idx = np.random.RandomState(42).choice(len(df_pd), size=5000, replace=False)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
tsne_coords = tsne.fit_transform(X_scaled[tsne_idx])

df_pd["TSNE1"] = np.nan
df_pd["TSNE2"] = np.nan
df_pd.loc[tsne_idx, "TSNE1"] = tsne_coords[:, 0]
df_pd.loc[tsne_idx, "TSNE2"] = tsne_coords[:, 1]

# 6. Fit K-Means with k=2, 3, 4, 5
print("Computing KMeans clusters...")
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_pd[f"Cluster_k{k}"] = km.fit_predict(X_scaled)

# 7. Save to parquet
out_parquet = "eda/clean_sample.parquet"
df_pd.to_parquet(out_parquet, index=False)
print(f"Saved {out_parquet} ({df_pd.shape[0]} rows, {df_pd.shape[1]} cols) in {time.time() - start_t:.2f} seconds.")

# Also save PCA feature loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2", "PC3"],
    index=active_numeric_cols
)
loadings.to_csv("eda/pca_loadings.csv")
print("Saved eda/pca_loadings.csv successfully.")
