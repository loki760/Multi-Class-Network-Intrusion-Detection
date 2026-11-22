import polars as pl
import numpy as np
import json
import time

print("Analyzing attack timeline and feature distributions...")
csv_path = "dataset/02-20-2018.csv"

lazy_df = pl.scan_csv(csv_path, infer_schema_length=10000, ignore_errors=True)

# 1. Timeline: extract hour from Timestamp "20/02/2018 HH:MM:SS" or similar format
# Let's inspect timestamp formats
time_sample = lazy_df.select("Timestamp").limit(10).collect()
print("Timestamp sample:\n", time_sample)

# Let's group by Timestamp hour and Label
timeline = lazy_df.select([
    pl.col("Timestamp").str.slice(11, 2).alias("hour"),
    pl.col("Label")
]).group_by(["hour", "Label"]).len().collect().sort(["hour", "Label"])

print("\n--- Hourly Flow Distribution ---")
print(timeline)

# 2. Let's inspect zero-variance columns using a 500k sample across both classes
print("\nExtracting stratified sample for deep statistical analysis and clustering...")
# Let's get 50,000 DDoS and 50,000 Benign for balanced clustering analysis
ddos_sample = lazy_df.filter(pl.col("Label") == "DDoS attacks-LOIC-HTTP").limit(50000).collect()
benign_sample = lazy_df.filter(pl.col("Label") == "Benign").limit(50000).collect()

combined_sample = pl.concat([ddos_sample, benign_sample])
print("Combined sample shape:", combined_sample.shape)

# Check zero-variance numeric columns
zero_var_cols = []
numeric_cols = [col for col, dtype in zip(combined_sample.columns, combined_sample.dtypes) if dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32, pl.UInt32]]

for col in numeric_cols:
    std_val = combined_sample[col].std()
    if std_val is None or std_val == 0 or np.isnan(std_val):
        zero_var_cols.append(col)

print(f"\nZero-variance columns ({len(zero_var_cols)}):", zero_var_cols)

# Key feature comparison
key_features = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean",
    "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Fwd IAT Mean", "Bwd IAT Mean", "Fwd Header Len", "Bwd Header Len",
    "Fwd Pkts/s", "Bwd Pkts/s", "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean",
    "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    "Down/Up Ratio", "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min"
]

print("\n--- Key Feature Statistics by Label (Sampled) ---")
summary_list = []
for f in key_features:
    b_mean = combined_sample.filter(pl.col("Label") == "Benign")[f].mean()
    b_std = combined_sample.filter(pl.col("Label") == "Benign")[f].std()
    d_mean = combined_sample.filter(pl.col("Label") == "DDoS attacks-LOIC-HTTP")[f].mean()
    d_std = combined_sample.filter(pl.col("Label") == "DDoS attacks-LOIC-HTTP")[f].std()
    summary_list.append({
        "Feature": f,
        "Benign Mean": round(float(b_mean), 2) if b_mean is not None else 0,
        "Benign Std": round(float(b_std), 2) if b_std is not None else 0,
        "DDoS Mean": round(float(d_mean), 2) if d_mean is not None else 0,
        "DDoS Std": round(float(d_std), 2) if d_std is not None else 0,
    })

summary_df = pl.DataFrame(summary_list)
print(summary_df)

# Save sample to parquet
combined_sample.write_parquet("eda/sample_flows_100k.parquet")
print("Saved eda/sample_flows_100k.parquet successfully.")
