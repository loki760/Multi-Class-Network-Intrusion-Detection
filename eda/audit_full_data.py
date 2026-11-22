import polars as pl
import numpy as np
import json
import time

start_t = time.time()
print("Starting full scan of dataset/02-20-2018.csv...")

# First check for repeated headers or string anomalies in numeric columns
# We can use pl.scan_csv with schema inference
csv_path = "dataset/02-20-2018.csv"

# Let's inspect column names and check if there are rows where Dst Port is 'Dst Port'
lazy_df = pl.scan_csv(
    csv_path,
    infer_schema_length=10000,
    ignore_errors=True
)

total_rows = lazy_df.select(pl.len()).collect().item()
print(f"Total rows collected: {total_rows}")

# Check labels
label_dist = lazy_df.group_by("Label").len().collect().sort("len", descending=True)
print("\n--- Label Distribution ---")
print(label_dist)

# Check unique IPs and Ports
ip_stats = lazy_df.select([
    pl.col("Src IP").n_unique().alias("n_src_ip"),
    pl.col("Dst IP").n_unique().alias("n_dst_ip"),
    pl.col("Dst Port").n_unique().alias("n_dst_port"),
    pl.col("Protocol").n_unique().alias("n_protocol")
]).collect()
print("\n--- Network Entities Unique Counts ---")
print(ip_stats)

# Top Destination IPs & Ports for DDoS vs Benign
top_dst_ddos = lazy_df.filter(pl.col("Label").str.contains("DDoS")).group_by(["Dst IP", "Dst Port", "Protocol"]).len().collect().sort("len", descending=True).head(10)
print("\n--- Top Targets for DDoS attacks ---")
print(top_dst_ddos)

top_src_ddos = lazy_df.filter(pl.col("Label").str.contains("DDoS")).group_by("Src IP").len().collect().sort("len", descending=True).head(10)
print("\n--- Top Attacker Source IPs ---")
print(top_src_ddos)

# Top Destination IPs & Ports for Benign
top_dst_benign = lazy_df.filter(pl.col("Label") == "Benign").group_by(["Dst IP", "Dst Port"]).len().collect().sort("len", descending=True).head(10)
print("\n--- Top Targets for Benign ---")
print(top_dst_benign)

# Timestamp analysis
timestamps = lazy_df.select(
    pl.col("Timestamp").min().alias("min_time"),
    pl.col("Timestamp").max().alias("max_time")
).collect()
print("\n--- Timestamp Span ---")
print(timestamps)

print(f"Scan finished in {time.time() - start_t:.2f} seconds.")
