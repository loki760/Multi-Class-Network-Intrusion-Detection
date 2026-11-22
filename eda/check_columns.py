import polars as pl
import numpy as np

csv_path = "dataset/02-20-2018.csv"

# Scan schema and null counts
print("Scanning column statistics and quality...")
lazy_df = pl.scan_csv(csv_path, infer_schema_length=10000, ignore_errors=True)

# Select first 100k to inspect types and test numeric conversion
df_sample = lazy_df.limit(100000).collect()
print("Sample shape:", df_sample.shape)

# Check all column types
numeric_cols = []
string_cols = []
for col_name, dtype in zip(df_sample.columns, df_sample.dtypes):
    if dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32, pl.UInt32]:
        numeric_cols.append(col_name)
    else:
        string_cols.append(col_name)

print("Numeric cols count:", len(numeric_cols))
print("String cols:", string_cols)

# Check nulls in sample
null_counts = {col: df_sample[col].null_count() for col in df_sample.columns if df_sample[col].null_count() > 0}
print("Columns with nulls in sample:", null_counts)

# Let's inspect 'Flow Byts/s' and 'Flow Pkts/s' which commonly have 'Infinity' or NaN in CICIDS
print("Flow Byts/s dtype:", df_sample['Flow Byts/s'].dtype)
print("Flow Pkts/s dtype:", df_sample['Flow Pkts/s'].dtype)
