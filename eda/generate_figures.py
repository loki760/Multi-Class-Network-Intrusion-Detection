import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

df = pd.read_parquet("eda/clean_sample.parquet")
df["log_Flow_Duration"] = np.log10(np.maximum(df["Flow Duration"], 1.0))
print("Generating comprehensive EDA figures...")

# 1. Figure 1: Hourly Flow Breakdown
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
hours = ["01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00"]
benign_flows = [819698, 822773, 831891, 807980, 810123, 799540, 815340, 820110, 835200, 881719, 902509, 660047]
ddos_flows = [797, 0, 0, 0, 0, 0, 0, 0, 0, 423621, 151773, 0]

ax.bar(hours, benign_flows, label="Benign Flows (92.75%)", color="#2b5c8f", width=0.6)
ax.bar(hours, ddos_flows, bottom=benign_flows, label="DDoS LOIC-HTTP Flows (7.25%)", color="#d62728", width=0.6)
ax.set_title("Temporal Flow Distribution on 20/02/2018 (All 7,948,748 Captured Flows)", fontweight="bold", pad=12)
ax.set_xlabel("Time of Day (UTC)")
ax.set_ylabel("Captured Flows per Hour")
ax.legend(frameon=True, facecolor="white", loc="upper left")
# Annotate attack burst
ax.annotate("LOIC-HTTP Assault Peak\n(423,621 attack flows)", xy=(9, 1305340), xytext=(6.5, 1250000),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
            fontweight="bold", color="#d62728", bbox=dict(boxstyle="round,pad=0.3", fc="#ffefef", ec="#d62728"))
plt.tight_layout()
plt.savefig("eda/figures/fig1_hourly_timeline.png")
plt.close()
print("Saved fig1_hourly_timeline.png")

# 2. Figure 2: Botnet Distribution
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
bot_ips = [
    "18.219.9.1", "18.218.229.235", "52.14.136.135", "18.218.55.126",
    "18.216.200.189", "18.216.24.42", "18.219.5.43", "18.218.11.51",
    "18.219.32.43", "18.218.115.60"
]
bot_flows = [58532, 58398, 58137, 57998, 57991, 57644, 57611, 57444, 57139, 55297]
y_pos = np.arange(len(bot_ips))
bars = ax.barh(y_pos, bot_flows, color="#c0392b", height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels(bot_ips)
ax.invert_yaxis()
ax.set_xlabel("Flow Count Contributed")
ax.set_title("Botnet Fleet Signature: Flow Contribution by Attacking Node", fontweight="bold", pad=12)
for bar in bars:
    w = bar.get_width()
    ax.text(w - 4500, bar.get_y() + bar.get_height()/2, f"{int(w):,}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
plt.tight_layout()
plt.savefig("eda/figures/fig2_botnet_distribution.png")
plt.close()
print("Saved fig2_botnet_distribution.png")

# 3. Figure 3: PCA Cluster Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
sample_sub = df.sample(n=6000, random_state=42)

# Subplot 1: Ground Truth
colors = {"Benign": "#2980b9", "DDoS attacks-LOIC-HTTP": "#e74c3c"}
for label, col in colors.items():
    mask = sample_sub["Label"] == label
    ax1.scatter(sample_sub.loc[mask, "PCA1"], sample_sub.loc[mask, "PCA2"],
                c=col, label=label, alpha=0.6, s=16, edgecolors="none")
ax1.set_title("PCA Projection (PC1 vs PC2) | Ground Truth", fontweight="bold")
ax1.set_xlabel("Principal Component 1 (Payload & Size Variance - 20.2%)")
ax1.set_ylabel("Principal Component 2 (Flags & Reset Variance - 16.6%)")
ax1.legend(frameon=True, facecolor="white")

# Subplot 2: K-Means (k=3)
cluster_colors = {0: "#27ae60", 1: "#e74c3c", 2: "#2980b9"}
for cid in sorted(sample_sub["Cluster_k3"].unique()):
    mask = sample_sub["Cluster_k3"] == cid
    ax2.scatter(sample_sub.loc[mask, "PCA1"], sample_sub.loc[mask, "PCA2"],
                c=cluster_colors.get(cid, "purple"), label=f"Cluster {cid}", alpha=0.6, s=16, edgecolors="none")
ax2.set_title("PCA Projection (PC1 vs PC2) | K-Means Clusters (k=3)", fontweight="bold")
ax2.set_xlabel("Principal Component 1 (Payload & Size Variance - 20.2%)")
ax2.set_ylabel("Principal Component 2 (Flags & Reset Variance - 16.6%)")
ax2.legend(frameon=True, facecolor="white")

plt.tight_layout()
plt.savefig("eda/figures/fig3_pca_clusters.png")
plt.close()
print("Saved fig3_pca_clusters.png")

# 4. Figure 4: t-SNE Non-Linear Manifold
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
tsne_sub = df.dropna(subset=["TSNE1", "TSNE2"])
for label, col in colors.items():
    mask = tsne_sub["Label"] == label
    ax.scatter(tsne_sub.loc[mask, "TSNE1"], tsne_sub.loc[mask, "TSNE2"],
               c=col, label=label, alpha=0.65, s=20, edgecolors="none")
ax.set_title("t-SNE Non-Linear Manifold (5,000 Flows)", fontweight="bold", pad=12)
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.legend(frameon=True, facecolor="white", loc="upper right")
ax.annotate("Ultra-Dense LOIC-HTTP\nCluster Island", xy=(tsne_sub[tsne_sub["Label"]=="DDoS attacks-LOIC-HTTP"]["TSNE1"].mean(),
                                                      tsne_sub[tsne_sub["Label"]=="DDoS attacks-LOIC-HTTP"]["TSNE2"].mean()),
            xytext=(tsne_sub[tsne_sub["Label"]=="DDoS attacks-LOIC-HTTP"]["TSNE1"].mean() - 15,
                    tsne_sub[tsne_sub["Label"]=="DDoS attacks-LOIC-HTTP"]["TSNE2"].mean() + 20),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=8),
            fontweight="bold", color="#e74c3c", bbox=dict(boxstyle="round,pad=0.3", fc="#ffefef", ec="#e74c3c"))
plt.tight_layout()
plt.savefig("eda/figures/fig4_tsne_clusters.png")
plt.close()
print("Saved fig4_tsne_clusters.png")

# 5. Figure 5: Feature Distributions (Boxplots)
fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
sub_box = df.sample(n=10000, random_state=42)

# Duration (log)
sns.boxplot(data=sub_box, x="Label", y="log_Flow_Duration", palette=["#e74c3c", "#2980b9"], ax=axes[0, 0])
axes[0, 0].set_title("Flow Duration (log10 microseconds)", fontweight="bold")
axes[0, 0].set_xlabel("")

# Init Fwd Win Byts
sns.boxplot(data=sub_box, x="Label", y="Init Fwd Win Byts", palette=["#e74c3c", "#2980b9"], ax=axes[0, 1])
axes[0, 1].set_title("Initial Forward TCP Window (Bytes)", fontweight="bold")
axes[0, 1].set_xlabel("")

# TotLen Bwd Pkts
sns.boxplot(data=sub_box, x="Label", y="TotLen Bwd Pkts", palette=["#e74c3c", "#2980b9"], ax=axes[1, 0])
axes[1, 0].set_title("Total Length of Backward Packets (Bytes)", fontweight="bold")
axes[1, 0].set_yscale("log")
axes[1, 0].set_xlabel("")

# Pkt Len Mean
sns.boxplot(data=sub_box, x="Label", y="Pkt Len Mean", palette=["#e74c3c", "#2980b9"], ax=axes[1, 1])
axes[1, 1].set_title("Mean Packet Length (Bytes)", fontweight="bold")
axes[1, 1].set_xlabel("")

plt.tight_layout()
plt.savefig("eda/figures/fig5_feature_distributions.png")
plt.close()
print("Saved fig5_feature_distributions.png")

# 6. Figure 6: Cluster Contingency Heatmaps for k=2, 3, 4
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=300)
for idx, k in enumerate([2, 3, 4]):
    ct = pd.crosstab(df["Label"], df[f"Cluster_k{k}"], normalize="columns") * 100.0
    sns.heatmap(ct, annot=True, fmt=".1f", cmap="YlOrRd", cbar=True, ax=axes[idx], vmin=0, vmax=100)
    axes[idx].set_title(f"Cluster Purity (%) | k={k}", fontweight="bold")
    axes[idx].set_xlabel("Assigned Cluster")
    axes[idx].set_ylabel("Ground Truth" if idx == 0 else "")

plt.tight_layout()
plt.savefig("eda/figures/fig6_cluster_purity_heatmap.png")
plt.close()
print("Saved fig6_cluster_purity_heatmap.png")

# 7. Figure 7: Correlation Matrix of Discriminant Features
disc_features = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Pkt Len Mean", "Pkt Len Max",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt"
]
corr = df[disc_features].corr()
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
ax.set_title("Pearson Correlation Matrix of Key Discriminant Network Features", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("eda/figures/fig7_correlation_matrix.png")
plt.close()
print("Saved fig7_correlation_matrix.png")

print("All figures successfully generated in eda/figures/!")
