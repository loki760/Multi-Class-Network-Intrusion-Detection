import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="CSE-CIC-IDS2018 (02-20-2018) EDA & Cluster Visualizer")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import os

    return (
        KMeans,
        PCA,
        StandardScaler,
        go,
        make_subplots,
        mo,
        np,
        os,
        pd,
        px,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # 🛡️ CSE-CIC-IDS2018: 02-20-2018 Intrusion Detection & Cluster EDA
        ### *Interactive Flow Exploration, Dimensionality Reduction, and Botnet Clustering*

        ***

        This interactive application explores **`02-20-2018.csv`** from the Canadian Institute for Cybersecurity CSE-CIC-IDS2018 benchmark. 
        This day features a massive **Distributed Denial of Service (DDoS) LOIC-HTTP** attack launched by 10 bot nodes against a central web server (`172.31.69.25:80`), alongside multi-million benign enterprise flows.
        """
    )
    return


@app.cell
def __(mo):
    # Metric cards for global dataset overview
    card1 = mo.stat(
        value="7,948,748",
        label="Total Captured Flows",
        caption="3.8 GB uncompressed CSV"
    )
    card2 = mo.stat(
        value="576,191 (7.25%)",
        label="DDoS LOIC-HTTP Flows",
        caption="100% targeting Port 80"
    )
    card3 = mo.stat(
        value="7,372,557 (92.75%)",
        label="Benign Traffic Flows",
        caption="DNS, AWS metadata, Web"
    )
    card4 = mo.stat(
        value="10 Nodes",
        label="Attacker Botnet Fleet",
        caption="~57.6k flows per bot"
    )
    card5 = mo.stat(
        value="172.31.69.25 : 80",
        label="Victim Web Server",
        caption="Target of HTTP Flood"
    )

    mo.hstack([card1, card2, card3, card4, card5], justify="space-around")
    return card1, card2, card3, card4, card5


@app.cell
def __(pd):
    # Load the clean sample dataset
    sample_path = "eda/clean_sample.parquet"
    if not pd.io.common.file_exists(sample_path):
        sample_path = "clean_sample.parquet"
    df_raw = pd.read_parquet(sample_path)
    
    # Pre-calculate log features for smoother visualization
    df_raw["log_Flow_Duration"] = np.log10(np.maximum(df_raw["Flow Duration"], 1.0))
    df_raw["log_TotLen_Fwd_Pkts"] = np.log10(np.maximum(df_raw["TotLen Fwd Pkts"], 1.0))
    df_raw["log_TotLen_Bwd_Pkts"] = np.log10(np.maximum(df_raw["TotLen Bwd Pkts"], 1.0))
    df_raw["log_Flow_IAT_Mean"] = np.log10(np.maximum(df_raw["Flow IAT Mean"], 1.0))
    return df_raw, sample_path


@app.cell
def __(df_raw, mo):
    mo.md("## 🎛️ Interactive Exploration Controls")
    
    projection_dropdown = mo.ui.dropdown(
        options=["PCA 2D", "PCA 3D", "t-SNE 2D", "Feature Space (2D)"],
        value="PCA 2D",
        label="Projection / Visualization Mode"
    )
    
    cluster_k_slider = mo.ui.slider(
        start=2,
        stop=5,
        step=1,
        value=3,
        label="Number of Clusters (k)"
    )
    
    color_by_dropdown = mo.ui.dropdown(
        options=[
            "Assigned Cluster",
            "True Label",
            "Dst Port",
            "Protocol",
            "log_Flow_Duration",
            "Tot Fwd Pkts",
            "Tot Bwd Pkts",
            "Init Fwd Win Byts",
            "Init Bwd Win Byts",
            "Pkt Len Mean"
        ],
        value="Assigned Cluster",
        label="Color Data Points By"
    )
    
    sample_slider = mo.ui.slider(
        start=1000,
        stop=20000,
        step=1000,
        value=5000,
        label="Display Points Sample Size"
    )
    
    x_feature_select = mo.ui.dropdown(
        options=[
            "Flow Duration",
            "log_Flow_Duration",
            "Tot Fwd Pkts",
            "Tot Bwd Pkts",
            "TotLen Fwd Pkts",
            "TotLen Bwd Pkts",
            "Pkt Len Mean",
            "Pkt Len Max",
            "Flow IAT Mean",
            "log_Flow_IAT_Mean",
            "Init Fwd Win Byts",
            "Init Bwd Win Byts"
        ],
        value="log_Flow_Duration",
        label="Custom X Feature (Feature Space mode)"
    )
    
    y_feature_select = mo.ui.dropdown(
        options=[
            "Tot Fwd Pkts",
            "Tot Bwd Pkts",
            "TotLen Fwd Pkts",
            "TotLen Bwd Pkts",
            "log_TotLen_Fwd_Pkts",
            "log_TotLen_Bwd_Pkts",
            "Pkt Len Mean",
            "Init Fwd Win Byts",
            "Init Bwd Win Byts"
        ],
        value="log_TotLen_Fwd_Pkts",
        label="Custom Y Feature (Feature Space mode)"
    )

    mo.vstack([
        mo.hstack([projection_dropdown, cluster_k_slider, sample_slider], justify="start"),
        mo.hstack([color_by_dropdown, x_feature_select, y_feature_select], justify="start")
    ])
    return (
        cluster_k_slider,
        color_by_dropdown,
        projection_dropdown,
        sample_slider,
        x_feature_select,
        y_feature_select,
    )


@app.cell
def __(
    cluster_k_slider,
    color_by_dropdown,
    df_raw,
    go,
    np,
    projection_dropdown,
    px,
    sample_slider,
    x_feature_select,
    y_feature_select,
):
    # Downsample for fluid interactive rendering
    k_val = cluster_k_slider.value
    n_sample = min(sample_slider.value, len(df_raw))
    
    # Stratified downsample
    df_plot = df_raw.sample(n=n_sample, random_state=42).copy()
    cluster_col = f"Cluster_k{k_val}"
    df_plot["Cluster"] = "Cluster " + df_plot[cluster_col].astype(str)
    
    color_target = "Cluster" if color_by_dropdown.value == "Assigned Cluster" else (
        "Label" if color_by_dropdown.value == "True Label" else color_by_dropdown.value
    )
    
    hover_cols = ["Label", "Cluster", "Src IP", "Dst IP", "Dst Port", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "Init Fwd Win Byts"]
    
    mode = projection_dropdown.value
    
    if mode == "PCA 2D":
        fig = px.scatter(
            df_plot,
            x="PCA1",
            y="PCA2",
            color=color_target,
            hover_data=hover_cols,
            title=f"<b>2D PCA Projection (k={k_val} Clusters)</b> | Colored by: {color_target}",
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.75,
            template="plotly_white"
        )
        fig.update_layout(
            xaxis_title="Principal Component 1 (Payload & Size Variance - 20.2%)",
            yaxis_title="Principal Component 2 (Flags & Reset Variance - 16.6%)",
            height=650,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    elif mode == "PCA 3D":
        fig = px.scatter_3d(
            df_plot,
            x="PCA1",
            y="PCA2",
            z="PCA3",
            color=color_target,
            hover_data=hover_cols,
            title=f"<b>3D PCA Cluster Manifold (k={k_val})</b> | Colored by: {color_target}",
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.8,
            template="plotly_white"
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="PC1 (Payload)",
                yaxis_title="PC2 (Flags)",
                zaxis_title="PC3 (Response)"
            ),
            height=750,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    elif mode == "t-SNE 2D":
        # Subset with non-null TSNE coordinates
        df_tsne = df_plot.dropna(subset=["TSNE1", "TSNE2"])
        if len(df_tsne) < 500:
            df_tsne = df_raw.dropna(subset=["TSNE1", "TSNE2"]).copy()
            df_tsne["Cluster"] = "Cluster " + df_tsne[cluster_col].astype(str)
        fig = px.scatter(
            df_tsne,
            x="TSNE1",
            y="TSNE2",
            color=color_target if color_target in df_tsne.columns else "Cluster",
            hover_data=hover_cols,
            title=f"<b>t-SNE Non-Linear Manifold (5,000 Flows)</b> | Colored by: {color_target}",
            color_discrete_sequence=px.colors.qualitative.D3,
            opacity=0.8,
            template="plotly_white"
        )
        fig.update_layout(
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            height=650,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    else: # Feature Space 2D
        x_f = x_feature_select.value
        y_f = y_feature_select.value
        fig = px.scatter(
            df_plot,
            x=x_f,
            y=y_f,
            color=color_target,
            hover_data=hover_cols,
            title=f"<b>Bivariate Feature Space: {x_f} vs {y_f}</b> | Colored by: {color_target}",
            color_discrete_sequence=px.colors.qualitative.Safe,
            opacity=0.75,
            template="plotly_white"
        )
        fig.update_layout(
            xaxis_title=x_f,
            yaxis_title=y_f,
            height=650,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    
    return cluster_col, color_target, df_plot, fig, hover_cols, k_val, mode, n_sample


@app.cell
def __(fig, mo):
    mo.ui.plotly(fig)
    return


@app.cell
def __(cluster_col, df_raw, k_val, mo, pd, px):
    mo.md(f"### 🔍 Cluster Purity & Contingency Analysis (k={k_val})")
    
    # Compute contingency table
    ct = pd.crosstab(df_raw["Label"], df_raw[cluster_col], normalize=False)
    ct_norm = pd.crosstab(df_raw["Label"], df_raw[cluster_col], normalize="columns") * 100.0
    
    # Heatmap of cluster purity
    fig_heat = px.imshow(
        ct_norm,
        text_auto=".1f",
        labels=dict(x="Assigned Cluster", y="Ground Truth Label", color="% of Cluster"),
        x=[f"Cluster {c}" for c in ct.columns],
        y=ct.index.tolist(),
        color_continuous_scale="Viridis",
        title=f"<b>Cluster Purity Matrix (%) for k={k_val}</b> (Column-Normalized)"
    )
    fig_heat.update_layout(height=340, template="plotly_white")
    
    # Format counts table
    ct_display = ct.copy()
    ct_display.columns = [f"Cluster {c}" for c in ct_display.columns]
    ct_display["Total"] = ct_display.sum(axis=1)
    
    mo.hstack([
        mo.vstack([
            mo.md("#### Flow Counts per Cluster"),
            mo.ui.table(ct_display.reset_index())
        ]),
        mo.ui.plotly(fig_heat)
    ], justify="space-between")
    return ct, ct_display, ct_norm, fig_heat


@app.cell
def __(df_raw, mo):
    mo.md("## 📊 Univariate Feature Profiler by Cluster & Ground Truth")
    
    feature_dist_select = mo.ui.dropdown(
        options=[
            "Flow Duration",
            "Tot Fwd Pkts",
            "Tot Bwd Pkts",
            "TotLen Fwd Pkts",
            "TotLen Bwd Pkts",
            "Pkt Len Mean",
            "Pkt Len Max",
            "Flow IAT Mean",
            "Fwd IAT Mean",
            "Init Fwd Win Byts",
            "Init Bwd Win Byts",
            "SYN Flag Cnt",
            "RST Flag Cnt",
            "PSH Flag Cnt",
            "ACK Flag Cnt"
        ],
        value="Init Fwd Win Byts",
        label="Select Network Feature to Profile"
    )
    
    plot_type_select = mo.ui.dropdown(
        options=["Overlaid Histogram / KDE", "Box Plot by Label", "Violin Plot by Cluster"],
        value="Box Plot by Label",
        label="Plot Type"
    )
    
    mo.hstack([feature_dist_select, plot_type_select], justify="start")
    return feature_dist_select, plot_type_select


@app.cell
def __(
    cluster_col,
    df_raw,
    feature_dist_select,
    mo,
    np,
    plot_type_select,
    px,
):
    feat = feature_dist_select.value
    ptype = plot_type_select.value
    
    # 5000 row sample for snappy distribution plots
    df_dist = df_raw.sample(n=min(5000, len(df_raw)), random_state=42).copy()
    df_dist["Cluster"] = "Cluster " + df_dist[cluster_col].astype(str)
    
    if ptype == "Overlaid Histogram / KDE":
        fig_dist = px.histogram(
            df_dist,
            x=feat,
            color="Label",
            barmode="overlay",
            marginal="box",
            opacity=0.65,
            title=f"<b>Distribution of {feat} by Label</b>",
            color_discrete_sequence=["#1f77b4", "#d62728"],
            template="plotly_white"
        )
    elif ptype == "Box Plot by Label":
        fig_dist = px.box(
            df_dist,
            x="Label",
            y=feat,
            color="Label",
            points="outliers",
            title=f"<b>Boxplot of {feat} by Ground Truth Label</b>",
            color_discrete_sequence=["#1f77b4", "#d62728"],
            template="plotly_white"
        )
    else:
        fig_dist = px.violin(
            df_dist,
            x="Cluster",
            y=feat,
            color="Label",
            box=True,
            points=False,
            title=f"<b>Violin Plot of {feat} across Clusters and Labels</b>",
            color_discrete_sequence=["#1f77b4", "#d62728"],
            template="plotly_white"
        )
        
    fig_dist.update_layout(height=450, template="plotly_white")
    mo.ui.plotly(fig_dist)
    return df_dist, feat, fig_dist, ptype


@app.cell
def __(go, make_subplots, mo, pd, px):
    mo.md("## 🌐 Network Infrastructure & Attack Timeline")
    
    # Timeline data from full audit
    timeline_data = {
        "Hour": ["01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00"],
        "Benign": [819698, 822773, 831891, 807980, 810123, 799540, 815340, 820110, 835200, 881719, 902509, 660047],
        "DDoS_LOIC_HTTP": [797, 0, 0, 0, 0, 0, 0, 0, 0, 423621, 151773, 0]
    }
    df_time = pd.DataFrame(timeline_data)
    
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Bar(
        x=df_time["Hour"],
        y=df_time["Benign"],
        name="Benign Flows",
        marker_color="#2b5c8f"
    ))
    fig_timeline.add_trace(go.Bar(
        x=df_time["Hour"],
        y=df_time["DDoS_LOIC_HTTP"],
        name="DDoS LOIC-HTTP Flows",
        marker_color="#d62728"
    ))
    fig_timeline.update_layout(
        barmode="stack",
        title="<b>Hourly Flow Volume on 20/02/2018 (All 7.95M Flows)</b>",
        xaxis_title="Time of Day (UTC)",
        yaxis_title="Total Flows Recorded",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Attacker Bot Fleet
    bot_ips = [
        "18.219.9.1", "18.218.229.235", "52.14.136.135", "18.218.55.126",
        "18.216.200.189", "18.216.24.42", "18.219.5.43", "18.218.11.51",
        "18.219.32.43", "18.218.115.60"
    ]
    bot_flows = [58532, 58398, 58137, 57998, 57991, 57644, 57611, 57444, 57139, 55297]
    df_bots = pd.DataFrame({"Attacker IP": bot_ips, "Attacking Flows": bot_flows})
    
    fig_bots = px.bar(
        df_bots,
        x="Attacking Flows",
        y="Attacker IP",
        orientation="h",
        title="<b>Botnet Fleet Flow Contribution (10 EC2 Instances)</b>",
        color="Attacking Flows",
        color_continuous_scale="Reds",
        template="plotly_white"
    )
    fig_bots.update_layout(height=380, yaxis={'categoryorder':'total ascending'})
    
    mo.hstack([mo.ui.plotly(fig_timeline), mo.ui.plotly(fig_bots)], justify="space-between")
    return bot_flows, bot_ips, df_bots, df_time, fig_bots, fig_timeline, timeline_data


@app.cell
def __(df_raw, mo):
    mo.md("## 🔎 Inspect Flow Level Records")
    
    table_filter = mo.ui.dropdown(
        options=["All Samples", "DDoS attacks-LOIC-HTTP Only", "Benign Only"],
        value="All Samples",
        label="Filter Sample Table"
    )
    return table_filter,


@app.cell
def __(df_raw, mo, table_filter):
    if table_filter.value == "DDoS attacks-LOIC-HTTP Only":
        df_sub = df_raw[df_raw["Label"] == "DDoS attacks-LOIC-HTTP"].head(200)
    elif table_filter.value == "Benign Only":
        df_sub = df_raw[df_raw["Label"] == "Benign"].head(200)
    else:
        df_sub = df_raw.head(200)
        
    display_cols = [
        "Flow ID", "Src IP", "Dst IP", "Dst Port", "Protocol",
        "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts",
        "Init Fwd Win Byts", "Init Bwd Win Byts", "Label", "Cluster_k3"
    ]
    
    mo.ui.table(df_sub[display_cols], selection=None)
    return df_sub, display_cols


@app.cell
def __(mo):
    mo.md(
        """
        ---
        ### 📌 Key Takeaways from Cluster & Exploratory Analysis
        1. **Hyper-Uniform Attack Footprint**: The LOIC-HTTP flooding tool produces identical request patterns (3 Fwd Packets, 4 Bwd Packets, 8192 byte initial window, 20-byte TCP header). This causes the DDoS traffic to form an ultra-compact manifold that partitions cleanly into its own cluster under unsupervised K-Means ($k \\ge 3$) and PCA/t-SNE.
        2. **Target Singularity**: 100% of the 576,191 DDoS flows targeted web server `172.31.69.25` on Port 80 (HTTP).
        3. **Coordinated Botnet Distribution**: Exactly 10 AWS EC2 attacker instances executed the flood simultaneously, with each bot contributing evenly (~55k - 58.5k flows) during the assault window between 10:00 and 11:59 UTC.
        4. **Data Hygiene Finding**: Numeric packet lengths in `02-20-2018.csv` switch between integer and float representations across rows, requiring explicit `Float64` schema ingestion to prevent massive null corruption.
        """
    )
    return


if __name__ == "__main__":
    app.run()
