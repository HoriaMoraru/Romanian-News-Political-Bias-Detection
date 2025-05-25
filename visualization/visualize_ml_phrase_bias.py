import pandas as pd
import numpy as np
import plotly.express as px

DOMAIN_EMBEDDINGS_FILE = "dataset/ml/phrase_embeddings.csv"
PLOT_OUTPUT_FILE = "visualization/plots/media_phrases_bias_space.html"

phrase_embeddings = pd.read_csv(DOMAIN_EMBEDDINGS_FILE, index_col=0)

# Create DataFrame with first 2 bias dimensions
df = pd.DataFrame({
    "phrase": phrase_embeddings.index,
    "bias_dim_1": phrase_embeddings["bias_dim_1"],
    "bias_dim_2": phrase_embeddings["bias_dim_2"],
})

# Compute bias strength and select top 300
df["bias_strength"] = np.sqrt(df["bias_dim_1"]**2 + df["bias_dim_2"]**2)
df = df.nlargest(300, "bias_strength")

# Center for better spread
df["bias_dim_1"] -= df["bias_dim_1"].mean()
df["bias_dim_2"] -= df["bias_dim_2"].mean()

# Plot
fig = px.scatter(
    df,
    x="bias_dim_1",
    y="bias_dim_2",
    hover_name="phrase",
    title="Top 300 Phrases in Bias Space",
    width=1000,
    height=700
)

fig.update_traces(marker=dict(size=5, opacity=0.7))
fig.write_html(PLOT_OUTPUT_FILE)
