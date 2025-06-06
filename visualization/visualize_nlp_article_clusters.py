import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS, CategoricalColorMapper
import bokeh.palettes as bp
import umap
import logging

# ───────────────────────────────────────────────────────────────────────────────
INPUT_CSV = "dataset/nlp/bert_article_embeddings_umap_kmeans_clusters.csv"
OUTPUT_HTML = "visualization/plots/nlp_umap_article_embeddings_clusters.html"
PLOT_TITLE = "UMAP 20D → 2D + Clusters (click a dot to open article)"
PLOT_WIDTH = 1000
PLOT_HEIGHT = 750
DOT_SIZE = 6
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    UMAP_COLS = [col for col in df.columns if col.startswith("UMAP") and col[4:].isdigit()]

    df["cluster"] = df["cluster"].astype(str)

    logging.info("Reducing 20D UMAP to 2D for visualization...")
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    umap_2d = reducer.fit_transform(df[UMAP_COLS].values)
    df["UMAP2D_0"] = umap_2d[:, 0]
    df["UMAP2D_1"] = umap_2d[:, 1]

    unique_clusters = sorted(df["cluster"].unique())
    palette = ["#1f77b4", "#d62728"]  # blue / red
    color_mapper = CategoricalColorMapper(factors=unique_clusters, palette=palette)

    source = ColumnDataSource(df)

    hover = HoverTool(tooltips=[
        ("Source", "@source"),
        ("Cluster", "@cluster"),
        ("URL", "@url"),
    ])

    tap_callback = CustomJS(args=dict(source=source), code="""
        const indices = source.selected.indices;
        if (indices.length > 0) {
            const url = source.data['url'][indices[0]];
            window.open(url, "_blank");
        }
    """)

    p = figure(
        title=PLOT_TITLE,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        tools=[hover, "tap", "pan", "wheel_zoom", "box_zoom", "reset"],
        active_drag="pan",
        active_scroll="wheel_zoom"
    )

    p.circle(
        x="UMAP2D_0",
        y="UMAP2D_1",
        source=source,
        size=DOT_SIZE,
        alpha=0.7,
        line_color=None,
        fill_color={"field": "cluster", "transform": color_mapper}
    )

    p.add_tools(TapTool(callback=tap_callback))

    p.xaxis.axis_label = "UMAP 2D Dim 0"
    p.yaxis.axis_label = "UMAP 2D Dim 1"
    p.grid.grid_line_alpha = 0.3

    output_file(OUTPUT_HTML, title=PLOT_TITLE)
    save(p)

    logging.info(f"✅ Interactive UMAP plot saved to {OUTPUT_HTML}")
