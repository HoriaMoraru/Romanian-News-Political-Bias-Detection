import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS, LabelSet, CategoricalColorMapper
from sklearn.cluster import KMeans
import bokeh.palettes as bp
import umap
import logging

# Configuration
INPUT_CSV = "dataset/nlp/bert_article_embeddings_umap.csv"
OUTPUT_HTML = "visualization/plots/nlp_umap_source_embeddings.html"
PLOT_TITLE = "Clustered Source Centroids in UMAP Space"
PLOT_WIDTH = 900
PLOT_HEIGHT = 700
DOT_SIZE = 12
ALPHA = 0.8
N_CLUSTERS = 2
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

df = pd.read_csv(INPUT_CSV)
embedding_cols = [col for col in df.columns if col.startswith("UMAP") and col[4:].isdigit() and int(col[4:]) < 15]
source_embeddings = df.groupby("source")[embedding_cols].mean().reset_index()

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
source_embeddings["cluster"] = kmeans.fit_predict(source_embeddings[embedding_cols]).astype(str)

umap_2d = umap.UMAP(n_components=2, metric="cosine", random_state=RANDOM_STATE).fit_transform(source_embeddings[embedding_cols])
source_embeddings["UMAP2D_0"] = umap_2d[:, 0]
source_embeddings["UMAP2D_1"] = umap_2d[:, 1]

palette = bp.Category10[10] if N_CLUSTERS <= 10 else bp.viridis(N_CLUSTERS)
color_mapper = CategoricalColorMapper(factors=sorted(source_embeddings["cluster"].unique()), palette=palette)

source_bokeh = ColumnDataSource(source_embeddings)

hover = HoverTool(tooltips=[("Source", "@source"), ("# Cluster", "@cluster")])
tap_callback = CustomJS(args=dict(source=source_bokeh), code="""
    const inds = source.selected.indices;
    if (inds.length === 0) return;
    const idx = inds[0];
    const domain = source.data['source'][idx];
    const url = "https://" + domain;
    window.open(url);
""")

p = figure(
    title=PLOT_TITLE,
    width=PLOT_WIDTH,
    height=PLOT_HEIGHT,
    tools=[hover, "tap", "pan", "wheel_zoom", "box_zoom", "reset"],
    active_drag="pan",
    active_scroll="wheel_zoom"
)

p.scatter(
    x="UMAP2D_0",
    y="UMAP2D_1",
    source=source_bokeh,
    size=DOT_SIZE,
    fill_color={"field": "cluster", "transform": color_mapper},
    line_color="black",
    alpha=ALPHA
)

labels = LabelSet(
    x="UMAP2D_0",
    y="UMAP2D_1",
    text="source",
    source=source_bokeh,
    text_font_size="8px",
    x_offset=5,
    y_offset=5
)
p.add_layout(labels)
p.add_tools(TapTool(callback=tap_callback))
p.xaxis.axis_label = "UMAP 2D Dim 0"
p.yaxis.axis_label = "UMAP 2D Dim 1"
p.grid.grid_line_alpha = 0.3

output_file(OUTPUT_HTML, title=PLOT_TITLE)
save(p)

logging.info(f"✅ Interactive clustered source-centroid UMAP saved to: {OUTPUT_HTML}")
