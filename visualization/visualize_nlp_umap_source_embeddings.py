import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS, LabelSet
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_CSV = "dataset/nlp/bert_article_embeddings_umap.csv"
OUTPUT_HTML = "visualization/plots/umap_sources_interactive.html"
PLOT_TITLE = "Source Centroids in UMAP Space"
PLOT_WIDTH = 900
PLOT_HEIGHT = 700
DOT_SIZE = 12
DOT_COLOR = "navy"
ALPHA = 0.8

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)

    grouped = (
        df.groupby("source")
          .agg(
              centroid_x=("UMAP0", "mean"),
              centroid_y=("UMAP1", "mean"),
              count=("UMAP0", "size")
          )
          .reset_index()
    )

    source_bokeh = ColumnDataSource(grouped)

    hover = HoverTool(
        tooltips=[
            ("Source", "@source"),
            ("# Articles", "@count")
        ]
    )

    tap_callback = CustomJS(
        args=dict(source=source_bokeh),
        code="""
        const inds = source.selected.indices;
        if (inds.length === 0) {
            return;
        }
        const idx = inds[0];
        const domain = source.data['source'][idx];
        const url = "https://" + domain;
        window.open(url);
        """
    )

    p = figure(
        title=PLOT_TITLE,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        tools=[hover, "tap", "pan", "wheel_zoom", "box_zoom", "reset"],
        active_drag="pan",
        active_scroll="wheel_zoom"
    )

    p.scatter(
        x="centroid_x",
        y="centroid_y",
        source=source_bokeh,
        size=DOT_SIZE,
        fill_color=DOT_COLOR,
        line_color="black",
        alpha=ALPHA
    )

    labels = LabelSet(
        x="centroid_x",
        y="centroid_y",
        text="source",
        source=source_bokeh,
        text_font_size="8px",
        x_offset=5,
        y_offset=5
    )
    p.add_layout(labels)

    p.add_tools(TapTool(callback=tap_callback))

    p.xaxis.axis_label = "UMAP Dimension 0 (centroid)"
    p.yaxis.axis_label = "UMAP Dimension 1 (centroid)"
    p.grid.grid_line_alpha = 0.3

    output_file(OUTPUT_HTML, title="UMAP Source Centroids")
    save(p)

    logging.info(f"✅ Interactive source-centroid UMAP saved to: {OUTPUT_HTML}")
