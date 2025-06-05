import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_CSV = "dataset/nlp/bert_article_embeddings_umap.csv"
OUTPUT_HTML = "visualization/plots/nlp_umap_article_embeddings.html"
PLOT_TITLE = "Interactive UMAP of Articles (click a dot to open its URL)"
PLOT_WIDTH = 900
PLOT_HEIGHT = 700
DOT_SIZE = 6

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)

    source = ColumnDataSource(df)

    hover = HoverTool(
        tooltips=[
            ("URL", "@url"),
            ("Source", "@source")
        ]
    )

    tap_callback = CustomJS(
        args=dict(source=source),
        code="""
        const inds = source.selected.indices;
        if (inds.length == 0) {
            return;
        }
        const idx = inds[0];
        const url = source.data['url'][idx];
        // Open it in a new browser tab
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

    p.circle(
        x="UMAP0",
        y="UMAP1",
        source=source,
        size=DOT_SIZE,
        alpha=0.6,
        line_color=None,
        fill_color="navy"
    )

    p.add_tools(TapTool(callback=tap_callback))

    p.xaxis.axis_label = "UMAP Dimension 0"
    p.yaxis.axis_label = "UMAP Dimension 1"
    p.grid.grid_line_alpha = 0.3

    output_file(OUTPUT_HTML, title="UMAP Interactive Visualization")
    save(p)

    logging.info(f"✅ Interactive UMAP saved to: {OUTPUT_HTML}")
