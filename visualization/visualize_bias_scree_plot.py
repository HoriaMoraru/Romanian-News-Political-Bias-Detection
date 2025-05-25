import matplotlib.pyplot as plt
import pandas as pd

SVD_FILE = "dataset/ml/svd.csv"
PLOT_OUTPUT_FILE = "visualization/plots/svd_scree_plot.png"

df = pd.read_csv(SVD_FILE)
plt.plot(df["bias_dim"], df["singular_value"], marker="o")
plt.title("Scree Plot of Singular Values")
plt.xlabel("Bias Dimension")
plt.ylabel("Singular Value")
plt.grid(True)
plt.savefig(PLOT_OUTPUT_FILE, dpi=300)
