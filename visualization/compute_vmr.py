import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PHRASE_FREQUENCY_MATRIX_FILE = "dataset/ml/phrase_domain_frequency_matrix_v2.csv"
VMR_HISOTRAM_FILE = "visualization/plots/variance_to_mean_ratio_histogram.png"

logging.info(f"Loading phrase frequency matrix from {PHRASE_FREQUENCY_MATRIX_FILE}...")
phrase_frequency_matrix = pd.read_csv(PHRASE_FREQUENCY_MATRIX_FILE, index_col=0)

phrase_means = phrase_frequency_matrix.mean(axis=1)
phrase_vars = phrase_frequency_matrix.var(axis=1)

vmr = phrase_vars / (phrase_means + 1e-6)  # add small epsilon to avoid div by 0

print(f"VMR mean: {vmr.mean():.2f}")
print(f"% of phrases with VMR > 1.5: {(vmr > 1.5).mean() * 100:.2f}%")
print(f"% of phrases with VMR > 5.0: {(vmr > 5.0).mean() * 100:.2f}%")

plt.hist(vmr, bins=100, range=(0, 10), color='steelblue')
plt.title("Variance-to-Mean Ratio (VMR) across phrases")
plt.xlabel("VMR")
plt.ylabel("Number of phrases")
plt.grid(True)
plt.savefig(VMR_HISOTRAM_FILE, bbox_inches='tight')
