"""02_datatype_categorical.py -- categorical data (nominal vs ordinal): sample generation + visualization."""  # module comment
import numpy as np                                                    # numerical computing
import pandas as pd                                                   # tabular handling
import matplotlib.pyplot as plt                                       # plotting
from _common import RNG, save, dump_csv                               # shared helpers

# ---------------------------------------------------------------- 1) generate samples
industries = ["Electronics", "Pharmaceuticals", "Machinery", "Consumer", "Finance", "Energy"]  # nominal categories: industry (no ordering)
ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "C"]                   # ordinal categories: credit rating (clear ordering)
n = 300                                                               # sample size: 300 listed companies
ind = RNG.choice(industries, size=n, p=[0.28, 0.13, 0.18, 0.16, 0.15, 0.10])  # draw industries by the given proportions (nominal)
rate = RNG.choice(ratings, size=n, p=[0.03, 0.09, 0.18, 0.30, 0.22, 0.13, 0.05])  # draw ratings by the given proportions (ordinal)
year = RNG.choice([2023, 2024, 2025], size=n)                          # record the year for the cross-tab
df = pd.DataFrame({"industry": ind, "rating": rate, "year": year})     # assemble the sample table
dump_csv(df, "categorical_samples.csv")                                # dump to CSV (repo data/ folder)

# ---------------------------------------------------------------- 2) visualization
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                    # one row, three columns: nominal bar / ordinal bar / cross heatmap
counts_ind = df["industry"].value_counts().reindex(industries)         # count each industry (keep the given order)
axes[0].bar(industries, counts_ind.values, color="#8A0C3C", alpha=0.85)  # left: frequency bar chart of nominal categories
axes[0].set_title("Nominal: industry distribution")                    # left title
axes[0].tick_params(axis="x", rotation=30)                             # rotate the x labels to avoid overlap

counts_rate = df["rating"].value_counts().reindex(ratings)              # count each rating (in the ordered AAA to C sequence)
axes[1].bar(ratings, counts_rate.values, color="#C9A227", alpha=0.9)    # middle: frequency bar chart of ordinal categories
axes[1].set_title("Ordinal: credit rating (order matters)")            # middle title

cross = pd.crosstab(df["industry"], df["rating"]).reindex(index=industries, columns=ratings)  # industry x rating cross-tab
im = axes[2].imshow(cross.values, cmap="RdPu", aspect="auto")           # right: cross-tab heatmap
axes[2].set_xticks(range(len(ratings)), ratings, fontsize=8)            # right x-axis: rating ticks
axes[2].set_yticks(range(len(industries)), industries, fontsize=8)      # right y-axis: industry ticks
axes[2].set_title("Category cross: industry x rating")                 # right title
axes[2].grid(False)                                                     # a heatmap needs no grid lines
plt.colorbar(im, ax=axes[2], fraction=0.046)                            # add a colorbar for the frequency
plt.tight_layout()                                                      # adjust the layout
save(fig, "fig_categorical.png")                                        # save the figure
