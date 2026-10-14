"""l4_05_chart_zoo.py -- chart zoo: five univariate charts + an overview of seven common chart types.

The top half answers "what do you see when you redraw the same column a different way"; the bottom
half turns the original course's chart list (bar chart, distribution, trend, scatter, correlation
matrix, heatmap, flow/network graph) into a single reference overview, and puts a "choose a chart
by question" hint in the last cell.
"""
import numpy as np  # numeric computation
import pandas as pd  # table handling
import matplotlib.pyplot as plt  # every chart type is drawn with matplotlib
from _common_l4 import RNG, OKABE, save, dump  # reuse the shared helpers and the colorblind-friendly palette

# ---------------------------------------------------------------- sample 1: city economic data
CITY_NAMES = ["Shenzhen", "Shanghai", "Beijing", "Guangzhou", "Chongqing", "Suzhou", "Chengdu", "Hangzhou", "Wuhan", "Nanjing",  # first line of the city-name list
              "Ningbo", "Qingdao", "Wuxi", "Changsha", "Zhengzhou", "Foshan", "Hefei", "Fuzhou", "Jinan", "Xi'an",  # second line of the city-name list
              "Dongguan", "Quanzhou", "Nantong", "Changzhou", "Dalian", "Shenyang", "Kunming", "Nanchang", "Guiyang", "Harbin",  # third line of the city-name list
              "Xiamen", "Shijiazhuang", "Taiyuan", "Nanning", "Xuzhou", "Wenzhou", "Yantai", "Zhuhai", "Huizhou", "Zhongshan"]  # 40 cities
M = len(CITY_NAMES)  # number of cities
base_gdp = RNG.lognormal(mean=9.3, sigma=0.45, size=M)  # city GDP (100 million CNY, right-skewed)
pop = RNG.lognormal(mean=6.1, sigma=0.4, size=M)  # resident population (10k)
cities = pd.DataFrame({  # assemble the city table
    "city": CITY_NAMES,  # city name
    "province": RNG.choice(["Guangdong", "Jiangsu", "Zhejiang", "Shandong", "Sichuan", "Hubei", "Fujian", "Other"], M),  # province (abbreviated)
    "gdp_yi": np.round(base_gdp, 1),  # GDP (100 million CNY)
    "pop_wan": np.round(pop, 1),  # population (10k)
    "lat": np.round(RNG.uniform(22.5, 45.5, M), 3),  # latitude (used by the map)
    "lon": np.round(RNG.uniform(103.0, 126.0, M), 3),  # longitude (used by the map)
    "growth_pct": np.round(RNG.normal(5.2, 1.6, M), 2),  # growth rate (%)
})  # end of the city-table definition
cities["gdp_per_cap"] = (cities["gdp_yi"] * 1e8 / (cities["pop_wan"] * 1e4)).round(0)  # GDP per capita (CNY)
dump(cities, "l4_city_gdp.csv")  # write the city sample to disk
counts = pd.DataFrame({  # sample 2: category counts
    "category": ["Electronics", "Pharmaceuticals", "Machinery", "Consumer", "Finance", "Energy", "Logistics"],  # seven industries
    "count": [128, 64, 92, 110, 47, 38, 55]})  # number of firms per industry
dump(counts, "l4_category_counts.csv")  # write the category counts to disk

# ---------------------------------------------------------------- figure A: five ways to draw the same variable
gdp = cities["gdp_yi"].values  # single variable: city GDP
fig, axes = plt.subplots(1, 5, figsize=(13.6, 3.5))  # one row, five cells
axes[0].hist(gdp, bins=20, color="#8A0C3C", alpha=0.85)  # (1) histogram: view the distribution shape
axes[0].set_title("(1) Histogram")  # subplot title
axes[0].set_xlabel("GDP (100 million CNY)"); axes[0].set_ylabel("number of cities")  # axis labels
axes[1].boxplot(gdp, patch_artist=True,  # (2) boxplot: view quartiles and outliers
                boxprops=dict(facecolor="#C9A227", alpha=0.6))  # box fill
axes[1].set_title("(2) Boxplot")  # subplot title
axes[1].set_ylabel("GDP (100 million CNY)")  # axis label
axes[2].violinplot(gdp, showmeans=True)  # (3) violin plot: view density and shape
axes[2].set_title("(3) Violin plot")  # subplot title
axes[2].set_ylabel("GDP (100 million CNY)")  # axis label
sorted_gdp = np.sort(gdp)  # sort ascending for the ECDF
ecdf = np.arange(1, len(sorted_gdp) + 1) / len(sorted_gdp)  # cumulative probability
axes[3].step(sorted_gdp, ecdf, where="post", color="#2E6F8E", lw=1.8)  # (4) empirical distribution function
axes[3].set_title("(4) Empirical CDF (ECDF)")  # subplot title
axes[3].set_xlabel("GDP (100 million CNY)"); axes[3].set_ylabel("cumulative share")  # axis labels
axes[4].axis("off")  # (5) stem-and-leaf chart rendered as text
stems = {}  # group by tens digit
for v in np.round(sorted_gdp, 0).astype(int):  # loop over each observation
    stems.setdefault(v // 1000, []).append(v % 1000 // 100)  # thousands digit as the stem, hundreds digit as the leaf
lines = [f"{s} | " + " ".join(str(x) for x in vals) for s, vals in sorted(stems.items())[:9]]  # join into text lines
axes[4].text(0.02, 0.95, "(5) Stem-and-leaf (thousands | hundreds)\n\n" + "\n".join(lines),  # top-left text
             family=["Consolas", "Microsoft YaHei"], fontsize=7.6,  # monospace font + per-glyph fallback for CJK
             va="top", ha="left")  # top alignment
plt.tight_layout()  # adjust the five cells
save(fig, "fig_univariate_zoo.png")  # save: the univariate five-piece set

# ---------------------------------------------------------------- figure B: overview of seven common chart types
fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.1))  # two rows, four columns, eight cells
top = cities.nlargest(8, "gdp_yi")  # top eight cities by GDP
axes[0, 0].bar(top["city"], top["gdp_yi"], color="#8A0C3C", alpha=0.9)  # (1) bar chart: compare magnitudes
axes[0, 0].set_title("(1) Bar chart: compare")  # subplot title
axes[0, 0].tick_params(axis="x", rotation=45, labelsize=7.5)  # rotate the x-axis labels
axes[0, 1].hist(cities["gdp_per_cap"], bins=18, color="#C9A227", alpha=0.9)  # (2) distribution
axes[0, 1].set_title("(2) Distribution: view the shape")  # subplot title
axes[0, 1].set_xlabel("GDP per capita (CNY)", fontsize=9)  # axis label
trend = cities.sort_values("gdp_yi")["gdp_yi"].values  # (3) trend: demonstrate with a sorted series
axes[0, 2].plot(range(len(trend)), trend, "-", color="#2E6F8E", lw=1.8)  # line
axes[0, 2].set_title("(3) Trend: view the change")  # subplot title
axes[0, 2].set_xlabel("rank (sorted by GDP)", fontsize=9)  # axis label
axes[0, 3].scatter(cities["pop_wan"], cities["gdp_yi"], s=22,  # (4) scatter: view the relationship
                   color="#711426", alpha=0.7)  # point color and transparency
axes[0, 3].set_title("(4) Scatter: view the relationship")  # subplot title
axes[0, 3].set_xlabel("population (10k)", fontsize=9); axes[0, 3].set_ylabel("GDP (100 million CNY)", fontsize=9)  # axis labels
corr_cols = ["gdp_yi", "pop_wan", "gdp_per_cap", "growth_pct"]  # (5) correlation matrix: four numeric columns
corr = cities[corr_cols].corr()  # compute the correlation matrix
im = axes[1, 0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)  # draw the heatmap
axes[1, 0].set_xticks(range(len(corr_cols)), ["GDP", "Population", "Per-capita", "Growth"], fontsize=8)  # x-axis
axes[1, 0].set_yticks(range(len(corr_cols)), ["GDP", "Population", "Per-capita", "Growth"], fontsize=8)  # y-axis
axes[1, 0].grid(False)  # a heatmap does not need a grid
axes[1, 0].set_title("(5) Correlation matrix: pairwise relations", fontsize=9.5)  # subplot title
axes[1, 1].barh(counts["category"], counts["count"], color=OKABE[:len(counts)], alpha=0.9)  # (6) category bar chart
axes[1, 1].set_title("(6) Bar chart: composition and ranking", fontsize=9.5)  # subplot title
axes[1, 1].tick_params(axis="y", labelsize=8)  # y-axis label font size
nodes = {"Upstream": (0.2, 0.7), "Midstream": (0.5, 0.75), "Downstream": (0.8, 0.7),  # (7) network: illustrated with three nodes
         "Overseas": (0.5, 0.25), "Domestic": (0.2, 0.3), "Substitutes": (0.8, 0.3)}  # six node coordinates
edges = [("Upstream", "Midstream"), ("Midstream", "Downstream"), ("Upstream", "Domestic"),  # supply-chain links
         ("Midstream", "Overseas"), ("Downstream", "Substitutes")]  # more edges
for a, b in edges:  # draw each edge
    axes[1, 2].annotate("", xy=nodes[b], xytext=nodes[a],  # from the source to the target
                        arrowprops=dict(arrowstyle="->", color="#66717D", lw=1.4))  # gray arrow
for name, (x, y) in nodes.items():  # draw each node
    axes[1, 2].scatter([x], [y], s=420, color="#8A0C3C", alpha=0.85, zorder=3)  # node dot
    axes[1, 2].text(x, y, name, color="white", ha="center", va="center", fontsize=8.6, zorder=4)  # node text
axes[1, 2].set_xlim(0, 1); axes[1, 2].set_ylim(0, 1)  # fixed coordinate range
axes[1, 2].axis("off")  # turn off the axes
axes[1, 2].set_title("(7) Network: structure and flows")  # subplot title
axes[1, 3].axis("off")  # (8) chart-selection hints rendered as text
axes[1, 3].text(0.0, 0.95,  # start at the top-left corner
                "(8) Choosing a chart by question\n"  # (8) chart-selection hints, given question by question
                "· Compare magnitudes -> bar chart (sort it!)\n"  # comparison questions -> bar chart
                "· View distribution -> histogram/boxplot/ECDF\n"  # distribution questions -> histogram/boxplot
                "· View change -> line chart (mind the baseline and smoothing)\n"  # time questions -> line chart
                "· View relationships -> scatter/correlation matrix (avoid overplotting)\n"  # relationship questions -> scatter
                "· View composition -> stacked bar/Sankey (use pie charts sparingly)\n"  # composition questions -> stacked bar/Sankey
                "· View space -> choropleth/symbol map\n"  # spatial questions -> map
                "· View structure -> network (few nodes, few colors)",  # structure questions -> network
                fontsize=8.6, va="top", ha="left", linespacing=1.5)  # slightly larger line spacing for readability
plt.tight_layout()  # adjust the eight cells
save(fig, "fig_chart_zoo.png")  # save: chart overview
print("cities:", cities.shape, "| top city:", top["city"].iloc[0])  # closing log
