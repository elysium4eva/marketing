"""l4_06_plotly_interactive.py -- interactive charts: interactive for exploration, static for reporting.

The script uses plotly to generate three interactive charts that open in a browser (line, scatter,
Sankey) and exports them as single-file HTML into the repository; it also uses matplotlib to draw a
"static vs interactive" comparison figure, explaining the scenarios each suits (exploration vs final report).
"""
import numpy as np  # numeric computation
import pandas as pd  # data-table handling
import matplotlib.pyplot as plt  # static comparison figure
import plotly.express as px  # plotly quick-plotting interface
import plotly.graph_objects as go  # low-level interface: Sankey and other specialized charts
from _common_l4 import RNG, INT, DAT, save, dump  # reuse the shared helpers, the interactive directory, and the data directory

# ---------------------------------------------------------------- 1) synthesize monthly energy data
months = pd.date_range("2015-01-01", periods=132, freq="MS")  # 132 months starting 2015-01
SOURCES = ["Coal", "Natural gas", "Hydro", "Wind+Solar"]  # four energy types
rows = []  # record per month and per energy type
for i, m in enumerate(months):  # loop over each month
    season = 1 + 0.12 * np.sin(2 * np.pi * (m.month - 1) / 12)  # seasonal swing (higher demand in winter and summer)
    trend = 1 + 0.0025 * i  # slowly rising long-term trend
    share = {"Coal": 0.58 - 0.0016 * i, "Natural gas": 0.16 + 0.0006 * i,  # coal's share falls year by year
             "Hydro": 0.14, "Wind+Solar": 0.12 + 0.0012 * i}  # clean energy's share rises
    for s in SOURCES:  # loop over the four energy types
        base = 320 * share[s]  # base generation for this energy type
        value = base * season * trend * np.exp(RNG.normal(0, 0.04))  # combine season, trend, and noise
        rows.append({"month": m, "source": s, "generation_twh": round(float(value), 2)})  # record one row
energy = pd.DataFrame(rows)  # summarize into a long table
dump(energy, "l4_energy.csv")  # write the sample to disk

# ---------------------------------------------------------------- 2) three interactive charts (exported to HTML)
line_fig = px.line(energy, x="month", y="generation_twh", color="source",  # line: colored by energy type
                   title="Monthly generation (hover for values, box-select to zoom)",  # the title spells out the interaction
                   labels={"month": "month", "generation_twh": "generation (TWh)", "source": "source"})  # axis names
line_fig.write_html(INT / "energy_trend.html", include_plotlyjs="cdn")  # single-file HTML (CDN reference)
print("[interactive]", INT / "energy_trend.html")  # print the artifact path
survey = pd.read_csv(DAT / "l4_survey_raw.csv")  # read the survey sample (from l4_01)
plot_df = survey.dropna(subset=["satisfaction", "income_k", "spend_k"])  # plotly does not accept missing values: drop those rows first
print("plotly sample:", len(plot_df), "(of", len(survey), "raw; missing rows excluded from interactive charts)")  # print the sample-size difference
scatter_fig = px.scatter(plot_df, x="income_k", y="spend_k", color="channel",  # scatter: colored by channel
                         size="satisfaction", hover_data=["city", "education"],  # show details on hover
                         title="Income vs spend (point size = satisfaction, zoom and filter)",  # title
                         labels={"income_k": "monthly income (thousand CNY)", "spend_k": "monthly spend (thousand CNY)", "channel": "channel"})  # axis names
scatter_fig.write_html(INT / "income_spend.html", include_plotlyjs="cdn")  # export the interactive scatter
print("[interactive]", INT / "income_spend.html")  # print the artifact path
flow = energy.groupby("source", as_index=False)["generation_twh"].sum()  # aggregate by energy type
flow["sector"] = ["Industry", "Residential", "Industry", "Transport"]  # simplified destination (teaching illustration)
labels = flow["source"].tolist() + flow["sector"].tolist()  # Sankey node labels (energy + sector)
sankey_fig = go.Figure(data=[go.Sankey(  # build the Sankey with graph_objects
    node=dict(label=labels, pad=14, thickness=16,  # node style
              color=["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"] * 2),  # node colors
    link=dict(source=list(range(len(flow))),  # link source indices
              target=[len(flow) + i for i in range(len(flow))],  # link target indices
              value=flow["generation_twh"].round(1).tolist()))])  # link width = value
sankey_fig.update_layout(title_text="Energy flow schematic (Sankey: good for composition and flow)",  # title
                         font=dict(size=12))  # font size
sankey_fig.write_html(INT / "energy_sankey.html", include_plotlyjs="cdn")  # export the Sankey
print("[interactive]", INT / "energy_sankey.html")  # print the artifact path

# ---------------------------------------------------------------- 3) static chart: for the final report
pivot = energy.pivot(index="month", columns="source", values="generation_twh")  # wide table for easy stacking
fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9))  # static on the left, interactive illustration on the right
for s, c in zip(SOURCES, ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):  # one color per energy type
    axes[0].plot(pivot.index, pivot[s].rolling(3).mean(), lw=1.6,  # draw a 3-month moving average (smooths noise)
                 color=c, label=s)  # color and legend
axes[0].set_title("Static chart: for the final report (fixed, printable)")  # subplot title
axes[0].set_xlabel("month"); axes[0].set_ylabel("generation (TWh)")  # axis labels
axes[0].legend(fontsize=8.4, ncol=2)  # two-column legend
axes[1].plot(pivot.index, pivot["Wind+Solar"], lw=1.6, color="#2E6F8E")  # the series "selected" in the interactive chart
axes[1].fill_between(pivot.index, pivot["Wind+Solar"].min() * 0.9,  # highlight a range to mimic box-select
                     pivot["Wind+Solar"].max() * 1.05, alpha=0.12, color="#C9A227")  # light highlight
axes[1].annotate("Hover for values\nBox-select to zoom\nClick the legend to filter series",  # annotate the interactions
                 xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9.2,  # text position
                 va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3",  # text background box
                                                fc="#F7F3F2", ec="#D6CDD0"))  # box colors
axes[1].set_title("Interactive chart: for exploration (zoom, filter, hover)")  # subplot title
axes[1].set_xlabel("month"); axes[1].set_ylabel("generation (TWh)")  # axis labels
plt.tight_layout()  # adjust the two cells
save(fig, "fig_interactive_vs_static.png")  # save: interactive vs static comparison
print("energy rows:", len(energy), "| interactive files ->", INT)  # closing log
