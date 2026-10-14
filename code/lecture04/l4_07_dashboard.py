"""l4_07_dashboard.py -- build a dashboard in Python (instead of Power BI / Tableau).

This lecture does not use commercial BI software: the dashboard is assembled from plotly subplots
plus KPI cards and exported as a single-file HTML that is interactive on open (hover, zoom, legend
filtering); a static matplotlib version is also drawn for easy inclusion in slides and printing.
Both artifacts use the same data and the same definitions.
"""
import numpy as np  # numeric computation
import pandas as pd  # data aggregation
import matplotlib.pyplot as plt  # static dashboard
import plotly.graph_objects as go  # low-level interface: precise subplot assembly
from plotly.subplots import make_subplots  # dashboard layout
from _common_l4 import INT, DAT, save, dump  # reuse the shared helpers and the interactive directory

# ---------------------------------------------------------------- 1) read in the samples generated earlier
survey = pd.read_csv(DAT / "l4_survey_raw.csv")  # survey sample (from l4_01)
cities = pd.read_csv(DAT / "l4_city_gdp.csv")  # city sample (from l4_05)
energy = pd.read_csv(DAT / "l4_energy.csv", parse_dates=["month"])  # energy sample (from l4_06)

# ---------------------------------------------------------------- 2) KPI metrics
kpi = {  # the four key metrics at the top of the dashboard
    "sample size": f"{len(survey):,}",  # number of respondents
    "mean satisfaction": f"{survey['satisfaction'].mean():.2f} / 5",  # mean satisfaction
    "repurchase rate": f"{survey['revisit'].mean():.1%}",  # repurchase share
    "median GDP per capita": f"{cities['gdp_per_cap'].median() / 1e4:.1f} 10k CNY",  # median city GDP per capita
}  # end of the KPI dictionary
print("KPI:", kpi)  # print the key metrics
dump(pd.DataFrame([{"metric": k, "value": v} for k, v in kpi.items()]), "l4_dashboard_kpi.csv")  # write the KPI to disk

# ---------------------------------------------------------------- 3) plotly interactive dashboard
fig = make_subplots(rows=2, cols=2,  # 2x2 panels
                    subplot_titles=("Satisfaction by channel", "Income vs spend",  # four subplot titles
                                    "Monthly generation (by energy)", "City GDP vs population"))  # more titles
for ch, c in zip(survey["channel"].unique(), ["#8A0C3C", "#C9A227", "#2E6F8E"]):  # draw a box per channel
    sub = survey.loc[survey["channel"] == ch, "satisfaction"].dropna()  # take this channel's satisfaction
    fig.add_trace(go.Box(y=sub, name=ch, marker_color=c, boxmean=True),  # boxplot (with a mean marker)
                  row=1, col=1)  # place in the top-left cell
fig.add_trace(go.Scatter(x=survey["income_k"], y=survey["spend_k"],  # income vs spend scatter
                         mode="markers", marker=dict(size=5, opacity=0.5, color="#711426"),  # semi-transparent points
                         name="respondent"), row=1, col=2)  # place in the top-right cell
for src, c in zip(energy["source"].unique(), ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):  # draw a trend per energy type
    sub = energy[energy["source"] == src]  # take this energy series
    fig.add_trace(go.Scatter(x=sub["month"], y=sub["generation_twh"],  # line
                             mode="lines", name=src, line=dict(color=c)),  # color and legend
                  row=2, col=1)  # place in the bottom-left cell
fig.add_trace(go.Scatter(x=cities["pop_wan"], y=cities["gdp_yi"],  # city population vs GDP
                         mode="markers", text=cities["city"],  # show the city name on hover
                         marker=dict(size=8, color=cities["growth_pct"],  # point size and color encoding
                                     colorscale="RdPu", showscale=True),  # use a continuous palette for growth
                         name="city"), row=2, col=2)  # place in the bottom-right cell
fig.update_layout(title_text="Consumer survey and energy/city metrics dashboard (Python + plotly)",  # overall title
                  height=760, showlegend=True, template="plotly_white",  # height and theme
                  title_x=0.02)  # left-align the title
fig.write_html(INT / "dashboard.html", include_plotlyjs="cdn")  # export a single-file HTML
print("[interactive]", INT / "dashboard.html")  # print the artifact path

# ---------------------------------------------------------------- 4) matplotlib static dashboard
fig2 = plt.figure(figsize=(13.2, 6.4))  # canvas: KPI at the top + 2x2 panels
gs = fig2.add_gridspec(2, 4, height_ratios=[0.28, 1.0], hspace=0.42, wspace=0.35)  # custom grid
for i, (k, v) in enumerate(kpi.items()):  # four KPI cards
    ax_card = fig2.add_subplot(gs[0, i])  # one cell per card
    ax_card.axis("off")  # a card draws no axes
    ax_card.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_card.transAxes,  # card background box
                                    fc="#F7F3F2", ec="#D6CDD0", lw=0.8))  # light fill with a thin border
    ax_card.text(0.5, 0.62, v, ha="center", va="center", fontsize=15, color="#8A0C3C", weight="bold")  # value
    ax_card.text(0.5, 0.22, k, ha="center", va="center", fontsize=9.5, color="#66717D")  # metric name
ax1 = fig2.add_subplot(gs[1, 0])  # top-left: satisfaction by channel
data = [survey.loc[survey["channel"] == ch, "satisfaction"].dropna() for ch in survey["channel"].unique()]  # satisfaction series per channel (missing dropped)
bp = ax1.boxplot(data, patch_artist=True)  # boxplot
ax1.set_xticklabels(list(survey["channel"].unique()))  # group labels (use set_xticklabels from 3.9 on)
for patch, c in zip(bp["boxes"], ["#8A0C3C", "#C9A227", "#2E6F8E"]):  # color the boxes
    patch.set_facecolor(c); patch.set_alpha(0.6)  # fill and transparency
ax1.set_title("Satisfaction by channel"); ax1.set_ylabel("satisfaction (1-5)")  # title and axis label
ax2 = fig2.add_subplot(gs[1, 1])  # top-right: income vs spend
ax2.scatter(survey["income_k"], survey["spend_k"], s=8, color="#711426", alpha=0.45)  # scatter
ax2.set_title("Income vs spend"); ax2.set_xlabel("monthly income (thousand CNY)"); ax2.set_ylabel("monthly spend (thousand CNY)")  # title and axes
ax3 = fig2.add_subplot(gs[1, 2])  # bottom-left: energy trend
pivot = energy.pivot(index="month", columns="source", values="generation_twh")  # wide table
for src, c in zip(pivot.columns, ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):  # draw a line per energy type
    ax3.plot(pivot.index, pivot[src].rolling(3).mean(), lw=1.3, color=c, label=src)  # 3-month smoothing
ax3.set_title("Monthly generation (by energy)"); ax3.set_ylabel("TWh"); ax3.legend(fontsize=7.6, ncol=2)  # title and legend
ax4 = fig2.add_subplot(gs[1, 3])  # bottom-right: city population vs GDP
sc = ax4.scatter(cities["pop_wan"], cities["gdp_yi"], s=26,  # point size and color encode growth
                 c=cities["growth_pct"], cmap="RdPu")  # continuous palette
ax4.set_title("City GDP vs population"); ax4.set_xlabel("population (10k)"); ax4.set_ylabel("GDP (100 million CNY)")  # title and axes
plt.colorbar(sc, ax=ax4, fraction=0.046, label="growth %")  # colorbar
save(fig2, "fig_dashboard.png")  # save the static dashboard
print("dashboard KPI cards:", len(kpi), "| panels: 4")  # closing log
