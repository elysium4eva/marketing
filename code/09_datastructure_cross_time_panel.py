"""09_datastructure_cross_time_panel.py -- data structures: cross-section / time series / panel data."""  # module comment
import numpy as np                                                    # numerical computing
import pandas as pd                                                   # tabular handling
import matplotlib.pyplot as plt                                       # plotting
from _common import RNG, save, dump_csv                               # shared helpers

# ---------------------------------------------------------------- 1) cross-section data: GDP per capita of 30 cities in a single year
cities = ["Shenzhen", "Shanghai", "Beijing", "Guangzhou", "Hangzhou", "Suzhou", "Nanjing", "Ningbo", "Wuxi", "Wuhan",  # city names (cross-section units)
          "Chengdu", "Changsha", "Qingdao", "Tianjin", "Zhengzhou", "Hefei", "Fuzhou", "Jinan", "Xi'an", "Chongqing",  # keep listing cities
          "Dongguan", "Foshan", "Changzhou", "Nantong", "Dalian", "Shenyang", "Kunming", "Nanchang", "Guiyang", "Harbin"]  # 30 units in total
gdp = RNG.normal(150000, 32000, len(cities)).clip(60000, 260000)        # GDP per capita (CNY): a one-time cross-section observation
cross = pd.DataFrame({"city": cities, "year": 2025,                     # unit id + observation year (the same year)
                      "gdp_per_capita": gdp.round(0)})                  # the observed variable
dump_csv(cross, "structure_cross_section.csv")                          # dump the cross-section data

# ---------------------------------------------------------------- 2) time series: 120 months of consumer price index
months = pd.period_range("2016-01", periods=120, freq="M").astype(str)  # build 120 month labels starting from 2016-01
trend = np.linspace(100.0, 118.0, 120)                                  # a slowly rising long-term trend
season = 0.7 * np.sin(2 * np.pi * (np.arange(120) % 12) / 12)           # annual seasonality (period 12)
cpi = trend + season + RNG.normal(0, 0.35, 120)                         # trend + season + noise = the observed value
ts = pd.DataFrame({"month": months, "cpi": cpi.round(2),                # time label + observed value
                   "yoy_pct": (pd.Series(cpi).pct_change(12) * 100).round(2)})  # YoY growth (lag of 12 periods)
dump_csv(ts, "structure_time_series.csv")                               # dump the time series

# ---------------------------------------------------------------- 3) panel data: 8 firms x 10 years
firms = ["F%02d" % i for i in range(1, 9)]                              # 8 cross-section units (firms)
years = list(range(2016, 2026))                                         # 10 time points (years)
rec = []                                                                # initialize the record list
for fi, f in enumerate(firms):                                          # loop over each firm
    base = 50 + 12 * fi                                                 # each firm has a different initial revenue scale
    growth = RNG.normal(0.08, 0.05)                                     # each firm has a different average growth (individual effect)
    for y in years:                                                     # loop over each year
        rec.append({"firm": f, "year": y,                          # the panel's two-way id: unit + time
                    "revenue": round(base * (1 + growth) ** (y - 2016)  # revenue compounds at the firm's own growth rate
                                     * np.exp(RNG.normal(0, 0.06)), 2),  # add an annual shock (noise)
                    "roa": round(RNG.normal(0.06 + 0.004 * fi, 0.015), 4)})  # another dependent variable: return on assets
panel = pd.DataFrame(rec)                                               # assemble the long-format panel data
dump_csv(panel, "structure_panel.csv")                                  # dump the panel data

# ---------------------------------------------------------------- 4) visualization
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9))                     # one row, three columns: cross-section / time series / panel
top = cross.sort_values("gdp_per_capita", ascending=False).head(12)      # sort the cross-section by size and take the top 12
axes[0].barh(top["city"][::-1], top["gdp_per_capita"][::-1],            # left: horizontal bar chart of the cross-section units (sorted)
            color="#8A0C3C", alpha=0.85)                                # unified color
axes[0].set_title("Cross-section: GDP per capita (top 12)", fontsize=10)  # left title
axes[0].set_xlabel("CNY / person")                                      # left x-axis meaning
axes[0].tick_params(axis="y", labelsize=8)                              # shrink the y-axis label font

axes[1].plot(range(120), ts["cpi"], color="#8A0C3C", lw=1.2,           # middle: time-series line (long-term trend)
             label="CPI index")                                         # legend: the level series
axes[1].plot(range(120), ts["cpi"].rolling(12).mean(),                 # overlay a 12-period moving average
             color="#C9A227", lw=2.0, label="12-period moving average")  # smoothing makes the trend clearer
axes[1].set_title("Time series: 120 months of CPI", fontsize=10)       # middle title
axes[1].set_xlabel("Month n (from 2016-01)")                            # middle x-axis meaning
axes[1].legend(fontsize=8)                                              # show the legend

for f in firms:                                                         # right: draw one line per firm
    sub = panel[panel["firm"] == f]                                     # take this firm's time series
    axes[2].plot(sub["year"], sub["revenue"], "-o", ms=2.5, lw=1.0)     # a panel = many time series shown together as small multiples
axes[2].set_title("Panel: 8 firms x 10 years", fontsize=10)            # right title
axes[2].set_xlabel("Year")                                              # right x-axis meaning
axes[2].set_ylabel("Revenue / 100M CNY")                                # right y-axis meaning
plt.tight_layout()                                                      # adjust the layout
save(fig, "fig_structures.png")                                         # save the figure
