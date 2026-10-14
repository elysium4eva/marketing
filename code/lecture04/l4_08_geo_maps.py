"""l4_08_geo_maps.py -- geographic visualization: choropleth, symbol map, flow map, and spatial autocorrelation.

The script first synthesizes province-level and country-level indicators, then demonstrates four
geographic charts: (1) a choropleth (area fill); (2) a symbol map (bubble size/color encoding);
(3) a trade flow map (origin-to-destination lines); (4) a Moran scatter (spatial autocorrelation).
The China province-level boundary file is downloaded and cached at build time; if unavailable, the
script automatically falls back to a symbol map and the pipeline still runs.
"""
import json  # parse GeoJSON
import urllib.request  # download the boundary file (optional step)
import numpy as np  # numeric computation
import pandas as pd  # data processing
import matplotlib.pyplot as plt  # static maps
from matplotlib.collections import PatchCollection  # draw many polygons in batch
from matplotlib.patches import Polygon  # a single polygon
import plotly.express as px  # world map (interactive)
import plotly.graph_objects as go  # low-level interface: flow lines
from _common_l4 import RNG, DAT, INT, SEQ, save, dump  # reuse the shared helpers

# ---------------------------------------------------------------- 1) synthesize province-level and country-level indicators
PROVINCES = {  # short name, longitude, latitude, and code for the 31 province-level divisions
    "北京": (116.4, 39.9, "110000"), "天津": (117.2, 39.1, "120000"), "河北": (114.5, 38.0, "130000"),  # provinces of North and Northeast China (lon, lat, admin code)
    "山西": (112.5, 37.9, "140000"), "内蒙古": (111.7, 40.8, "150000"), "辽宁": (123.4, 41.8, "210000"),  # provinces of North and Northeast China (cont.)
    "吉林": (125.3, 43.9, "220000"), "黑龙江": (126.6, 45.8, "230000"), "上海": (121.5, 31.2, "310000"),  # provinces of Northeast and East China
    "江苏": (118.8, 32.1, "320000"), "浙江": (120.2, 30.3, "330000"), "安徽": (117.3, 31.9, "340000"),  # provinces of East China
    "福建": (119.3, 26.1, "350000"), "江西": (115.9, 28.7, "360000"), "山东": (117.0, 36.7, "370000"),  # provinces of East and Central China
    "河南": (113.6, 34.8, "410000"), "湖北": (114.3, 30.6, "420000"), "湖南": (113.0, 28.2, "430000"),  # provinces of Central China
    "广东": (113.3, 23.1, "440000"), "广西": (108.3, 22.8, "450000"), "海南": (110.3, 20.0, "460000"),  # provinces of South China
    "重庆": (106.5, 29.6, "500000"), "四川": (104.1, 30.7, "510000"), "贵州": (106.7, 26.6, "520000"),  # provinces of Southwest China
    "云南": (102.7, 25.0, "530000"), "西藏": (91.1, 29.6, "540000"), "陕西": (108.9, 34.3, "610000"),  # provinces of Southwest and Northwest China
    "甘肃": (103.8, 36.1, "620000"), "青海": (101.8, 36.6, "630000"), "宁夏": (106.3, 38.5, "640000"),  # provinces of Northwest China
    "新疆": (87.6, 43.8, "650000"),  # end of the province-level divisions
}  # end of the province dictionary
prov = pd.DataFrame([{"province": k, "lon": v[0], "lat": v[1],  # expand into a table
                      "code": v[2]} for k, v in PROVINCES.items()])  # name, coordinates, and code
prov["gdp_yi"] = np.round(RNG.lognormal(mean=9.2, sigma=0.75, size=len(prov)), 1)  # provincial GDP (100 million CNY, right-skewed)
grad = np.exp(0.055 * (prov["lon"] - 100))  # smooth east-high/west-low gradient (to create spatial clustering)
prov["gdp_pc"] = (55000 * grad * np.exp(RNG.normal(0, 0.10, len(prov)))).round(0)  # GDP per capita = gradient x noise
prov["pop_wan"] = (prov["gdp_yi"] * 1e8 / (prov["gdp_pc"] * 1e4)).round(1)  # back out population from per capita, keeping units consistent
dump(prov, "l4_province_gdp.csv")  # write the province sample to disk

COUNTRIES = {  # ISO3 and coordinates for 24 economies
    "CHN": ("China", 35.0, 103.0), "USA": ("United States", 39.0, -98.0), "JPN": ("Japan", 36.2, 138.2),  # major economies (ISO3 -> name, latitude, longitude)
    "KOR": ("South Korea", 36.5, 127.9), "DEU": ("Germany", 51.2, 10.4), "FRA": ("France", 46.6, 2.4),  # major European economies
    "GBR": ("United Kingdom", 54.0, -2.0), "ITA": ("Italy", 42.8, 12.8), "ESP": ("Spain", 40.2, -3.7),  # Southern European economies
    "NLD": ("Netherlands", 52.2, 5.3), "RUS": ("Russia", 60.0, 90.0), "BRA": ("Brazil", -10.0, -52.0),  # Western Europe and emerging economies
    "IND": ("India", 22.0, 79.0), "IDN": ("Indonesia", -2.0, 118.0), "VNM": ("Vietnam", 14.1, 108.3),  # emerging economies of Asia
    "THA": ("Thailand", 15.9, 101.0), "MYS": ("Malaysia", 4.2, 101.9), "SGP": ("Singapore", 1.35, 103.8),  # Southeast Asian economies
    "AUS": ("Australia", -25.0, 134.0), "CAN": ("Canada", 56.0, -106.0), "MEX": ("Mexico", 23.6, -102.5),  # Oceanian and North American economies
    "ZAF": ("South Africa", -29.0, 24.0), "SAU": ("Saudi Arabia", 24.0, 45.0), "TUR": ("Turkey", 39.0, 35.0),  # African and Middle Eastern economies
}  # end of the country dictionary
cty = pd.DataFrame([{"iso3": k, "country": v[0], "lat": v[1], "lon": v[2]}  # expand the country table
                    for k, v in COUNTRIES.items()])  # build by iterating over the dictionary
cty["gdp_pc"] = np.round(RNG.lognormal(mean=9.6, sigma=0.85, size=len(cty)), 0)  # GDP per capita (USD)
cty["trade_share"] = np.round(RNG.uniform(0.5, 12.0, len(cty)), 2)  # trade as a share of GDP (%)
dump(cty, "l4_country_indicators.csv")  # write the country sample to disk

# try to download the China province-level boundary (public GeoJSON); fall back to a symbol map on failure
GEO = DAT / "china_provinces.geojson"  # local cache path
if not GEO.exists():  # only try once
    url = ("https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json")  # public province-level boundary
    try:  # should not abort the pipeline when the network is unavailable
        with urllib.request.urlopen(url, timeout=20) as resp:  # open the remote file
            GEO.write_bytes(resp.read())  # write to the local cache
        print("[geojson] downloaded:", GEO)  # report a successful download
    except Exception as err:  # catch network errors
        print("[geojson] download failed, will use a symbol map:", err)  # warn about the fallback
geojson = json.loads(GEO.read_text(encoding="utf-8")) if GEO.exists() else None  # read the cache (if present)
print("geojson features:", len(geojson["features"]) if geojson else 0)  # print the feature count

# ---------------------------------------------------------------- 2) static comparison: choropleth vs symbol map
fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))  # three-cell comparison
value = prov.set_index("province")["gdp_pc"]  # GDP per capita indexed by the province short name
vmin, vmax = float(value.min()), float(value.max())  # color-scale range
cmap = plt.get_cmap(SEQ)  # continuous palette
if geojson:  # boundary file present: draw a real choropleth
    patches, colors = [], []  # collect polygons and colors
    for feat in geojson["features"]:  # loop over each province-level feature
        name = feat["properties"].get("name", "")  # feature name (e.g. "广东省" = Guangdong Province)
        short = name.replace("省", "").replace("市", "").replace("自治区", "").replace("壮族", "")  # strip suffixes
        short = short.replace("回族", "").replace("维吾尔", "").replace("特别行政区", "")  # keep cleaning
        if short not in value.index:  # skip if the name is not in the data
            continue  # move to the next feature
        geom = feat["geometry"]  # geometry object
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]  # normalize to a list
        for poly in polys:  # each polygon (including inner rings)
            ring = np.array(poly[0])  # take the outer-ring coordinates
            if ring.shape[0] < 3:  # fewer than three points cannot form a polygon
                continue  # skip
            patches.append(Polygon(ring, closed=True))  # build the polygon
            val = float(value[short])  # GDP per capita of this province
            colors.append(cmap((val - vmin) / (vmax - vmin + 1e-9)))  # normalize then pick the color
    coll = PatchCollection(patches, facecolor=colors, edgecolor="white", lw=0.4)  # draw in batch
    axes[0].add_collection(coll)  # add to the axes
    axes[0].set_xlim(73, 136); axes[0].set_ylim(17, 54)  # China extent
    axes[0].set_aspect(1.2)  # approximately equal aspect (latitude compensation)
    axes[0].set_title("(1) Choropleth: GDP per capita (area fill)", fontsize=9.5)  # subplot title
else:  # no boundary file: fall back to a symbol map
    sc = axes[0].scatter(prov["lon"], prov["lat"], s=prov["gdp_pc"] / 900,  # bubble size encodes the value
                         c=prov["gdp_pc"], cmap=SEQ, alpha=0.85, edgecolors="white", lw=0.5)  # bubble color uses the same encoding
    axes[0].set_title("(1) boundary file unavailable: falling back to a symbol map")  # subplot title
    plt.colorbar(sc, ax=axes[0], fraction=0.046, label="GDP per capita (CNY)")  # colorbar
axes[0].grid(False)  # a map does not need a grid
axes[0].tick_params(labelsize=7.5)  # tick font size
sm = axes[1].scatter(prov["lon"], prov["lat"], s=prov["gdp_yi"] * 3.2,  # symbol map: size = total GDP
                     c=prov["gdp_pc"], cmap=SEQ, alpha=0.8, edgecolors="#1E2630", lw=0.4)  # color = per capita
axes[1].set_title("(2) Symbol map: size = total, color = ratio", fontsize=9.5)  # subplot title
axes[1].set_xlim(73, 136); axes[1].set_ylim(17, 54)  # same range as the China map for easy comparison
axes[1].set_aspect(1.2)  # equal aspect
axes[1].grid(False)  # a map does not need a grid
plt.colorbar(sm, ax=axes[1], fraction=0.046, label="GDP per capita (CNY)")  # colorbar
axes[2].axis("off")  # the third cell holds chart-selection advice
axes[2].text(0.0, 0.95,  # explanatory text
             "Which one and when?\n\n"  # map-selection notes (when to use a choropleth, when a symbol map)
             "· To express a ratio/intensity (per capita, share, density) -> choropleth\n"  # ratio/intensity metrics -> choropleth
             "   but large areas dominate (area = visual weight)\n\n"  # reminder: large areas carry more visual weight
             "· To express a total (GDP, population, trade value) -> symbol map\n"  # total metrics -> symbol map
             "   avoid letting a large province with a small population distort the conclusion\n\n"  # avoid area distorting the conclusion
             "· To express connections (trade, migration) -> flow map / network\n\n"  # connection metrics -> flow map/network
             "· Classification: equal interval vs quantile vs natural breaks\n"  # the classification method shifts the color distribution
             "   change the classification and the color distribution changes -- put it in the caption",  # the classification method must go in the caption
             fontsize=9.2, va="top", ha="left", linespacing=1.5)  # slightly larger line spacing
plt.tight_layout()  # adjust the three cells
save(fig, "fig_map_choropleth_vs_symbol.png")  # save: map comparison

# ---------------------------------------------------------------- 3) spatial autocorrelation: Moran's I and the Moran scatter
coords = prov[["lon", "lat"]].values  # province centroid coordinates
z = (value - value.mean()) / value.std(ddof=0)  # standardized GDP per capita
z = z.reindex(prov["province"]).values  # align to the table order
K = 4  # define spatial adjacency with 4 nearest neighbors
d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)  # squared pairwise distances
np.fill_diagonal(d2, np.inf)  # a point is not its own neighbor
W = np.zeros_like(d2)  # initialize the weight matrix
idx = np.argsort(d2, axis=1)[:, :K]  # take the K nearest neighbors of each province
for i, nbrs in enumerate(idx):  # fill the weights row by row
    W[i, nbrs] = 1.0  # neighbor weight is 1
W = W / W.sum(axis=1, keepdims=True)  # row-standardize (each row sums to 1)
lag = W @ z  # spatial lag: the weighted average of neighbors
moran_i = float((z * lag).sum() / (z * z).sum())  # Moran's I formula
print(f"Moran's I (K={K} nearest neighbors) = {moran_i:.3f}")  # print the spatial autocorrelation strength
quad = np.where((z > 0) & (lag > 0), "HH",  # High-High (clustered high values)
                np.where((z < 0) & (lag < 0), "LL",  # Low-Low (clustered low values)
                         np.where((z > 0) & (lag < 0), "HL", "LH")))  # High-Low, Low-High (outliers)
moran_tbl = pd.DataFrame({"province": prov["province"], "z": z.round(3),  # record the standardized values and the lag
                          "spatial_lag": lag.round(3), "quadrant": quad})  # record the quadrant
dump(moran_tbl, "l4_moran.csv")  # write the Moran results to disk
print("quadrant counts:", pd.Series(quad).value_counts().to_dict())  # print the quadrant counts

fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))  # Moran scatter + quadrant map
qcolor = {"HH": "#8A0C3C", "LL": "#2E6F8E", "HL": "#C9A227", "LH": "#66717D"}  # four-quadrant colors
for q, c in qcolor.items():  # draw points quadrant by quadrant
    m = quad == q  # mask for this quadrant
    axes[0].scatter(z[m], lag[m], s=34, color=c, alpha=0.85, label=f"{q} ({m.sum()})")  # scatter
axes[0].axhline(0, color="#1E2630", lw=1.0); axes[0].axvline(0, color="#1E2630", lw=1.0)  # quadrant dividers
b, a = np.polyfit(z, lag, 1)  # fitted slope (i.e. the direction of Moran's I)
xs = np.linspace(z.min(), z.max(), 20)  # fitted-line x coordinates
axes[0].plot(xs, a + b * xs, color="#711426", lw=1.8, ls="--")  # draw the fitted line
axes[0].set_title("Moran scatter: do neighbors move with the value itself")  # subplot title
axes[0].set_xlabel("standardized GDP per capita (z)")  # x-axis
axes[0].set_ylabel("spatial lag (neighbor mean)")  # y-axis
axes[0].legend(fontsize=8.4)  # legend (with per-quadrant counts)
axes[0].text(0.02, 0.96, f"Moran's I = {moran_i:.3f}",  # in-figure annotation of Moran's I
             transform=axes[0].transAxes, va="top", fontsize=10,  # position and font size
             color="#8A0C3C", weight="bold")  # emphasis color
for q, c in qcolor.items():  # right panel: quadrant map
    m = quad == q  # mask for this quadrant
    axes[1].scatter(prov["lon"][m], prov["lat"][m], s=95, color=c, alpha=0.9,  # color encodes the quadrant
                    edgecolors="white", lw=0.6, label=q)  # white edges are clearer
axes[1].set_xlim(73, 136); axes[1].set_ylim(17, 54)  # China extent
axes[1].set_aspect(1.2)  # equal aspect
axes[1].grid(False)  # a map does not need a grid
axes[1].set_title("Spatial clustering: HH and LL appear in patches")  # subplot title
axes[1].legend(fontsize=8.4, ncol=2)  # two-column legend
plt.tight_layout()  # adjust the two cells
save(fig, "fig_moran_scatter.png")  # save: spatial autocorrelation

# ---------------------------------------------------------------- 4) interactive maps: world choropleth + flow map
world = px.choropleth(cty, locations="iso3", color="gdp_pc",  # world GDP per capita choropleth
                      hover_name="country", locationmode="ISO-3",  # locate by ISO-3 code
                      color_continuous_scale="RdPu",  # the same continuous palette as the slides
                      title="World GDP per capita (choropleth, zoom and hover)")  # title
world.write_html(INT / "map_world_choropleth.html", include_plotlyjs="cdn")  # export the interactive map
print("[interactive]", INT / "map_world_choropleth.html")  # print the artifact path
hub = "CHN"  # illustrate trade flows originating from China
hub_lat = float(cty.loc[cty["iso3"] == hub, "lat"].iloc[0])  # origin latitude
hub_lon = float(cty.loc[cty["iso3"] == hub, "lon"].iloc[0])  # origin longitude
links = cty[cty["iso3"] != hub].copy()  # all other countries are destinations
links["value"] = (cty.loc[cty["iso3"] == hub, "gdp_pc"].iloc[0] / links["gdp_pc"]).round(2)  # simplified flow
fig_flow = go.Figure()  # assemble by hand with graph_objects
fig_flow.add_trace(go.Scattergeo(  # destinations: bubble size and color encode the trade share
    lat=links["lat"], lon=links["lon"], text=links["country"],  # coordinates and hover text
    marker=dict(size=links["trade_share"] * 2.2,  # bubble size (scale, not abused as radius)
                color=links["trade_share"], colorscale="RdPu",  # color encodes the same value
                showscale=True, colorbar=dict(title="trade share %")),  # colorbar
    mode="markers", name="trade partner"))  # legend name
for row in links.itertuples():  # draw each origin-to-destination line
    fig_flow.add_trace(go.Scattergeo(  # one trace per line
        lat=[hub_lat, row.lat], lon=[hub_lon, row.lon],  # the two endpoints
        mode="lines", line=dict(width=0.9, color="#8A0C3C"),  # thin red line
        opacity=0.45, showlegend=False))  # transparency and legend
fig_flow.update_layout(title_text="Trade links (bubble = trade share, line = flow direction)",  # title
                       geo=dict(showland=True, landcolor="#F7F3F2",  # land fill
                                showcountries=True, countrycolor="#D6CDD0",  # country borders
                                projection_type="natural earth"))  # projection choice (affects the shape)
fig_flow.write_html(INT / "map_trade_flows.html", include_plotlyjs="cdn")  # export the flow map
print("[interactive]", INT / "map_trade_flows.html")  # print the artifact path
print("provinces:", len(prov), "| countries:", len(cty), "| geojson:", bool(geojson))  # closing log
