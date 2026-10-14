"""l4_13_highdim.py -- high-dimensional data visualization: dimension-reduction projections, parallel coordinates, clustered heatmaps, and interpretability visuals.

A 12-dimensional customer-behavior dataset (600 rows) answers three questions:
(1) How do you "see" high-dimensional data? -- a comparison of PCA / t-SNE / UMAP projections and their pitfalls;
(2) How do you compare multivariate profiles? -- parallel coordinates, radar charts, and a cluster-reordered heatmap;
(3) How do you show what a model bases its decisions on? -- permutation importance, partial dependence (PDP), and ICE curves.
"""
import numpy as np  # numeric computation
import pandas as pd  # data tables
import matplotlib.pyplot as plt  # static figures
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list  # hierarchical clustering and reordering
from sklearn.decomposition import PCA  # principal component analysis (linear reduction)
from sklearn.manifold import TSNE, trustworthiness  # t-SNE and neighborhood trustworthiness
from sklearn.ensemble import RandomForestClassifier  # the model used for interpretability visuals
from sklearn.inspection import permutation_importance  # permutation importance
from sklearn.preprocessing import StandardScaler  # standardization (required before reduction)
from _common_l4 import RNG, OKABE, INT, save, dump  # reuse the shared helpers and the interactive directory
try:  # UMAP is an optional dependency
    from umap import UMAP  # nonlinear reduction (local + global structure)
    HAS_UMAP = True  # mark as available
except Exception as exc:  # fall back when not installed
    print("[umap] not available, fall back to Isomap:", exc)  # report the fallback
    from sklearn.manifold import Isomap  # use Isomap as a same-family substitute
    HAS_UMAP = False  # mark as unavailable

# ---------------------------------------------------------------- 1) synthesize 12-dimensional customer-behavior data
N = 600  # number of customers
SEGMENTS = ["High-value loyal", "Price-sensitive", "New explorers"]  # three latent segments (to color the projections)
seg = RNG.choice(SEGMENTS, N, p=[0.3, 0.4, 0.3])  # randomly assign segments
FEATURES = {  # feature name -> (mean, std) baseline
    "recency_days": (60, 25),  # days since last purchase
    "frequency": (12, 5),  # purchase frequency
    "monetary": (2600, 900),  # cumulative amount
    "avg_order_value": (220, 60),  # order value
    "return_rate": (0.08, 0.04),  # return rate
    "discount_share": (0.24, 0.10),  # share of discounted orders
    "category_breadth": (4.5, 1.6),  # number of categories purchased
    "session_count": (28, 10),  # number of sessions
    "avg_session_min": (7.5, 2.5),  # average session length (minutes)
    "email_open_rate": (0.32, 0.12),  # email open rate
    "support_tickets": (1.2, 1.0),  # support tickets
    "tenure_months": (22, 9),  # months on service
}
rows = {}  # generate feature by feature
for name, (mu, sd) in FEATURES.items():  # loop over the features
    base = RNG.normal(mu, sd, N)  # baseline values
    if name in ("recency_days", "return_rate", "support_tickets"):  # these three: lower for the high-value segment
        base -= np.where(seg == "High-value loyal", sd * 0.9, 0.0)  # loyal customers are more active with fewer returns and tickets
        base += np.where(seg == "New explorers", sd * 0.7, 0.0)  # new customers are less active
    if name in ("frequency", "monetary", "avg_order_value", "tenure_months", "category_breadth"):
        base += np.where(seg == "High-value loyal", sd * 1.1, 0.0)  # the high-value segment is markedly higher
        base -= np.where(seg == "Price-sensitive", sd * 0.3, 0.0)  # the price-sensitive segment is slightly lower
    if name in ("discount_share",):
        base += np.where(seg == "Price-sensitive", sd * 1.4, 0.0)  # the price-sensitive segment relies more on discounts
    rows[name] = np.round(np.clip(base, 0, None), 3)  # drop negative values and keep three decimal places
customers = pd.DataFrame(rows)  # assemble the feature table
customers["segment"] = seg  # latent segment (for teaching color-coding only)
# create a "churn" label: nonlinearly determined by several features, for later importance / PDP / ICE
logit = (-0.035 * customers["recency_days"] + 0.09 * customers["support_tickets"]  # the longer since the last purchase and the more tickets, the more likely to churn
         - 0.10 * customers["frequency"] - 0.0004 * customers["monetary"]  # the more often they buy and the more they spend, the less likely to churn
         + 2.2 * customers["return_rate"] + 1.1 * customers["discount_share"]  # returns and discount reliance raise churn
         + 0.35 * np.sin(customers["avg_session_min"] / 3.0))  # nonlinear effect of session length
prob = 1 / (1 + np.exp(-(logit - logit.mean())))  # normalize into a probability
customers["churn"] = (RNG.random(N) < prob).astype(int)  # generate 0/1 labels by probability
dump(customers, "l4_highdim_customers.csv")  # write the sample to disk
X = customers.drop(columns=["segment", "churn"]).values  # feature matrix (12-dimensional)
X_std = StandardScaler().fit_transform(X)  # standardize: a required step before reduction
print("high-dim data:", X.shape, "| segments:", dict(pd.Series(seg).value_counts()),
      "| churn rate:", round(float(customers['churn'].mean()), 3))  # print a data overview

# ---------------------------------------------------------------- 2) three dimension-reduction projections
pca = PCA(n_components=6, random_state=42).fit(X_std)  # keep 6 principal components
Z_pca = pca.transform(X_std)[:, :3]  # take the first three components (two for 2D, three for the 3D interactive)
evr = pca.explained_variance_ratio_ * 100  # variance share explained by each component (%)
Z_tsne = TSNE(n_components=2, perplexity=30, init="pca",  # t-SNE: emphasizes local neighborhoods
              learning_rate="auto", random_state=42).fit_transform(X_std)  # a fixed seed ensures reproducibility
if HAS_UMAP:  # use UMAP when available
    Z_umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2,  # neighbor count and minimum distance are the two key parameters
                  random_state=42).fit_transform(X_std)  # balances local and global structure
    umap_label = "UMAP"  # legend name
else:  # fall back to Isomap
    Z_umap = Isomap(n_neighbors=15, n_components=2).fit_transform(X_std)  # isometric mapping (same-family method)
    umap_label = "Isomap (UMAP fallback)"  # label honestly
# neighborhood trustworthiness: are high-dimensional neighbors still neighbors in low dimensions (closer to 1 is better)
tw_pca = trustworthiness(X_std, Z_pca[:, :2], n_neighbors=10)  # PCA trustworthiness
tw_tsne = trustworthiness(X_std, Z_tsne, n_neighbors=10)  # t-SNE trustworthiness
tw_umap = trustworthiness(X_std, Z_umap, n_neighbors=10)  # UMAP trustworthiness
metrics = pd.DataFrame({  # summarize projection quality
    "method": ["PCA", "t-SNE", umap_label],  # three methods
    "trustworthiness(10-NN)": [round(tw_pca, 3), round(tw_tsne, 3), round(tw_umap, 3)],  # trustworthiness values
    "interpretability": ["variance is interpretable (first two components %.1f%% + %.1f%%)" % (evr[0], evr[1]),  # PCA has a variance basis
               "only local structure is readable", "local + global, parameter-sensitive"],  # qualitative notes on the other two
})
dump(metrics, "l4_highdim_projection_metrics.csv")  # write the projection metrics to disk
print(metrics.to_string(index=False))  # print to the terminal

# ---------------------------------------------------------------- 3) figure 1: comparison of the three projections
seg_colors = {"High-value loyal": OKABE[0], "Price-sensitive": OKABE[1], "New explorers": OKABE[2]}  # colorblind-friendly three colors
fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.5))  # three projections + one quality comparison
for name, c in seg_colors.items():  # draw points segment by segment
    m = seg == name  # mask for this segment
    axes[0].scatter(Z_pca[m, 0], Z_pca[m, 1], s=9, color=c, alpha=0.6, label=name)  # PCA projection
axes[0].set_xlabel(f"PC1 ({evr[0]:.1f}%)", fontsize=9)  # annotate the explained variance
axes[0].set_ylabel(f"PC2 ({evr[1]:.1f}%)", fontsize=9)  # annotate the explained variance
axes[0].set_title("(1) PCA: linear, variance is interpretable")  # subplot title
axes[0].legend(fontsize=7.4, markerscale=1.6)  # legend
for name, c in seg_colors.items():  # t-SNE projection
    m = seg == name  # mask for this segment
    axes[1].scatter(Z_tsne[m, 0], Z_tsne[m, 1], s=9, color=c, alpha=0.6)  # draw points
axes[1].set_title("(2) t-SNE: local neighborhoods clear, distances unreadable")  # subplot title
axes[1].set_xticks([]); axes[1].set_yticks([])  # reduced coordinates carry no units
for name, c in seg_colors.items():  # UMAP / Isomap projection
    m = seg == name  # mask for this segment
    axes[2].scatter(Z_umap[m, 0], Z_umap[m, 1], s=9, color=c, alpha=0.6)  # draw points
axes[2].set_title(f"(3) {umap_label}: balances local + global")  # subplot title
axes[2].set_xticks([]); axes[2].set_yticks([])  # reduced coordinates carry no units
axes[3].bar(range(1, 7), evr, color=OKABE[3], alpha=0.9)  # scree plot: variance explained by each component
axes[3].plot(range(1, 7), np.cumsum(evr), "-o", ms=4, color=OKABE[5], label="cumulative")  # cumulative explained variance
axes[3].set_title("(4) Scree plot and neighborhood trustworthiness")  # subplot title
axes[3].set_xlabel("principal-component index", fontsize=9); axes[3].set_ylabel("explained variance %", fontsize=9)  # axis labels
axes[3].legend(fontsize=7.6)  # legend
axes[3].text(0.03, 0.95, f"trustworthiness: PCA {tw_pca:.2f} | t-SNE {tw_tsne:.2f} | {umap_label} {tw_umap:.2f}",
             transform=axes[3].transAxes, va="top", fontsize=7.4, color="#1E2630")  # in-figure comparison
plt.tight_layout()  # adjust the four cells
save(fig, "fig_highdim_projections.png")  # save: projection comparison figure

# ---------------------------------------------------------------- 4) figure 2: parallel coordinates, clustered heatmap, and radar chart
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6), gridspec_kw={"width_ratios": [1.5, 1.1, 1.0]})
show_cols = list(FEATURES.keys())[:8]  # parallel coordinates show the first eight features
Xs = pd.DataFrame(X_std, columns=list(FEATURES.keys()))[show_cols]  # standardized subset (for parallel coordinates)
Xall = pd.DataFrame(X_std, columns=list(FEATURES.keys()))  # all 12 standardized dimensions (the radar chart needs the later features)
for name, c in seg_colors.items():  # draw a profile line per segment
    m = (seg == name)  # segment mask
    axes[0].plot(range(len(show_cols)), Xs[m].mean().values, "-o", ms=4, lw=1.8, color=c, label=name)  # segment-mean profile
axes[0].set_xticks(range(len(show_cols)))  # one tick per feature
axes[0].set_xticklabels(show_cols, rotation=35, ha="right", fontsize=7.2)  # feature names (rotated to avoid overlap)
axes[0].set_ylabel("standardized value (z)", fontsize=9)  # y-axis: standardized, hence dimensionless
axes[0].set_title("(1) Parallel coordinates: multivariate profiles at a glance")  # subplot title
axes[0].legend(fontsize=7.6)  # legend
axes[0].axhline(0, color="#66717D", lw=0.8, ls="--")  # mean reference line
corr = Xs.corr().values  # feature correlation matrix
Z = linkage(Xs.T.values, method="average")  # hierarchically cluster the variables
order = leaves_list(Z)  # variable order after clustering (for reordering the heatmap)
im = axes[1].imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)  # the reordered correlation heatmap
axes[1].set_xticks(range(len(order)), [show_cols[i] for i in order], rotation=45, ha="right", fontsize=6.6)  # x-axis labels
axes[1].set_yticks(range(len(order)), [show_cols[i] for i in order], fontsize=6.6)  # y-axis labels
axes[1].grid(False)  # a heatmap does not need a grid
axes[1].set_title("(2) Cluster-reordered heatmap: blocks = groups")  # subplot title
plt.colorbar(im, ax=axes[1], fraction=0.046)  # colorbar
radar_cols = ["frequency", "monetary", "avg_order_value", "return_rate", "discount_share", "tenure_months"]
angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False)  # radar-chart angles
angles_closed = np.concatenate([angles, angles[:1]])  # closed angles
ax_r = plt.subplot(1, 3, 3, polar=True)  # turn the third panel into polar coordinates
for name, c in seg_colors.items():  # draw a radar per segment
    vals = Xall.assign(seg=seg).groupby("seg")[radar_cols].mean().loc[name].values  # segment means
    vals = np.concatenate([vals, vals[:1]])  # close the data
    ax_r.plot(angles_closed, vals, color=c, lw=1.8, label=name)  # draw the line
    ax_r.fill(angles_closed, vals, color=c, alpha=0.12)  # semi-transparent fill
ax_r.set_xticks(angles, [c.replace("_", "\n") for c in radar_cols], fontsize=6.4)  # axis labels
ax_r.set_title("(3) Radar chart: use sparingly (area exaggerates differences)", fontsize=10.5, pad=14)  # subplot title
ax_r.tick_params(labelsize=6.4)  # tick font size
plt.tight_layout()  # adjust the three cells
save(fig, "fig_parallel_cluster.png")  # save: multivariate profile figure

# ---------------------------------------------------------------- 5) figure 3: permutation importance, PDP, and ICE
clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=4,  # a random forest as an interpretable model
                             random_state=42).fit(X_std, customers["churn"])  # train on the churn label
perm = permutation_importance(clf, X_std, customers["churn"],  # permutation importance: shuffle a feature and see how much performance drops
                              n_repeats=10, random_state=42, scoring="roc_auc")  # ten repeats give the error
imp = pd.DataFrame({"feature": list(FEATURES.keys()),  # feature name
                    "importance": perm.importances_mean.round(4),  # mean importance
                    "std": perm.importances_std.round(4)})  # importance standard deviation
imp = imp.sort_values("importance", ascending=False)  # sort from high to low
dump(imp, "l4_highdim_importance.csv")  # write the importance table to disk
print("top-3 importance:\n", imp.head(3).to_string(index=False))  # print the top three
top = imp["feature"].head(3).tolist()  # take the top three important features
axes[0].barh(imp["feature"][::-1][:8], imp["importance"][::-1][:8],  # draw horizontal bars for the top eight
             xerr=imp["std"][::-1][:8], capsize=3, color=OKABE[5], alpha=0.9)  # with error bars
axes[0].set_xlabel("drop in ROC-AUC", fontsize=9)  # axis label
axes[0].set_title("(1) Permutation importance (with error from 10 repeats)")  # subplot title
axes[0].tick_params(axis="y", labelsize=7.4)  # feature-name font size
grid_vals = np.linspace(0, 1, 60)  # quantile grid (0-1)
pdp_curves = {}  # record the PDP curves
for feat in top[:2]:  # draw one PDP for each of the top two features
    j = list(FEATURES.keys()).index(feat)  # column index of this feature in the matrix
    qs = np.quantile(X_std[:, j], grid_vals)  # take the feature's actual quantiles
    preds = []  # mean prediction at each quantile
    for v in qs:  # loop over the quantiles
        X_tmp = X_std.copy()  # copy the feature matrix
        X_tmp[:, j] = v  # replace that column entirely with v
        preds.append(clf.predict_proba(X_tmp)[:, 1].mean())  # mean churn probability
    pdp_curves[feat] = (qs, np.array(preds))  # record the curve
for (feat, (xs, ys)), c in zip(pdp_curves.items(), [OKABE[0], OKABE[1]]):  # draw the PDP curves
    axes[1].plot(xs, ys, lw=2.0, color=c, label=feat)  # curve
axes[1].set_xlabel("standardized feature value (z)", fontsize=9)  # axis label
axes[1].set_ylabel("mean predicted churn probability", fontsize=9)  # axis label
axes[1].set_title("(2) PDP: how features affect predictions on average")  # subplot title
axes[1].legend(fontsize=7.6)  # legend
ice_feat = top[0]  # ICE uses the most important feature
j = list(FEATURES.keys()).index(ice_feat)  # column index of this feature
qs = np.quantile(X_std[:, j], grid_vals)  # quantile grid
idx_sample = RNG.choice(len(X_std), 60, replace=False)  # draw 60 individuals for the ICE curves
ice = np.zeros((len(idx_sample), len(qs)))  # store the ICE curves
for r, i in enumerate(idx_sample):  # per individual
    for k, v in enumerate(qs):  # per quantile
        X_tmp = X_std[i:i + 1].copy()  # take only this individual's row
        X_tmp[0, j] = v  # change the feature value
        ice[r, k] = clf.predict_proba(X_tmp)[0, 1]  # the individual's predicted probability
    axes[2].plot(qs, ice[r], color="#66717D", alpha=0.25, lw=0.9)  # gray line: individual curve
axes[2].plot(qs, ice.mean(axis=0), color=OKABE[5], lw=2.2, label="mean (= PDP)")  # the mean line is the PDP
axes[2].set_xlabel(f"{ice_feat} (standardized)", fontsize=9)  # axis label
axes[2].set_ylabel("predicted churn probability", fontsize=9)  # axis label
axes[2].set_title("(3) ICE: individual curves reveal heterogeneity and interactions")  # subplot title
axes[2].legend(fontsize=7.6)  # legend
plt.tight_layout()  # adjust the three cells
save(fig, "fig_interpretability.png")  # save: interpretability visualization

# ---------------------------------------------------------------- 6) interactive 3D embedding (plotly)
import plotly.graph_objects as go  # use the low-level interface for 3D scatter
fig3d = go.Figure()  # new canvas
for name, c in seg_colors.items():  # add a layer per segment
    m = seg == name  # segment mask
    fig3d.add_trace(go.Scatter3d(x=Z_pca[m, 0], y=Z_pca[m, 1], z=Z_pca[m, 2],  # the first three components
                                 mode="markers", name=name,  # legend name
                                 marker=dict(size=3, color=c, opacity=0.6)))  # point style
fig3d.update_layout(title="PCA 3D embedding (rotate and zoom: interactive exploration of high-dimensional structure)",  # title
                    scene=dict(xaxis_title=f"PC1 {evr[0]:.0f}%",  # 3D axis labels
                               yaxis_title=f"PC2 {evr[1]:.0f}%",  # second axis
                               zaxis_title=f"PC3 {evr[2]:.0f}%"))  # third axis
fig3d.write_html(INT / "embedding_3d.html", include_plotlyjs="cdn")  # export the rotatable 3D embedding
print("[interactive]", INT / "embedding_3d.html")  # print the artifact path
print("high-dim figures done:", ["fig_highdim_projections.png", "fig_parallel_cluster.png", "fig_interpretability.png"])
