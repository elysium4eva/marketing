"""l3_05_cleaning_pipeline.py -- walk a dirty order table through the five-step cleaning routine.

Step one removes duplicates and irrelevant observations, step two repairs structural errors, step
three classifies outliers (rather than dropping them all), step four handles missing values and step
five runs validation and QA. Every step is written to an audit log so the cleaning stays traceable
and reproducible -- which is exactly why the slides insist on starting from a template.
"""
import numpy as np                                                  # quantiles and other numeric work
import pandas as pd                                                 # table operations needed for cleaning
import matplotlib.pyplot as plt                                     # before/after cleaning figure
from _common_l3 import DAT, save, dump                              # path constants plus the figure and dump helpers

raw = pd.read_csv(DAT / "l3_orders_dirty.csv", dtype=str)           # read the dirty sample produced by l3_01
raw["qty"] = pd.to_numeric(raw["qty"], errors="coerce")             # the quantity column must go back to numeric
log = []                                                            # one log line per cleaning action
def record(step, before, after, note):                              # write one step into the audit log
    """Append one cleaning record to the audit log."""              # function contract
    log.append({"step": step, "rows_before": before, "rows_after": after,   # row counts before and after
                "rows_removed": before - after, "note": note})      # how many rows were removed and why
    print(f"{step:38s} {before:5d} -> {after:5d}   {note}")         # also print this step in the terminal

# ---------------------------------------------------------------- step one: duplicates and irrelevant observations
n0 = len(raw)                                                       # baseline row count
df = raw.drop_duplicates().copy()                                   # drop fully duplicated rows
n1 = len(df)                                                        # row count after exact deduplication
record("Step 1a exact duplicate rows", n0, n1, "identical rows removed")   # log this step
df = df.drop_duplicates(subset="order_id", keep="first")            # keep one row per business key
n2 = len(df)                                                        # row count after key deduplication
record("Step 1b duplicate business keys", n1, n2, "same order_id kept once")   # log this step
irrelevant = df["order_id"].isna()                                  # irrelevant = not even a key
df = df.loc[~irrelevant].copy()                                     # drop rows that cannot be linked back
n3 = len(df)                                                        # row count after relevance filtering
record("Step 1c irrelevant observations", n2, n3, "rows without a key removed")   # log this step

# ---------------------------------------------------------------- step two: structural errors
city_map = {"shenzhen": "Shenzhen", "shēnzhèn": "Shenzhen",         # unify every spelling onto one canonical form
            "guangzhou city": "Guangzhou", "guang zhou": "Guangzhou"}   # the alias mapping table
def normalise_city(value):                                          # look up the alias table case-insensitively
    """Trim, collapse whitespace and map aliases onto canonical city names."""   # function contract
    key = " ".join(str(value).strip().lower().split())              # normalise the form first
    return city_map.get(key, key.title())                           # replace on a hit, otherwise title-case it
df["city"] = df["city_raw"].map(normalise_city)                     # apply the mapping row by row
pay_map = {"paid": "paid", "unpaid": "unpaid", "n/a": "not_applicable",   # different spellings of the same status
           "-": "not_applicable", "not applicable": "not_applicable"}   # map them all onto one value
df["payment"] = (df["payment_raw"].str.strip().str.lower()          # trim and lower-case first
                 .map(pay_map).fillna("unknown"))                   # then map; anything unrecognised is explicitly unknown
df["date"] = pd.to_datetime(df["date_raw"], format="mixed", dayfirst=True, errors="coerce")   # step two: parse the dates
bad_dates = int(df["date"].isna().sum())                            # count the dates that fail to parse
record("Step 2 structural errors", n3, len(df), f"cities/payments normalised, {bad_dates} bad dates")   # write the log entry
print("city variants:", raw["city_raw"].nunique(), "->", df["city"].nunique())   # evidence of consistency
print("payment variants:", raw["payment_raw"].nunique(), "->", df["payment"].nunique())   # evidence of unit consistency

# ---------------------------------------------------------------- step three: outliers (classify first, treat later)
amount = pd.to_numeric(df["amount_raw"].astype(str)                 # the amount strings still carry symbols
                       .str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")   # strip the symbols, then convert to numeric
df["amount"] = amount                                               # attach the parsed result back to the table
impossible = (df["amount"] <= 0) | (df["amount"] > 100000)          # rule-based check: physically impossible
q1, q3 = df["amount"].quantile(.25), df["amount"].quantile(.75)     # compute the quartiles
iqr = q3 - q1                                                       # interquartile range
statistical = (df["amount"] < q1 - 1.5 * iqr) | (df["amount"] > q3 + 1.5 * iqr)   # statistically far out
df["flag_impossible"] = impossible                                  # flag only, never delete silently
df["flag_outlier"] = statistical & ~impossible                      # statistically extreme but possibly real in business terms
record("Step 3 outlier classification", len(df), len(df),           # this step removes no rows
       f"{int(impossible.sum())} impossible, {int(statistical.sum())} statistical outliers")   # log the counts

# ---------------------------------------------------------------- step four: handle missing values
miss_before = df[["qty", "amount", "date"]].isna().mean().mul(100).round(2)   # missing rate before treatment
median_qty = df["qty"].median()                                     # the median is robust to skew and outliers
df["qty_missing"] = df["qty"].isna().astype(int)                    # missing-indicator column: missingness is information in itself
df["qty"] = df["qty"].fillna(median_qty)                            # fill with the median and record the reason
miss_after = df[["qty", "amount", "date"]].isna().mean().mul(100).round(2)   # missing rate after treatment
record("Step 4 missing handling", len(df), len(df),                 # this step does not remove rows either
       f"qty filled with median {median_qty:.1f}, indicator column added")   # log the treatment
print("missing rate before:\n", miss_before.to_string())            # print the missing rate before
print("missing rate after:\n", miss_after.to_string())              # print the missing rate after

# ---------------------------------------------------------------- step five: validation and QA
assert df["order_id"].is_unique, "order_id must be unique after cleaning"   # assertion: keys are unique
assert df["date"].notna().all(), "unparsable dates must be resolved"   # assertion: dates are valid
assert df["amount"].notna().all(), "amount must be numeric after cleaning"   # assertion: amounts are numeric
assert set(df["payment"].unique()) <= {"paid", "unpaid", "not_applicable", "unknown"}, "unexpected payment"   # allowed value domain
assert df["city"].nunique() <= 5, "city normalisation failed"       # assertion: normalisation worked
clean = df[["order_id", "date", "city", "amount", "qty", "qty_missing", "payment",   # unify the final column order
            "flag_impossible", "flag_outlier"]].copy()              # the column set the analysis layer can use
dump(clean, "l3_orders_clean.csv")                                  # write the cleaned table
dump(pd.DataFrame(log), "l3_cleaning_log.csv")                      # write the cleaning audit log

# ---------------------------------------------------------------- samples for the structural-errors slide
city_variants = (raw["city_raw"].value_counts().rename_axis("city_raw")   # count each raw spelling
                 .reset_index(name="rows"))                         # reshape into two columns
dump(city_variants, "l3_city_names_raw.csv")                        # write the raw variants
mapping = pd.DataFrame({"raw": list(city_map), "canonical": list(city_map.values())})   # the alias mapping table
dump(mapping, "l3_city_names_map.csv")                              # write the mapping table

# ---------------------------------------------------------------- figure A: before versus after
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # three panels: dropped rows / missing / distribution
steps = [r["step"].split()[1] for r in log[:3]] + ["Step 4"]        # short labels for the first three steps (for the figure)
axes[0].bar([r["step"].split()[1] for r in log[:3]],                # names of the first three actions
            [r["rows_removed"] for r in log[:3]], color="#8A0C3C", alpha=0.85)   # how many rows each one removed
axes[0].set_title("Step 1: rows removed")                           # subplot title
axes[0].tick_params(axis="x", rotation=20)                          # rotate the labels for readability
axes[0].set_ylabel("rows")                                          # meaning of the y axis
x = np.arange(len(miss_before))                                     # x coordinates for the grouped bars
axes[1].bar(x - 0.2, miss_before.values, width=0.4, label="before", color="#C9A227")   # missing rate before
axes[1].bar(x + 0.2, miss_after.values, width=0.4, label="after", color="#8A0C3C")   # missing rate after
axes[1].set_xticks(x, miss_before.index)                            # x axis: column names
axes[1].set_title("Step 4: missing rate (%)")                       # subplot title
axes[1].legend()                                                    # the legend explains the two colours
positive = df["amount"] > 0                                         # logs require positive amounts
vals = [np.log10(df.loc[df[k] & positive, "amount"]) for k in ["flag_outlier", "flag_impossible"]]   # the two flag types
axes[2].hist(np.log10(df.loc[positive, "amount"]), bins=30,         # distribution of all positive amounts (log scale)
             color="#66717D", alpha=0.6, label="all rows")          # grey background histogram
for v, lab, c in zip(vals, ["statistical", "impossible"], ["#C9A227", "#8A0C3C"]):   # overlay the two flag types
    axes[2].hist(v, bins=15, color=c, alpha=0.85, label=lab)        # one histogram each
axes[2].set_title("Step 3: amount distribution (log10, positive only)")   # subplot title
axes[2].legend()                                                    # legend
plt.tight_layout()                                                  # adjust the three panels
save(fig, "fig_cleaning_before_after.png")                          # save: figure for the cleaning slide

# ---------------------------------------------------------------- figure B: repairing structural errors
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # variants / mapping / result
v = city_variants.set_index("city_raw")["rows"].sort_values()       # how often each raw spelling occurs
axes[0].barh(v.index, v.values, color="#8A0C3C", alpha=0.85)        # how messy the raw column is
axes[0].set_title(f"raw spellings ({len(v)} variants)")             # subplot title
axes[0].tick_params(axis="y", labelsize=8)                          # shrink the font for long labels
axes[1].axis("off")                                                 # show the mapping as a table, no axes
tbl = axes[1].table(cellText=mapping.values, colLabels=["raw", "canonical"],   # the alias mapping table
                    loc="center", cellLoc="center")                 # centre it
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.3)   # compact but readable
axes[1].set_title("alias mapping applied")                          # subplot title
final_counts = df["city"].value_counts()                            # counts of the canonical categories after cleaning
axes[2].bar(final_counts.index, final_counts.values, color="#C9A227", alpha=0.9)   # the normalised category column
axes[2].set_title(f"after cleaning ({df['city'].nunique()} cities)")   # subplot title
axes[2].tick_params(axis="x", rotation=20)                          # rotate the labels
plt.tight_layout()                                                  # adjust the three panels
save(fig, "fig_structural_errors.png")                              # save: figure for the structural-errors slide
print("clean shape:", clean.shape, "| audit log rows:", len(log))   # closing log line
