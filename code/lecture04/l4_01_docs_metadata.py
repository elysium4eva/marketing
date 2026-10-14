"""l4_01_docs_metadata.py -- build a survey sample and score documentation completeness against the original 20-item checklist.

The original course's stance: start documenting **before data collection**, and at least
keep the documentation together with the data. This script first writes this survey data
to disk, then auto-generates a field dictionary (codebook), and finally scores it item by
item against the 20 documentation elements, pointing out what is still missing.
"""
import numpy as np  # random sampling
import pandas as pd  # table handling
import matplotlib.pyplot as plt  # documentation completeness figure
from _common_l4 import RNG, DAT, save, dump, style_axes, annotated  # reuse the shared and caption helpers

N = 500  # survey sample size
CITIES = ["Shenzhen", "Shanghai", "Beijing", "Guangzhou", "Chengdu", "Hangzhou",  # the cities, weights sum to 1
          "Wuhan", "Xi'an"]  # second line of the city names
CITY_P = [0.26, 0.2, 0.18, 0.12, 0.08, 0.08, 0.05, 0.03]  # sampling share of each city
CHANNELS = ["Online", "Store", "Phone"]  # reach channels
EDU = ["High school or below", "Bachelor", "Master or above"]  # education tiers

survey = pd.DataFrame({  # assemble the survey sample
    "respondent_id": [f"R{i:04d}" for i in range(1, N + 1)],  # respondent id (primary key)
    "city": RNG.choice(CITIES, N, p=CITY_P),  # city (categorical)
    "channel": RNG.choice(CHANNELS, N, p=[0.55, 0.3, 0.15]),  # reach channel (categorical)
    "education": RNG.choice(EDU, N, p=[0.32, 0.55, 0.13]),  # education (ordinal)
    "age": RNG.integers(18, 66, N),  # age (numeric)
    "income_k": np.round(RNG.lognormal(2.55, 0.5, N), 1),  # monthly income (thousand CNY, right-skewed)
    "satisfaction": RNG.choice([1, 2, 3, 4, 5], N, p=[0.05, 0.1, 0.25, 0.4, 0.2]),  # satisfaction 1-5
    "spend_k": np.round(RNG.lognormal(1.6, 0.7, N), 2),  # monthly spend (thousand CNY, right-skewed)
    "revisit": RNG.random(N) < 0.42,  # whether they will repurchase (boolean)
})  # end of the survey sample definition
survey.loc[RNG.choice(N, 12, replace=False), "income_k"] = np.nan  # income is sensitive: inject 12 missing values
survey.loc[RNG.choice(N, 5, replace=False), "satisfaction"] = np.nan  # satisfaction also has a few missing values
dump(survey, "l4_survey_raw.csv")  # write the sample to disk

# ---------------------------------------------------------------- auto-generated field dictionary (codebook)
def field_dictionary(df, units):  # build one dictionary entry per column
    """Return the field dictionary: column name, type, non-null rate, unique count, range/sample, unit."""  # function contract
    rows = []  # one row per field
    for col in df.columns:  # loop over all columns
        s = df[col]  # take the column
        if pd.api.types.is_numeric_dtype(s):  # numeric: give the range and quantiles
            rng_txt = f"{s.min():.2f} ~ {s.max():.2f}"  # min to max
        elif pd.api.types.is_bool_dtype(s):  # boolean: give the share of True
            rng_txt = f"True share {s.mean():.1%}"  # share of True
        else:  # categorical: give the value count and examples
            rng_txt = f"{s.nunique()} categories, e.g. {', '.join(map(str, s.dropna().unique()[:3]))}"  # first three values
        rows.append({"field": col, "type": str(s.dtype),  # column name and dtype
                     "non_null_rate": f"{s.notna().mean():.1%}",  # non-null rate (completeness)
                     "unique_values": int(s.nunique()),  # number of unique values
                     "range_or_sample": rng_txt,  # value-domain description
                     "unit": units.get(col, "-")})  # unit (if known)
    return pd.DataFrame(rows)  # summarize into a table
UNITS = {"income_k": "thousand CNY/month", "spend_k": "thousand CNY/month", "age": "years",  # units of key fields
         "satisfaction": "1-5 scale", "revisit": "0/1"}  # notes for the rating and boolean fields
dictionary = field_dictionary(survey, UNITS)  # build the field dictionary
dump(dictionary, "l4_data_dictionary.csv")  # write it as CSV
md = ["# Data Dictionary (l4_survey_raw.csv)", "",  # also write Markdown for easy inclusion in reports
      "| field | type | non_null_rate | unique_values | range_or_sample | unit |",  # header
      "| --- | --- | --- | --- | --- | --- |"]  # separator row
for _, r in dictionary.iterrows():  # build the table row by row
    md.append(f"| {r['field']} | {r['type']} | {r['non_null_rate']} | {r['unique_values']} | {r['range_or_sample']} | {r['unit']} |")  # build one Markdown row
(DAT / "l4_data_dictionary.md").write_text("\n".join(md), encoding="utf-8")  # write the Markdown
print("[data]", DAT / "l4_data_dictionary.md")  # print the artifact path

# ---------------------------------------------------------------- documentation health check against the original 20-item list
DOC_ITEMS = {  # the 20 documentation elements -> this project's completion status
    "Title": "l4_survey_raw.csv consumer satisfaction and repurchase survey (Spring 2026)",  # title
    "Creator": "Big Data Analytics course team (synthetic data for teaching)",  # creator
    "Identifier": "SZU-BDA-2026-L4-SURVEY-01",  # internal id
    "Subject": "consumer satisfaction, repurchase intent, channel differences",  # keywords
    "Funders": "None (built by the course)",  # funding info
    "Rights": "for teaching use; contains no real personal information",  # intellectual property
    "Access": "course repository website/data/lecture04/",  # access path
    "Language": "English (field names)",  # content language
    "Dates": "collected 2026-03; published 2026-03-20",  # lifecycle dates
    "Location": "8 cities in China (see the city field)",  # spatial coverage
    "Methodology": "quota sampling, three channels (online/store/phone), 12-item questionnaire",  # generation method
    "Data processing": "missing values flagged; top 1% of income winsorized; no imputation",  # processing steps
    "Sources": "course synthetic sample (structure inspired by e-commerce consumer surveys)",  # data sources
    "File list": "l4_survey_raw.csv, l4_data_dictionary.csv",  # file list
    "File formats": "CSV (UTF-8-SIG)",  # file format
    "File structure": "one row = one respondent (long format)",  # data structure and granularity
    "Variable list": "see l4_data_dictionary.csv",  # variable list
    "Code lists": "satisfaction 1-5; channel={online,store,phone}",  # code list
    "Versions": "v1.0 (2026-03-20)",  # version and timestamp
    "Checksums": "SHA-256 in the release-package README",  # checksum
}  # end of the 20-item documentation dictionary
status = []  # per-item completion status
for item, value in DOC_ITEMS.items():  # loop over all 20 items
    filled = bool(str(value).strip()) and value != "-"  # non-empty counts as "filled"
    weak = isinstance(value, str) and ("TBD" in value or "None" == value.strip())  # an explicit "TBD/none" counts as weak
    status.append({"element": item, "content": value,  # record the element and content
                   "status": "filled" if (filled and not weak) else ("weak" if filled else "missing")})  # three-tier status
doc_status = pd.DataFrame(status)  # summarize into a table
dump(doc_status, "l4_docs_coverage.csv")  # write the check table to disk
counts = doc_status["status"].value_counts()  # count the three tiers
print("documentation coverage:", counts.to_dict())  # print the coverage overview

# ---------------------------------------------------------------- figure: documentation completeness dashboard
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), gridspec_kw={"width_ratios": [1, 1.5, 1.2]})  # three panels: status counts / field missing rate / dictionary size
labels = ["filled", "weak", "missing"]  # order of the three status tiers
vals = [int(counts.get(k, 0)) for k in labels]  # count per tier
colors = ["#2E6F8E", "#C9A227", "#8A0C3C"]  # blue/gold/red respectively
axes[0].bar(labels, vals, color=colors, alpha=0.9)  # panel 1: counts of the three tiers
style_axes(axes[0], "Documentation element completeness", "status", "number of elements")  # shared style
for i, v in enumerate(vals):  # annotate the count on top of each bar
    axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9.5)  # value label
miss_rate = (survey.isna().mean() * 100).sort_values(ascending=False)  # missing rate of each column
mr = miss_rate[miss_rate > 0]  # only plot columns that have missing values
axes[1].barh(mr.index[::-1], mr.values[::-1], color="#8A0C3C", alpha=0.85)  # panel 2: missing rate
style_axes(axes[1], "Field non-null rate (completeness)", "missing rate %", "field")  # shared style
annotated(axes[1], f"overall non-null rate {survey.notna().mean().mean():.1%}", "lower right")  # in-figure annotation
axes[2].barh(["total unique values", "number of fields"],  # panel 3: dictionary size
             [int(dictionary["unique_values"].sum()), len(dictionary)],  # show the magnitude with bars
             color=["#C9A227", "#66717D"], alpha=0.9)  # two colors
style_axes(axes[2], "Data dictionary size", "count", "")  # shared style
plt.tight_layout()  # adjust the three subplots
save(fig, "fig_docs_coverage.png")  # save: one-page figure for documentation and metadata
print("dictionary rows:", len(dictionary), "| survey shape:", survey.shape)  # closing log
