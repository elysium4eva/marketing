"""l3_01_quality_profile.py -- build a "dirty" dataset and score its five quality dimensions.

The script deliberately plants the defects most often seen in an order table (mixed date formats,
duplicate primary keys, inconsistent category spellings, currency symbols mixed into the amounts,
missing values, impossible magnitudes), then quantifies each of the five quality dimensions used
in the slides (validity, accuracy, completeness, consistency, unit consistency).
"""
import numpy as np                                                 # random sampling and numeric work
import pandas as pd                                                # build tables and compute quality statistics
import matplotlib.pyplot as plt                                    # draw the four-panel quality dashboard
from _common_l3 import RNG, save, dump                             # reuse the shared seed, figure and dump helpers

N = 600                                                            # number of synthetic order rows
CITIES = ["Shenzhen", "shenzhen", "SHENZHEN ", " Shenzhen",        # the same city written five different ways
          "Guangzhou", "Guangzhou City", "Guang Zhou"]             # add more spellings to create inconsistency
order_id = [f"SO{2026000 + i}" for i in range(1, N + 1)]           # start from a clean order-id column
for i in RNG.choice(range(1, N), size=18, replace=False):          # pick 18 rows at random to copy the key from
    order_id[i] = order_id[i - 1]                                  # copy the previous order id: creates duplicate primary keys
dates = []                                                         # dates will arrive mixed across three formats
for i in range(N):                                                 # generate the raw date string row by row
    day = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(RNG.integers(0, 120)))   # draw a random day
    fmt = RNG.integers(0, 3)                                       # pick one source format at random
    dates.append(day.strftime("%Y-%m-%d") if fmt == 0 else         # format one: 2026-02-11
                 day.strftime("%d/%m/%Y") if fmt == 1 else         # format two: day/month/year
                 day.strftime("%Y%m%d"))                           # format three: compact form
city = [str(RNG.choice(CITIES)) for _ in range(N)]                 # the city column carries every spelling variant
amount_raw = []                                                    # amounts are always stored as raw strings
for i in range(N):                                                 # generate the raw amount string row by row
    value = float(RNG.lognormal(mean=6.2, sigma=0.7))              # log-normal spend: right-skewed
    style = RNG.integers(0, 4)                                     # four source conventions
    if style == 0:                                                 # convention one: currency symbol + thousands separator
        amount_raw.append(f"¥{value:,.2f}")                        # for example ¥1,234.56
    elif style == 1:                                               # convention two: currency-code suffix
        amount_raw.append(f"{value:.2f} RMB")                      # for example 1234.56 RMB
    elif style == 2:                                               # convention three: thousands as the unit (inconsistent units)
        amount_raw.append(f"{value / 1000:,.1f} thousand CNY")     # for example 1.2 thousand CNY
    else:                                                          # convention four: plain number with padding spaces
        amount_raw.append(f" {value:,.2f} ")                       # one leading and one trailing space
qty = RNG.integers(1, 12, N).astype(float)                         # make the quantity column float first
qty[RNG.choice(N, size=70, replace=False)] = np.nan                # blank the quantity on 70 random rows
amount_raw[3] = "-500.00"                                          # plant an impossible negative amount
amount_raw[7] = "0"                                                # plant an impossible zero amount
payment = RNG.choice(["paid", "Paid", "PAID", "unpaid", "N/A",     # payment status has synonymous spellings
                      "Not Applicable", "-"], size=N)              # "not applicable" is written three ways
raw = pd.DataFrame({"order_id": order_id,                          # order id (with duplicate primary keys)
                    "date_raw": dates,                             # date (mixed formats)
                    "city_raw": city,                              # city (inconsistent spellings)
                    "amount_raw": amount_raw,                      # amount (symbol and unit noise)
                    "qty": qty,                                    # quantity (with missing values)
                    "payment_raw": payment})                       # payment status (synonyms)
dump(raw, "l3_orders_dirty.csv")                                   # write to disk for the later scripts to read

# ---------------------------------------------------------------- quality profile: five dimensions quantified
parsed_date = pd.to_datetime(raw["date_raw"], format="mixed", dayfirst=True, errors="coerce")   # best-effort parsing
amount = (raw["amount_raw"].str.replace(r"[^\d.\-]", "", regex=True)   # strip the symbols first for a quick check
          .pipe(pd.to_numeric, errors="coerce"))                   # anything unparsable becomes NaN
dup_rows = int(raw.duplicated().sum())                             # fully duplicated rows (0 in this dataset)
dup_keys = int(raw["order_id"].duplicated().sum())                 # duplicate business keys (18 planted)
missing_rate = raw.isna().mean().mul(100).round(2)                 # completeness: missing rate per column
bad_amount = int(((amount <= 0) | (amount > 100000)).sum())        # accuracy: impossible magnitudes
city_raw_n = raw["city_raw"].nunique()                             # consistency: variants seen in the raw column
city_clean_n = raw["city_raw"].str.strip().str.title().nunique()   # consistency: variants after normalisation
pay_raw_n = raw["payment_raw"].nunique()                           # unit consistency: how many spellings per status
pay_clean_n = (raw["payment_raw"].str.strip().str.lower()          # normalise case and trim whitespace first
               .replace({"n/a": "not_applicable", "-": "not_applicable",   # fold the synonymous "not applicable" spellings together
                         "not applicable": "not_applicable"}).nunique())   # and converge onto a single value
valid_date_rate = 100 * parsed_date.notna().mean()                 # validity: share of dates that parse
valid_id_rate = 100 * raw["order_id"].str.match(r"^SO\d{7}$").mean()   # validity: share of order ids matching the pattern
score = {                                                          # quality scorecard: each dimension scores 0-100
    "Validity": round(0.5 * valid_date_rate + 0.5 * valid_id_rate, 1),   # dates and order ids count equally
    "Accuracy": round(100 - 100 * bad_amount / N, 1),              # share of plausible amounts
    "Completeness": round(100 - float(missing_rate.mean()), 1),    # share of populated cells
    "Consistency": round(100 * city_clean_n / max(city_raw_n, 1), 1),   # fewer variants after normalisation scores higher
    "Uniqueness": round(100 * (N - dup_keys) / N, 1),              # share of unique business keys
}                                                                  # end of the score dictionary
report = pd.DataFrame({"dimension": list(score), "score": list(score.values())})   # assemble into a table
dump(report, "l3_quality_scorecard.csv")                           # write the scorecard to disk
print(report.to_string(index=False))                               # and also print it in the terminal

# ---------------------------------------------------------------- figure: quality dashboard
fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.4))                # one row of four panels, sized for a 16:9 slide
mr = missing_rate.sort_values(ascending=False)                     # columns with the worst missing rate first
axes[0].bar(mr.index, mr.values, color="#8A0C3C", alpha=0.85)      # panel one: completeness
axes[0].set_title("Completeness:\nmissing rate (%)", fontsize=11)   # subplot title
axes[0].tick_params(axis="x", rotation=45, labelsize=8)            # rotate the x labels to avoid overlap
axes[1].bar(["full-row\nduplicates", "duplicate\nkeys"], [dup_rows, dup_keys],   # the two duplicate counts
            color=["#66717D", "#8A0C3C"])                          # panel two: uniqueness
axes[1].set_title("Uniqueness:\nduplicated records", fontsize=11)   # subplot title
axes[1].tick_params(axis="x", labelsize=8)                         # shrink the x tick label size
axes[2].bar(["raw", "normalised"], [city_raw_n, city_clean_n],     # variants before and after normalisation
            color=["#C9A227", "#711426"])                          # panel three: consistency
axes[2].set_title("Consistency:\ncity spellings", fontsize=11)     # subplot title
axes[2].tick_params(axis="x", labelsize=9)                         # x tick label size
axes[3].barh(list(score)[::-1], list(score.values())[::-1], color="#8A0C3C", alpha=0.85)   # panel four: the scorecard
axes[3].set_xlim(0, 100)                                           # scores on a 0-100 scale
axes[3].tick_params(axis="y", labelsize=8.5)                       # dimension label size
axes[3].set_title("Quality scorecard\n(0-100)", fontsize=11)       # subplot title
plt.tight_layout()                                                 # adjust the spacing between the four panels
save(fig, "fig_quality_profile.png")                               # save for the slides
print("payment spellings raw -> clean:", pay_raw_n, "->", pay_clean_n)   # log: evidence of unit consistency
print("valid dates %:", round(valid_date_rate, 1), "| implausible amounts:", bad_amount)   # log: validity
