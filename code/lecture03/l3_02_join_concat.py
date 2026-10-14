"""l3_02_join_concat.py -- stack two extracts vertically and align their schemas.

Vertical concatenation looks trivial until the two files disagree on column names, order, types or
units. The script stacks two quarterly order extracts, first shows what happens if nothing is done,
then gives the clean result after explicit alignment, with a source column and two assertions
attached.
"""
import numpy as np                                                 # random numbers for the two extracts
import pandas as pd                                                # concatenation, type handling and reconciliation
from _common_l3 import RNG, dump                                   # reuse the shared seed and the dump helper

N_Q1, N_Q2 = 240, 260                                              # rows in each extract

q1 = pd.DataFrame({                                                # extract one: a fairly tidy schema
    "order_id": [f"Q1-{1000 + i}" for i in range(N_Q1)],           # order id
    "date": pd.to_datetime("2026-01-01") + pd.to_timedelta(RNG.integers(0, 90, N_Q1), "D"),   # date column
    "store_id": RNG.choice([f"S{i:02d}" for i in range(1, 9)], N_Q1),   # store code
    "amount": np.round(RNG.lognormal(6.0, 0.6, N_Q1), 2),          # amount (yuan)
    "qty": RNG.integers(1, 9, N_Q1),                               # quantity (integer)
})                                                                 # end of extract one
q2 = pd.DataFrame({                                                # extract two: a slightly different schema
    "OrderID": [f"Q2-{2000 + i}" for i in range(N_Q2)],            # the key column has a different name
    "date": (pd.to_datetime("2026-04-01") + pd.to_timedelta(RNG.integers(0, 91, N_Q2), "D")).strftime("%d/%m/%Y"),   # day/month/year
    "store": RNG.choice([f"S{i:02d}" for i in range(1, 9)], N_Q2),   # the store column is named differently too
    "amount": [f"¥{v:,.2f}" for v in RNG.lognormal(6.1, 0.6, N_Q2)],   # the amount is a string with a currency symbol
    "quantity": RNG.integers(1, 9, N_Q2),                          # the quantity column has a different name
    "channel": RNG.choice(["online", "store"], N_Q2),              # a column only this extract has
})                                                                 # end of extract two
dump(q1, "l3_orders_2026Q1.csv")                                   # write extract one
dump(q2, "l3_orders_2026Q2.csv")                                   # write extract two

before = pd.DataFrame({"extract": ["q1", "q2", "q1+q2"],           # basic reconciliation table before any cleaning
                       "rows": [len(q1), len(q2), len(q1) + len(q2)],   # the row counts
                       "missing_cells": [int(q1.isna().sum().sum()),   # missing cells in extract one
                                         int(q2.isna().sum().sum()),   # missing cells in extract two
                                         0]})                      # the combined row is not a real extract
print("columns q1:", list(q1.columns))                             # print extract one columns
print("columns q2:", list(q2.columns))                             # print extract two columns
print("columns only in q1:", sorted(set(q1.columns) - set(q2.columns)))   # schema differences in one direction
print("columns only in q2:", sorted(set(q2.columns) - set(q1.columns)))   # schema differences in the other direction

# ---------------------------------------------------------------- 1) naive concat: where it goes wrong
naive = pd.concat([q1, q2], ignore_index=True)                     # just stack them vertically
print("naive concat shape:", naive.shape)                          # 500 x 9: renamed columns become empty
print("naive missing per column:\n", naive.isna().sum())           # every renamed column turns into a hole

# ---------------------------------------------------------------- 2) align the schema explicitly
RENAME = {"OrderID": "order_id", "store": "store_id", "quantity": "qty"}   # old column name -> standard column name
def align(df, source):                                             # bring one extract into the target schema
    """Rename, unify types and tag the source so both files share one schema."""   # function contract
    out = df.rename(columns=RENAME).copy()                         # rename first, then work on a copy
    out["date"] = pd.to_datetime(out["date"], format="mixed", dayfirst=True, errors="coerce")   # unify the dates
    out["amount"] = pd.to_numeric(                                 # the amounts still carry a currency symbol
        out["amount"].astype(str).str.replace(r"[^\d.\-]", "", regex=True),   # strip the symbols and spaces first
        errors="coerce")                                           # anything unparsable becomes NaN
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").astype("Int64")   # nullable integer type
    out["order_id"] = out["order_id"].astype("string")             # cast the key to a string type explicitly
    out["store_id"] = out["store_id"].astype("string")             # cast the store column to string as well
    out["channel"] = out.get("channel", pd.Series(pd.NA, index=out.index)).astype("string")   # fill in the missing column
    out["source"] = source                                         # source column: which file this row came from
    return out[["order_id", "date", "store_id", "amount", "qty", "channel", "source"]]   # unify the column order
combined = pd.concat([align(q1, "Q1"), align(q2, "Q2")], ignore_index=True)   # 500 x 7
print("combined shape:", combined.shape)                           # row count preserved, column count unified
print("rows per source:", combined["source"].value_counts().to_dict())   # reconcile: how many rows per file
assert len(combined) == len(q1) + len(q2), "concat lost or invented rows"   # assertion: rows are conserved
assert combined["order_id"].is_unique, "duplicate order ids after concat"   # assertion: the keys stay unique
dump(combined, "l3_orders_combined.csv")                           # write the aligned combined table
dump(before, "l3_concat_accounting.csv")                           # write the reconciliation table
print(combined.head(3).to_string())                                # print the first rows in the terminal for a quick check
