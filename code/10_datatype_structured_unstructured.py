"""10_datatype_structured_unstructured.py -- structured / semi-structured / unstructured sample types and visualization."""  # module comment
import json                                                           # standard library: build semi-structured JSON data
import numpy as np                                                    # numerical computing
import pandas as pd                                                   # tabular handling (structured)
import matplotlib.pyplot as plt                                       # plotting
from _common import DAT, save, dump_csv                               # shared helpers

# ---------------------------------------------------------------- 1) structured: a 2D table of fixed-length fields
table = pd.DataFrame({                                                # structured data = rows x columns, columns have names and types
    "order_id": ["SO2026001", "SO2026002", "SO2026003", "SO2026004"],  # order id (string primary key)
    "date": ["2026-03-02", "2026-03-02", "2026-03-03", "2026-03-03"],  # order date (parseable as a date type)
    "amount": [1280.50, 640.00, 3980.75, 215.20],                      # order amount (float)
    "paid": [True, True, False, True],                                 # whether paid (boolean)
})                                                                    # end of the structured sample table
dump_csv(table, "structured_orders.csv")                              # dump the structured sample

# ---------------------------------------------------------------- 2) semi-structured: JSON that can nest and omit fields
records = [                                                           # semi-structured data = text that carries its own "schema"
    {"order_id": "SO2026001", "buyer": {"name": "Lin", "vip": True},   # record 1: contains a nested object
     "items": [{"sku": "A-1", "qty": 2}, {"sku": "B-7", "qty": 1}]},  # record 1: contains an array field
    {"order_id": "SO2026002", "buyer": {"name": "Chen"},                # record 2: missing the vip field (schema is not fixed)
     "items": [{"sku": "C-3", "qty": 5}]},                            # record 2: the array length differs too
    {"order_id": "SO2026003", "buyer": {"name": "Wang", "vip": False},  # record 3
     "items": [], "note": "invoice pending"},                               # record 3: has an extra note field
]                                                                     # end of the semi-structured record list
(DAT / "semistructured_orders.json").write_text(                      # write the semi-structured data into the repo
    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")  # don't escape non-ASCII + indent for readability
print("[data]", DAT / "semistructured_orders.json")                   # print the dump location

# ---------------------------------------------------------------- 3) unstructured: natural language / waveforms with no fixed fields
free_text = ("Revenue in the Southeast Asian market grew 30% YoY this quarter, mainly driven by cross-border e-commerce channels; "  # unstructured text, paragraph 1
             "management expects the gross margin to stay stable next quarter, but FX swings and rising ocean freight add cost pressure.")  # paragraph 2
(DAT / "unstructured_note.txt").write_text(free_text, encoding="utf-8")  # dump the unstructured text sample
wave_sample = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 400))          # another form of unstructured data: a raw waveform
np.save(DAT / "unstructured_wave.npy", wave_sample)                   # save as a binary array sample
print("[data]", DAT / "unstructured_note.txt")                        # print the dump location

# ---------------------------------------------------------------- 4) layout helper: manual character-count wrapping (more controllable than wrap)
def wrap_cjk(s, per_line=24):                                         # define the CJK line-wrap helper: characters per line
    """Wrap long text into multiple lines at a fixed character count, so auto layout does not blow up the canvas."""  # function description
    return "\n".join(s[i:i + per_line] for i in range(0, len(s), per_line))  # insert a newline every per_line characters

# ---------------------------------------------------------------- 5) visualization: the three forms side by side
fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.3))                   # one row, three columns: table / JSON / text
fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.04,     # set the margins manually, more controllable than tight_layout
                    wspace=0.12)                                      # horizontal spacing between subplots
for ax in axes:                                                       # handle all three subplots uniformly
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)                              # fix the coordinate range so the tight bbox is not stretched by text
    ax.set_xticks([]); ax.set_yticks([])                              # hide the ticks
    ax.grid(False)                                                    # turn off the grid

t = axes[0].table(cellText=table.astype(str).values,                  # left: render the DataFrame as a table
                  colLabels=list(table.columns),                      # column headers come from the column names
                  loc="center", cellLoc="center", bbox=[0.02, 0.42, 0.96, 0.5])  # fix the table position with a bbox
t.auto_set_font_size(False); t.set_fontsize(8)                        # disable auto font size, set it to 8pt manually
axes[0].set_title("Structured: a 2D table of fixed-length fields (CSV)", fontsize=12)  # left title

axes[1].text(0.02, 0.96, wrap_cjk(json.dumps(records[0],              # middle: show a JSON record (with nesting and arrays)
               ensure_ascii=False), 26)[:520],                        # after wrapping, keep only the first 520 characters
             family=["Consolas", "Microsoft YaHei"],                  # monospace font + CJK font per-glyph fallback
             fontsize=8.2, va="top", ha="left")                       # font size and top-left alignment
axes[1].set_title("Semi-structured: nested JSON (schema can change)", fontsize=12)  # middle title
axes[1].text(0.02, 0.18, "Fields may be omitted or added: parsing must tolerate this",  # add a key point at the bottom of the middle chart
             fontsize=8.5, color="#711426", va="bottom", ha="left")   # highlight in dark red

axes[2].text(0.02, 0.96, wrap_cjk(free_text, 22), fontsize=9.5,       # right: show field-free natural-language text
             va="top", ha="left")                                     # multi-line text after manual wrapping
axes[2].plot(np.linspace(0, 1, 400), 0.22 + 0.12 * wave_sample,       # draw a raw waveform in the lower half of the right chart
             color="#8A0C3C", lw=1.0)                                 # suggests that "unstructured" can also be a signal
axes[2].text(0.02, 0.06, "Text / audio / image: no fields to enumerate",  # add a key point at the bottom of the right chart
             fontsize=8.5, color="#711426", va="bottom", ha="left")   # highlight in dark red
axes[2].set_title("Unstructured: text / audio / image", fontsize=12)  # right title
save(fig, "fig_structured.png")                                       # save the figure (coordinate range is fixed, so no abnormal zoom)
