"""03_datatype_text.py -- text data: sample corpus + term-frequency/length visualization."""  # module comment
import re                                                             # regular expressions, used for tokenization
from collections import Counter                                       # counter, used to count term frequencies
import pandas as pd                                                   # tabular handling
import matplotlib.pyplot as plt                                       # plotting
from _common import save, dump_csv                                    # shared helpers

# ---------------------------------------------------------------- 1) sample corpus (financial news headline style)
docs = [                                                              # define a sample corpus of 20 short texts
    "CATL overseas revenue up 32% YoY in 2025",                        # text sample 1
    "Kweichow Moutai posts Q3 2025 results, gross margin stays stable",  # text sample 2
    "Central bank cuts reserve requirement ratio by 0.5 percentage point",  # text sample 3
    "BYD new-energy vehicle exports hit a record high",               # text sample 4
    "CSI 300 index closes up 1.2%, turnover tops one trillion",       # text sample 5
    "Cross-border e-commerce exports grow double-digit for three years",  # text sample 6
    "Semiconductor equipment imports down 8% MoM",                     # text sample 7
    "Solar module prices bottom out and rebound, utilization rises",  # text sample 8
    "Consumer confidence index stays above boom-or-bust line for two months",  # text sample 9
    "Steel futures main contract daily range widens",                 # text sample 10
    "Port container throughput up 6.4% YoY",                           # text sample 11
    "AI computing demand drives server shipment growth",              # text sample 12
    "Yuan exchange rate moves both ways, cross-border settlement share rises",  # text sample 13
    "Property sales area decline narrows as policy effects appear",   # text sample 14
    "Air passenger volume recovers to pre-pandemic level",            # text sample 15
    "Energy storage capacity expands fast, lithium prices swing",     # text sample 16
    "Biopharma financing warms up, innovative drug exports accelerate",  # text sample 17
    "Industrial robot output keeps double-digit growth",              # text sample 18
    "Crude oil prices firm up on geopolitical factors",               # text sample 19
    "Data element market builds faster, trading scale expands",       # text sample 20
]                                                                     # end of the corpus list
df = pd.DataFrame({"doc_id": range(1, len(docs) + 1),                 # text id column
                   "text": docs,                                       # text content column
                   "length": [len(d) for d in docs]})                  # character-length column (a numeric feature of text)
dump_csv(df, "text_samples.csv")                                      # dump the text samples

# ---------------------------------------------------------------- 2) featurization: tokenization + term frequency
def tokens(s):                                                        # define the tokenizer: CJK by character, letters/digits by word
    """Split a sentence into a token list: runs of letters/digits stay whole, CJK characters split one by one."""  # function description
    return re.findall(r"[A-Za-z0-9%\.]+|[\u4e00-\u9fff]", s)          # regex: an alphanumeric run or a single CJK character
all_tokens = [t for d in docs for t in tokens(d)]                     # collect the tokens of all documents into one list
freq = Counter(all_tokens).most_common(15)                            # take the 15 most frequent tokens
words, counts = [w for w, _ in freq], [c for _, c in freq]            # split into two lists: words and counts

# ---------------------------------------------------------------- 3) visualization
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                    # one row, three columns: term frequency / doc length / length distribution
axes[0].barh(range(len(words))[::-1], counts, color="#8A0C3C", alpha=0.85)  # left: horizontal bar chart of the top-15 frequencies
axes[0].set_yticks(range(len(words))[::-1], words, fontsize=8)         # set the y ticks to the word list
axes[0].set_title("Text -> term frequency (top 15 tokens)")            # left title
axes[0].set_xlabel("Occurrences")                                      # left x-axis meaning

axes[1].bar(df["doc_id"], df["length"], color="#66717D", alpha=0.9)    # middle: character count of each text
axes[1].set_title("Text length (a numeric feature)")                   # middle title
axes[1].set_xlabel("Document id")                                      # middle x-axis meaning
axes[1].set_ylabel("Character count")                                  # middle y-axis meaning

axes[2].hist(df["length"], bins=8, color="#C9A227", alpha=0.9)         # right: histogram of the text length distribution
axes[2].set_title("Length distribution (for truncation/padding)")      # right title
axes[2].set_xlabel("Character count")                                  # right x-axis meaning
plt.tight_layout()                                                     # adjust the layout
save(fig, "fig_text.png")                                              # save the figure
