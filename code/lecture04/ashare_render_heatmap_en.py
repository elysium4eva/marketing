"""ashare_render_heatmap_en.py -- English twin of render_heatmap.py.

Draws the two static figures used on the A-share sample frames of Chapter 4:
    fig_ashare_heatmap.png       squarified treemap of 31 Shenwan level-1 industries
    fig_ashare_distribution.png  industry bars on the left, stock histogram on the right

The upstream repository renders an interactive Plotly page; the courseware needs
static figures, so this script re-uses the same data payload and lays it out with
a squarified treemap (Bruls, Huizing & van Wijk, 2000) in matplotlib.

Differences from the Chinese original: industry names, tickers, titles, axis
labels and the colour-bar legend are rendered in English, and tiles are labelled
with the ticker rather than the Chinese stock name. Every line keeps its comment.

Usage:
    python ashare_render_heatmap_en.py heatmap_industry.json [output-dir]
"""
import json                                                        # read the payload produced upstream
import os                                                          # join output paths
import sys                                                         # read the input path from argv
import matplotlib                                                  # choose a headless backend first
matplotlib.use("Agg")                                              # no display is available here
import matplotlib.pyplot as plt                                    # figure API
import matplotlib.patches as mpatches                              # rectangle patches
import matplotlib.colors as mcolors                                # colour maps and norms
import numpy as np                                                 # numeric helpers

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]                  # English-only deck: no CJK needed
plt.rcParams["axes.unicode_minus"] = False                         # draw minus signs normally

FIG_W, FIG_H = 13.2, 6.6                                           # canvas of the treemap figure
AX_W_IN, AX_H_IN = FIG_W * 0.86, FIG_H * 0.88                      # size of the treemap axes, inches

INDUSTRY_EN = {                                                    # Shenwan level-1 names, zh -> en
    "电子": "Electronics",
    "通信": "Telecommunications",
    "电力设备": "Power Equipment",
    "机械设备": "Machinery",
    "计算机": "Computers",
    "有色金属": "Non-ferrous Metals",
    "医药生物": "Pharma & Biotech",
    "传媒": "Media",
    "基础化工": "Basic Chemicals",
    "银行": "Banks",
    "汽车": "Automobiles",
    "国防军工": "Defence",
    "非银金融": "Non-bank Financials",
    "公用事业": "Utilities",
    "家用电器": "Home Appliances",
    "食品饮料": "Food & Beverage",
    "建筑材料": "Building Materials",
    "农林牧渔": "Agriculture",
    "建筑装饰": "Construction & Decoration",
    "石油石化": "Petroleum & Petrochemicals",
    "交通运输": "Transportation",
    "煤炭": "Coal",
    "房地产": "Real Estate",
    "商贸零售": "Commerce & Retail",
    "轻工制造": "Light Industry",
    "美容护理": "Beauty Care",
    "钢铁": "Steel",
    "环保": "Environmental Protection",
    "综合": "Conglomerates",
    "纺织服饰": "Textiles & Apparel",
    "社会服务": "Social Services",
}

SHORT_EN = {                                                       # compact names for small tiles
    "电子": "Electronics",
    "通信": "Telecoms",
    "电力设备": "Power Equip",
    "机械设备": "Machinery",
    "计算机": "Computers",
    "有色金属": "Metals",
    "医药生物": "Pharma",
    "传媒": "Media",
    "基础化工": "Chemicals",
    "银行": "Banks",
    "汽车": "Autos",
    "国防军工": "Defence",
    "非银金融": "Non-bank Fin",
    "公用事业": "Utilities",
    "家用电器": "Appliances",
    "食品饮料": "Food & Bev",
    "建筑材料": "Bldg Mats",
    "农林牧渔": "Agri",
    "建筑装饰": "Constr & Decor",
    "石油石化": "Petrochem",
    "交通运输": "Transport",
    "煤炭": "Coal",
    "房地产": "Real Estate",
    "商贸零售": "Retail",
    "轻工制造": "Light Ind",
    "美容护理": "Beauty",
    "钢铁": "Steel",
    "环保": "Environ Prot",
    "综合": "Congloms",
    "纺织服饰": "Textiles",
    "社会服务": "Soc Services",
}


def en(name):                                                      # industry name in English
    """Return the English industry name, falling back to the original."""
    return INDUSTRY_EN.get(name, name)                             # unmapped names pass through


def short(name):                                                   # compact English name
    """Return a shorter English name for tiles with little room."""
    return SHORT_EN.get(name, INDUSTRY_EN.get(name, name))         # fall back to the long name


def ticker(code):                                                  # exchange code -> readable ticker
    """Turn 'sh688825' into '688825.SH' for the tile labels."""
    if not code or len(code) < 3:                                  # nothing to reformat
        return code or ""                                          # keep whatever we were given
    prefix, digits = code[:2].lower(), code[2:]                     # split exchange and number
    return "%s.%s" % (digits, {"sh": "SH", "sz": "SZ", "bj": "BJ"}.get(prefix, prefix.upper()))   # join


def _ratio(row, length):                                           # worst aspect ratio of a row
    """Return the worst aspect ratio of a candidate row (lower is better)."""   # short docstring
    s = float(sum(row))                                            # total area of the row
    if s <= 0 or length <= 0:                                      # degenerate input
        return float("inf")                                        # never accept it
    return max(max(row) * length ** 2 / s ** 2, s ** 2 / (min(row) * length ** 2))   # the standard measure


def squarify(sizes, x, y, dx, dy):                                 # squarified treemap layout
    """Return rectangles [(x, y, w, h)] filling (x, y, dx, dy) proportionally to sizes."""   # short docstring
    if not sizes:                                                  # nothing to lay out
        return []                                                  # empty result
    total = float(sum(sizes))                                      # total area asked for
    if total <= 0 or dx <= 0 or dy <= 0:                           # degenerate input
        return [(x, y, dx, dy)] * len(sizes)                       # give everything the full box
    areas = [s * (dx * dy) / total for s in sizes]                  # scaled areas
    rects = []                                                     # the output rectangles
    while areas:                                                   # consume the list
        length = min(dx, dy)                                       # the shorter free side
        row = [areas[0]]                                           # start a new row
        rest = areas[1:]                                           # the remaining items
        while rest and _ratio(row + [rest[0]], length) <= _ratio(row, length):   # growing helps
            row.append(rest.pop(0))                                # keep growing the row
        area = sum(row)                                            # the row's total area
        if dx >= dy:                                               # lay the row out vertically
            w = area / dy if dy else 0                             # its width
            yy = y                                                 # cursor along y
            for r in row:                                          # one tile per item
                h = r / w if w else 0                              # its height
                rects.append((x, yy, w, h))                        # append the tile
                yy += h                                            # advance
            x += w                                                 # shrink the free box
            dx -= w                                                # horizontally
        else:                                                      # lay the row out horizontally
            h = area / dx if dx else 0                             # its height
            xx = x                                                 # cursor along x
            for r in row:                                          # one tile per item
                w = r / h if h else 0                              # its width
                rects.append((xx, y, w, h))                        # append the tile
                xx += w                                            # advance
            y += h                                                 # shrink the free box
            dy -= h                                                # vertically
        areas = rest                                               # continue with the rest
    return rects                                                   # all rectangles


def _fits(text, fs, bw):                                           # does one line fit the tile?
    """True when `text` at `fs` points is narrower than the tile width `bw` (axes units)."""
    return len(text) * 0.60 * fs / 72.0 / AX_W_IN <= bw * 0.93     # 0.60 em per character, bold sans


def _wrap(text, fs, bw):                                           # break a name over several lines
    """Wrap `text` so that each line fits the tile width at `fs` points."""
    maxc = max(4, int(bw * 0.93 * AX_W_IN * 72.0 / (0.60 * fs)))   # characters per line
    lines, cur = [], ""                                            # accumulator
    for word in text.split():                                      # greedy first-fit wrapping
        cand = (cur + " " + word).strip()                          # try adding the next word
        if cur and len(cand) > maxc:                               # it no longer fits
            lines.append(cur)                                      # close the line
            cur = word                                             # start a new one
        else:                                                      # it still fits
            cur = cand                                             # keep growing the line
    if cur:                                                        # last line
        lines.append(cur)                                          # add it
    return lines                                                   # wrapped text


def draw_board_label(ax, bx, by, bw, bh, name, gb):                # industry pill inside its tile
    """Draw a board label that fits the tile, shrinking and wrapping as needed."""
    value = "%+.1f%%" % (gb * 100)                                 # the return, always with a sign
    for txt, fs in ((en(name), 8.2), (short(name), 7.2),           # long name first, then short ones
                    (short(name), 6.2), (short(name), 5.4)):       # and finally very small type
        if _fits("%s %s" % (txt, value), fs, bw):                  # one line is enough
            ax.text(bx + 0.004, by + bh - 0.008,                   # top-left corner of the tile
                    "%s %s" % (txt, value), fontsize=fs, fontweight="bold",   # text
                    color="white", va="top", ha="left",            # anchored inside the tile
                    bbox=dict(boxstyle="round,pad=0.14", fc="#2B2B2B", ec="none", alpha=0.82))   # dark pill
            return                                                 # done
        lines = _wrap(txt, fs, bw) + [value]                       # try the value on its own line
        widest = max(len(ln) for ln in lines)                      # longest line after wrapping
        tall = len(lines) * fs * 1.36 / 72.0 / AX_H_IN             # block height in axes units
        if _fits("x" * widest, fs, bw) and tall <= bh * 0.92:      # the block fits the tile
            ax.text(bx + 0.004, by + bh - 0.008,                   # top-left corner of the tile
                    "\n".join(lines), fontsize=fs, fontweight="bold",   # wrapped label
                    color="white", va="top", ha="left", multialignment="left",   # left aligned block
                    linespacing=1.12,                              # tight leading
                    bbox=dict(boxstyle="round,pad=0.14", fc="#2B2B2B", ec="none", alpha=0.82))   # dark pill
            return                                                 # done
    # nothing fits: the tile stays unlabelled and the distribution panel names that industry


def main(path, outdir="."):                                        # build the two figures
    """Read the payload, draw the treemap and the distribution figure."""   # short docstring
    payload = json.load(open(path, encoding="utf-8"))               # the upstream payload
    root = payload["root"]                                          # the A-share root node
    PSEUDO = {"创业板", "北交所", "科创板", "沪市主板", "深市主板", "中小板"}   # buckets for unmapped stocks
    boards = [b for b in root.get("children", [])                     # boards with stocks
              if b.get("children") and b.get("n") not in PSEUDO]      # keep real industries only
    boards.sort(key=lambda b: -b.get("v", 0))                       # largest turnover first
    meta = payload.get("title_meta", {})                            # caption metadata
    date = payload.get("trading_date", "").replace(" closed", "")    # trading date, no status word
    generated = payload.get("generated_at", "")                     # generation stamp

    # ------------------------------------------------------------ colour scale
    lo, hi = -0.05, 0.05                                            # clip the colour scale at 5%
    cmap = mcolors.LinearSegmentedColormap.from_list(               # green -> neutral -> red
        "szu_pnl", ["#0E7C4A", "#5DB37E", "#F4F1EC", "#E0715F", "#B3202C"])   # A-share convention
    norm = mcolors.Normalize(vmin=lo, vmax=hi, clip=True)           # symmetric around zero

    # ------------------------------------------------------------ layout
    fig = plt.figure(figsize=(13.2, 6.6))                           # 16:8 canvas for the slide
    ax = fig.add_axes([0.012, 0.045, 0.86, 0.88])                    # the treemap axes
    ax.set_xlim(0, 1)                                               # unit square
    ax.set_ylim(0, 1)                                               # unit square
    ax.axis("off")                                                  # no axes or ticks
    board_rects = squarify([b.get("v", 1) for b in boards], 0, 0, 1, 1)   # outer partition
    labelled = []                                                   # stock rectangles big enough to label
    for board, (bx, by, bw, bh) in zip(boards, board_rects):        # loop over the boards
        x0, y0, w0, h0 = bx, by, bw, bh                             # unpack the tile
        pad = 0.002                                                 # inner padding in axes units
        inner = (bx + pad, by + pad, max(bw - 2 * pad, 0), max(bh - 2 * pad, 0))   # inner box
        stocks = sorted(board["children"], key=lambda s: -s.get("v", 0))   # largest first
        srects = squarify([s.get("v", 1) for s in stocks], *inner)   # inner partition
        for stock, (sx, sy, sw, sh) in zip(stocks, srects):          # draw every stock
            g = stock.get("g")                                      # its percent change
            ax.add_patch(mpatches.Rectangle((sx, sy), sw, sh, fill=True,   # the coloured tile
                                            facecolor=cmap(norm(g if g is not None else 0.0)),
                                            edgecolor="white", linewidth=0.35))   # thin white grid
            if sw * sh > 0.0022:                                    # only large tiles get a name
                labelled.append((sw * sh, ticker(stock.get("c", "")), g, sx, sy, sw, sh))   # remember it
        ax.add_patch(mpatches.Rectangle((x0, y0), w0, h0, fill=False,   # the board outline
                                        edgecolor="#2B2B2B", linewidth=1.1))   # dark frame
        gb = board.get("g", 0.0) or 0.0                             # the board's percent change
        draw_board_label(ax, bx, by, bw, bh, board.get("n", ""), gb)   # name and change, if it fits
    labelled.sort(reverse=True)                                     # biggest tiles first
    for _, name, g, sx, sy, sw, sh in labelled[:16]:                # label the largest tiles only
        ax.text(sx + sw / 2, sy + sh / 2, name, ha="center", va="center",   # centred label
                fontsize=5.0, color="white" if abs(g or 0) > 0.012 else "#2B2B2B")   # readable on light tiles
    up_n = sum(1 for b in boards if (b.get("g") or 0) > 0)             # industries that rose
    ax.set_title("A-share market treemap $\\cdot$ close of %s $\\cdot$ %d Shenwan level-1 industries (%d up, %d down)" % (
                     date, len(boards), up_n, len(boards) - up_n),   # the figure title
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)   # left aligned
    ax.text(0.0, -0.028, "Sample: top %d stocks by turnover (of %d listed) $\\cdot$ Grouping: Shenwan level-1 $\\cdot$ Source: Tencent quotes $\\cdot$ Generated %s" % (
                meta.get("n_shown", 0), meta.get("filtered_total", 0), generated),
            transform=ax.transAxes, fontsize=7.2, color="#5A5A5A")   # the source line
    cax = fig.add_axes([0.885, 0.10, 0.016, 0.62])                   # the colour bar axes
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")   # draw it
    cb.set_label("Daily return (%)", fontsize=9)                     # the colour bar label
    cb.set_ticks([-0.05, -0.025, 0, 0.025, 0.05])                    # five ticks
    cb.set_ticklabels(["-5", "-2.5", "0", "+2.5", "+5"])             # percent labels
    cb.ax.tick_params(labelsize=8)                                   # tick size
    leg = fig.add_axes([0.885, 0.025, 0.10, 0.05])                    # a small legend strip
    leg.axis("off")                                                  # no axes
    leg.text(0, 0.5, "red up, green down", fontsize=8.5, color="#5A5A5A")   # the convention note
    fig.text(0.012, 0.975, "Data Analytics in Marketing $\\cdot$ Ch. 4 Data Description and Visualization",
             fontsize=8.5, color="#8a0c3c")                          # course tag
    heat = os.path.join(outdir, "fig_ashare_heatmap.png")            # output path
    fig.savefig(heat, dpi=200, facecolor="white")                    # save the treemap
    plt.close(fig)                                                   # release the figure

    # ------------------------------------------------------------ distribution figure
    all_pct = [s.get("g", 0.0) or 0.0 for b in boards for s in b["children"]]   # every shown stock
    fig2, axes = plt.subplots(1, 2, figsize=(11.6, 3.9))             # two panels
    names = [en(b.get("n", "")) for b in boards][::-1]                # board names, smallest last
    vals = [(b.get("g", 0.0) or 0.0) * 100 for b in boards][::-1]     # their percent changes
    cols = [cmap(norm(v / 100)) for v in vals]                        # colour by sign
    axes[0].barh(names, vals, color=cols, edgecolor="#2B2B2B", linewidth=0.5)   # the bars
    axes[0].axvline(0, color="#2B2B2B", lw=0.9)                      # zero reference line
    axes[0].set_title("Industry returns (turnover-weighted)", fontsize=11)   # panel title
    axes[0].set_xlabel("Daily return (%)")                            # axis label
    axes[0].tick_params(labelsize=6.4)                                # 31 labels need small type
    axes[0].set_xlim(min(vals) - 1.7, max(vals) + 0.6)                # room for the labels outside the bars
    for i, v in enumerate(vals):                                     # annotate each bar
        axes[0].text(v + (0.06 if v >= 0 else -0.06), i, "%+.2f%%" % v, va="center",   # value label
                     ha="left" if v >= 0 else "right", fontsize=5.6)   # aligned outside the bar
    axes[1].hist(np.array(all_pct) * 100, bins=40, color="#8C8C8C", alpha=0.85)   # the histogram
    axes[1].axvline(float(np.median(all_pct)) * 100, color="#B3202C", lw=1.4, ls="--",
                    label="median %+.2f%%" % (float(np.median(all_pct)) * 100))   # the median
    axes[1].legend(fontsize=8, loc="upper left")                     # legend
    axes[1].set_title("Stock returns, top %d by turnover" % len(all_pct), fontsize=11)   # panel title
    axes[1].set_xlabel("Daily return (%)")                            # axis label
    axes[1].set_ylabel("Number of stocks")                            # axis label
    for a in axes:                                                   # tidy both panels
        a.grid(alpha=0.22)                                           # light grid
        for sp in ("top", "right"):                                   # drop two spines
            a.spines[sp].set_visible(False)                          # for a cleaner look
    fig2.tight_layout()                                              # avoid overlaps
    dist = os.path.join(outdir, "fig_ashare_distribution.png")       # output path
    fig2.savefig(dist, dpi=200, facecolor="white")                   # save the second figure
    plt.close(fig2)                                                  # release it
    up = sum(1 for g in all_pct if g > 0)                             # how many rose
    down = sum(1 for g in all_pct if g < 0)                           # how many fell
    print("industries:", len(boards), "| stocks:", len(all_pct))     # report the sample
    print("up %d | down %d | median %+.2f%%" % (up, down, float(np.median(all_pct)) * 100))   # report
    print("root: %s %+.2f%%" % (root.get("n"), (root.get("g") or 0) * 100))   # report the index
    print("wrote", heat)                                             # report the outputs
    print("wrote", dist)                                             # report the outputs


if __name__ == "__main__":                                          # run as a script
    main(sys.argv[1] if len(sys.argv) > 1 else "heatmap_industry.json",   # data payload
         sys.argv[2] if len(sys.argv) > 2 else ".")                  # output directory
