"""08_datatype_spatiotemporal.py -- spatiotemporal data: trajectory samples + heatmap/speed-curve visualization."""  # module comment
import numpy as np                                                    # numerical computing
import pandas as pd                                                   # tabular handling
import matplotlib.pyplot as plt                                       # plotting
from _common import RNG, save, dump_csv                               # shared helpers

# ---------------------------------------------------------------- 1) build three trajectories (lat/lon around Nanshan, Shenzhen)
lat0, lon0 = 22.5333, 113.9300                                        # starting point: near Nanshan district, Shenzhen
tracks = ["SZ-B-A1234", "SZ-B-B5678", "SZ-B-C9012"]                   # three vehicle/rider ids
rows = []                                                             # initialize the per-point record list
for i, tid in enumerate(tracks):                                      # generate one trajectory at a time
    n = 120                                                           # 120 sample points per trajectory (about 2 minutes)
    speed = RNG.normal(9.0 + 2 * i, 1.5, n).clip(2, 18)               # speed in m/s: different means, with random fluctuation
    head = np.cumsum(RNG.normal(0, 0.12, n))                          # heading random walk (mimics turning)
    dlat = np.cos(head) * speed * 5 / 111000.0                        # latitude increment: displacement / about 111 km per degree
    dlon = np.sin(head) * speed * 5 / (111000.0 * np.cos(np.radians(lat0)))  # longitude increment: needs a cosine correction by latitude
    la = lat0 + np.cumsum(dlat) + RNG.normal(0, 1e-5, n)              # cumulate into the latitude series (plus GPS noise)
    lo = lon0 + np.cumsum(dlon) + RNG.normal(0, 1e-5, n)              # cumulate into the longitude series (plus GPS noise)
    for k in range(n):                                                # write one record per point
        rows.append({"track_id": tid, "t_sec": k * 5,                # trajectory id and timestamp (one point every 5 seconds)
                     "lat": round(float(la[k]), 6),                   # latitude (6 decimals, about 0.1 m)
                     "lon": round(float(lo[k]), 6),                   # longitude
                     "speed_mps": round(float(speed[k]), 2)})         # instantaneous speed
traj = pd.DataFrame(rows)                                             # assemble into a long table (one row per point)
dump_csv(traj, "spatiotemporal_tracks.csv")                           # dump the spatiotemporal samples

# ---------------------------------------------------------------- 2) visualization
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9))                   # one row, three columns: trajectory / density heatmap / speed time series
for tid in tracks:                                                    # plot one trajectory at a time
    sub = traj[traj["track_id"] == tid]                               # take the sub-table for this trajectory
    axes[0].plot(sub["lon"], sub["lat"], "-", lw=1.4, label=tid)      # left: the lon/lat polyline is the trajectory
axes[0].set_title("Trajectory (lat/lon series)")                      # left title
axes[0].set_xlabel("Longitude / deg E")                               # left x-axis meaning
axes[0].set_ylabel("Latitude / deg N")                                # left y-axis meaning
axes[0].legend(fontsize=7)                                            # show the vehicle legend
axes[0].ticklabel_format(useOffset=False)                             # disable the scientific offset to show real lat/lon

hb = axes[1].hexbin(traj["lon"], traj["lat"], gridsize=18,            # middle: hexagon-binned heatmap (spatial density)
                    cmap="magma", mincnt=1)                           # only show bins that contain points
axes[1].set_title("Spatial density heatmap (hexbin)")                 # middle title
axes[1].set_xlabel("Longitude / deg E")                               # middle x-axis meaning
axes[1].ticklabel_format(useOffset=False)                             # disable the scientific offset
axes[1].grid(False)                                                   # a heatmap needs no grid

for tid in tracks:                                                    # plot the speed curve of each trajectory
    sub = traj[traj["track_id"] == tid]                               # take this trajectory
    axes[2].plot(sub["t_sec"], sub["speed_mps"], lw=1.0,              # right: line of speed over time
                 label=tid)                                           # add the legend label
axes[2].set_title("Speed over time (spatiotemporal -> time series)")  # right title
axes[2].set_xlabel("Time / s")                                        # right x-axis meaning
axes[2].set_ylabel("Speed / (m/s)")                                   # right y-axis meaning
plt.tight_layout()                                                    # adjust the layout
save(fig, "fig_spatiotemporal.png")                                   # save the figure
