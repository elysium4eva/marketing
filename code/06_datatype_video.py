"""06_datatype_video.py -- video data: synthetic frame sequence + key-frame/temporal-signal visualization."""  # module comment
import numpy as np                                                    # numerical computing
import matplotlib.pyplot as plt                                       # plotting
from _common import DAT, save                                         # shared helpers (only the data folder and the save function)

H = W = 48                                                            # resolution of each frame: 48x48
NF = 12                                                               # 12 frames in total (illustrating 1 second at 12 fps)
frames, means = [], []                                                # initialize the frame list and the per-frame brightness means
for k in range(NF):                                                   # generate the video content frame by frame
    img = np.zeros((H, W))                                            # all-black background
    yy, xx = np.mgrid[0:H, 0:W]                                       # pixel coordinate grid
    cy, cx = H / 2, 2 + k * (W - 4) / (NF - 1)                        # ball center: moves from left to right over time
    ball = ((yy - cy) ** 2 + (xx - cx) ** 2) < 5 ** 2                 # mask of a ball with radius 5
    img[ball] = 220                                                   # set the ball pixels to a bright value
    bar_x = (W * k / NF)                                              # vertical light-bar position: moves over time
    img[:, int(bar_x):int(bar_x) + 3] += 60                           # brighten the light-bar pixels
    frames.append(np.clip(img, 0, 255))                               # append to the frame sequence
    means.append(float(img[ball].mean()))                             # record the mean brightness of the ball region
frames = np.stack(frames)                                             # stack into a (12, 48, 48) video tensor
np.save(DAT / "video_frames.npy", frames)                             # save the frame-tensor sample
(DAT / "video_frames").mkdir(parents=True, exist_ok=True)             # create the sampled-frame image folder
for i in range(NF):                                                   # export each frame as PNG to mimic "saving sampled frames"
    plt.imsave(DAT / "video_frames" / ("frame_%02d.png" % (i + 1)),   # save frame i to data/video_frames/
               frames[i], cmap="gray")                                # save with a grayscale color map
print("[data]", DAT / "video_frames.npy")                             # print the sample location

# ---------------------------------------------------------------- visualization
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                   # one row, three columns: key frame / frame difference / temporal signal
axes[0].imshow(frames[0], cmap="gray")                                # left: frame 1
axes[0].set_title("Key frame t=1 (48x48 grayscale)")                  # left title
axes[0].set_xticks([]); axes[0].set_yticks([])                         # hide the ticks
axes[0].grid(False)                                                   # turn off the grid
axes[1].imshow(np.abs(frames[6] - frames[0]), cmap="inferno")          # middle: difference between frame 7 and frame 1 (motion detection)
axes[1].set_title("Frame difference |I(t=7) - I(t=1)| (motion detection)")  # middle title
axes[1].set_xticks([]); axes[1].set_yticks([])                         # hide the ticks
axes[1].grid(False)                                                   # turn off the grid
axes[2].plot(range(1, NF + 1), means, "-o", color="#8A0C3C", ms=4)     # right: time series made of per-frame brightness means
axes[2].set_title("Per-frame signal: mean brightness vs t")            # right title
axes[2].set_xlabel("Frame index t")                                    # right x-axis meaning
axes[2].set_ylabel("Mean brightness")                                  # right y-axis meaning
plt.tight_layout()                                                    # adjust the layout
save(fig, "fig_video.png")                                            # save the figure
