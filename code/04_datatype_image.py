"""04_datatype_image.py -- image data: synthetic sample images + pixel-level visualization."""  # module comment
import numpy as np                                                    # numerical computing (an image is just a matrix)
import matplotlib.pyplot as plt                                       # plotting
from _common import FIG, DAT, save                                    # shared helpers (we write image samples directly here)

# ---------------------------------------------------------------- 1) generate three classes of synthetic images (32x32 grayscale)
H = W = 32                                                            # image height and width (32 px, easy to show in class)
yy, xx = np.mgrid[0:H, 0:W]                                           # build the pixel coordinate grid
cy, cx = (H - 1) / 2, (W - 1) / 2                                      # compute the image center coordinates
circle = (((yy - cy) ** 2 + (xx - cx) ** 2) < 9 ** 2).astype(float)    # class 0: a filled circle of radius 9
square = ((np.abs(yy - cy) <= 8) & (np.abs(xx - cx) <= 8)).astype(float)  # class 1: a square with side length 17
triangle = ((xx >= yy) & (xx <= W - 1 - yy)).astype(float)             # class 2: an isosceles triangle (pointed top, wide base)
protos = [circle, square, triangle]                                    # the prototype pattern of each of the three classes
labels = ["circle", "square", "triangle"]                              # names of the three classes

imgs, lab = [], []                                                     # initialize the sample list and the label list
for i, p in enumerate(protos):                                         # loop over the three classes
    for k in range(4):                                                 # generate 4 samples per class
        noise = np.random.default_rng(100 + i * 10 + k).normal(0, 0.08, (H, W))  # add noise to mimic real capture
        imgs.append(np.clip(p + noise, 0, 1))                          # add the noise and clip to [0,1]
        lab.append(i)                                                  # record the class id of this sample
imgs = np.stack(imgs)                                                  # stack into a (12, 32, 32) sample tensor
np.save(DAT / "image_samples.npy", imgs)                               # save as an npy sample (a common ML format)
np.save(DAT / "image_labels.npy", np.array(lab))                       # save the label array

grid = imgs.reshape(3, 4, H, W).transpose(0, 2, 1, 3).reshape(3 * H, 4 * W)  # tile into a 3-row, 4-column mosaic image
plt.imsave(FIG / "fig_image_montage_raw.png", grid, cmap="gray")       # also save the mosaic as a single PNG sample

# ---------------------------------------------------------------- 2) color image and channel decomposition
rgb = np.zeros((H, W, 3))                                              # initialize a color image (three RGB channels)
rgb[..., 0] = xx / (W - 1)                                             # red channel: increases linearly from left to right
rgb[..., 1] = yy / (H - 1)                                             # green channel: increases linearly from top to bottom
rgb[..., 2] = 0.8 * circle                                             # blue channel: bright only inside the circle

# ---------------------------------------------------------------- 3) visualization
fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.9))                    # one row, four columns: grayscale samples / color / channel / histogram
axes[0].imshow(grid, cmap="gray")                                      # first: grayscale sample mosaic (3 classes x 4 images)
axes[0].set_title("Grayscale image samples 32x32 (3 classes x 4)")     # first title
axes[0].set_xticks([]); axes[0].set_yticks([])                          # hide the ticks so it looks more like an "image"
axes[0].grid(False)                                                    # turn off the grid
axes[1].imshow(rgb)                                                    # second: color image
axes[1].set_title("Color image = three R/G/B matrices")                # second title
axes[1].set_xticks([]); axes[1].set_yticks([])                          # hide the ticks
axes[1].grid(False)                                                    # turn off the grid
axes[2].imshow(circle, cmap="gray")                                    # third: a single channel (grayscale matrix)
axes[2].set_title("Single channel I(x, y) in [0, 255]")                # third title
axes[2].set_xticks([]); axes[2].set_yticks([])                          # hide the ticks
axes[2].grid(False)                                                    # turn off the grid
axes[3].hist((imgs * 255).ravel(), bins=32, color="#8A0C3C", alpha=0.85)  # last: pixel-intensity histogram
axes[3].set_title("Pixel intensity distribution")                      # last title
axes[3].set_xlabel("Grayscale value")                                  # last x-axis meaning
plt.tight_layout()                                                     # adjust the layout
save(fig, "fig_image.png")                                             # save the figure
