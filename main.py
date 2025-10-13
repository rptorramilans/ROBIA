"""
Created on Mon Sep 29 2025 ‏‎11:31:25 
Code part of RHOD (UPC-BarcelonaTech)
Last modified on Mon Oct 13 2025
@author: raquel peñas torramilans
@contact: raquel.penas@upc.edu
"""

import numpy as np
import rasterio
from skimage.color import rgb2gray
from skimage.morphology import disk, dilation, opening, remove_small_objects
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter

# rhod parameters
import params as p
from best_model import best_model 

# 1) loading georeferenced image

with rasterio.open(p.input_tif) as src:
    img = src.read().transpose((1, 2, 0))
    R = src.transform
    crs = src.crs
    xWorld = np.linspace(src.bounds.left, src.bounds.right, src.width)
    yWorld = np.linspace(src.bounds.top, src.bounds.bottom, src.height)

plt.figure
plt.imshow(img, extent=[xWorld[0], xWorld[-1], yWorld[-1], yWorld[0]])
plt.title(p.title_original)
plt.xlabel(p.xlabel)
plt.ylabel(p.ylabel)
plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
plt.show()
print(f"original image loaded: {p.input_tif}")

# 2) masking obstruction structures

gray_img = rgb2gray(img.astype(np.uint8))
masking = gray_img > p.obs_thresh
masking = dilation(masking, disk(p.obs_disk))
masking = remove_small_objects(masking, min_size=p.obs_min)

plt.figure()
plt.imshow(masking, cmap="gray", extent=[xWorld[0], xWorld[-1], yWorld[-1], yWorld[0]])
plt.title(p.title_mask)
plt.xlabel(p.xlabel)
plt.ylabel(p.ylabel)
plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
plt.show()

mask_out = masking.astype(np.uint8)
with rasterio.open(
    p.obstruction_mask_tif,
    "w",
    driver="GTiff",
    height=mask_out.shape[0],
    width=mask_out.shape[1],
    count=1,
    dtype=mask_out.dtype,
    crs=crs,
    transform=R,
) as dst:
    dst.write(mask_out, 1)

print(f"obstruction mask saved: {p.obstruction_mask_tif}")

# 3) apply mask to RGB channels
img_float = img.astype(float)
img_float[masking] = np.nan
img_float[img_float == 0] = np.finfo(float).eps
Rch, Gch, Bch = img_float[..., 0], img_float[..., 1], img_float[..., 2]

# 4) redness selection using best_model

import pandas as pd
from skimage.morphology import disk, opening, remove_small_objects
from scipy.ndimage import binary_fill_holes

T = pd.read_csv("results_indices.txt", sep="\t")
idx_name = best_model["Index"]
row = T.loc[T["Index"] == idx_name]
if row.empty:
    raise ValueError(f"No se encontró el índice '{idx_name}' en results_indices.txt")
red_threshold = row["red_threshold"].values[0]
print(f"Using saved threshold {red_threshold:.4f} for index '{idx_name}'")

func_code = best_model["Func"]
if isinstance(func_code, str):
    func_handle = eval(func_code, {"np": np})
else:
    func_handle = func_code

idxR = func_handle(Rch, Gch, Bch)
mask_red = (idxR > red_threshold) & (~masking)

mask_red = binary_fill_holes(mask_red)
mask_red = opening(mask_red, disk(p.rho_disk))
mask_red = remove_small_objects(mask_red, min_size=p.rho_min)

plt.figure()
plt.imshow(~mask_red, cmap="gray", extent=[xWorld[0], xWorld[-1], yWorld[-1], yWorld[0]])
plt.title(f"{p.title_rhodamine} ({idx_name})")
plt.xlabel(p.xlabel)
plt.ylabel(p.ylabel)
plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
plt.show()

mask_red_out = (~mask_red).astype(np.uint8)
with rasterio.open(
    p.rhodamine_mask_tif,
    "w",
    driver="GTiff",
    height=mask_red_out.shape[0],
    width=mask_red_out.shape[1],
    count=1,
    dtype=mask_red_out.dtype,
    crs=crs,
    transform=R,
) as dst:
    dst.write(mask_red_out, 1)

print(f"Rhodamine mask saved: {p.rhodamine_mask_tif}")

# 5) apply calibration model
conc_map = np.full_like(idxR, np.nan)
conc_map[mask_red] = np.polyval(best_model["Poly"], idxR[mask_red])
conc_map[masking] = np.nan
conc_map[conc_map < p.conc_min] = np.nan

plt.figure()
plt.imshow(
    conc_map,
    extent=[xWorld[0], xWorld[-1], yWorld[-1], yWorld[0]],
    vmin=p.conc_min,
    vmax=p.conc_max)
plt.colorbar(label=p.colorbar_label)
plt.title(p.title_conc_map)
plt.xlabel(p.xlabel)
plt.ylabel(p.ylabel)
plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
plt.show()

norm = colors.Normalize(vmin=p.conc_min, vmax=p.conc_max)
cmap = plt.get_cmap("viridis")
rgba_img = (cmap(norm(conc_map)) * 255).astype(np.uint8)
alpha = (~np.isnan(conc_map)).astype(np.uint8) * 255

with rasterio.open(
    p.concentration_map_tif,
    "w",
    driver="GTiff",
    height=rgba_img.shape[0],
    width=rgba_img.shape[1],
    count=4,
    dtype="uint8",
    crs=crs,
    transform=R,
) as dst:
    dst.write(rgba_img[:, :, 0], 1)
    dst.write(rgba_img[:, :, 1], 2)
    dst.write(rgba_img[:, :, 2], 3)
    dst.write(alpha, 4)

print(f"Concentration map saved: {p.concentration_map_tif}")
