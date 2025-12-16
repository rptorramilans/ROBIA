"""
Created on Mon Sep 29 2025 ‏‎11:31:25 
Code part of ROBIA (UPC-BarcelonaTech)
Last modified on Mon Nov 24 2025
@author: raquel peñas torramilans
@contact: raquel.penas@upc.edu
"""

import os
import json
import numpy as np
import rasterio
from skimage.color import rgb2gray
from skimage.morphology import disk, dilation, opening, remove_small_objects
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import numpy.ma as ma
import matplotlib.patches as mpatches
import pandas as pd

# robia parameters
import params as p

# ==========================================================
#                   CALIBRATION (IF CHOOSED)
# ==========================================================

if p.calibration == 1:

    from extract_coordinates import extract_coordinates
    tabla_resultado = extract_coordinates()  # extreu RGB i guarda fitxer

    from calibration import calibrate
    best_model = calibrate()  # retorna dict amb millor índex

else:
    best_model = None

# ==========================================================
#                   IMAGE LIST
# ==========================================================

# choose single or multiple images
if p.processing_mode == 0:
    image_paths = [p.input_tif]
else:
    image_paths = [
        os.path.join(p.input_folder, f)
        for f in os.listdir(p.input_folder)
        if f.lower().endswith(('.tif', '.tiff'))
    ]

# ==========================================================
#                   MAIN LOOP
# ==========================================================

for path in image_paths:

    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\n--- Processing image: {name} ---")

    # 1) load image
    with rasterio.open(path) as src:
        img = src.read().transpose((1, 2, 0))
        R = src.transform
        crs = src.crs
        xWorld = np.linspace(src.bounds.left, src.bounds.right, src.width)
        yWorld = np.linspace(src.bounds.top, src.bounds.bottom, src.height)
        if src.nodata is not None:
            nodata_mask = np.all(img == src.nodata, axis=-1)
        else:
            nodata_mask = np.zeros(img.shape[:2], dtype=bool)

    img = img.astype(float)
    img[nodata_mask] = np.nan

    # 2) show original image
    plt.figure(figsize=(8, 8))
    img_norm = img[:, :, :3] / np.nanmax(img[:, :, :3])
    img_masked = ma.masked_invalid(img_norm)
    plt.imshow(img_masked, extent=[xWorld[0], xWorld[-1], yWorld[-1], yWorld[0]])
    plt.title(p.title_original)
    plt.xlabel(p.xlabel)
    plt.ylabel(p.ylabel)
    plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    plt.show()

    # 3) mask obstruction structures (optional)
    if p.processing_obstruction == 1:
        print("Creating obstruction mask...")
        img_norm = img.copy()
        img_norm -= np.nanmin(img_norm)
        if np.nanmax(img_norm) > 0:
            img_norm /= np.nanmax(img_norm)
        gray_img = rgb2gray(np.nan_to_num(img_norm, nan=0.0))
        masking = gray_img > p.obs_thresh
        masking = masking & (~nodata_mask)
        masking = dilation(masking, disk(p.obs_disk))
        masking = remove_small_objects(masking, min_size=p.obs_min)

        plt.figure(figsize=(8, 8))
        plt.imshow(masking, cmap="gray", extent=[xWorld[0], xWorld[-1], yWorld[-1], yWorld[0]])
        plt.title(p.title_mask)
        plt.xlabel(p.xlabel)
        plt.ylabel(p.ylabel)
        plt.show()

        mask_out = masking.astype(np.uint8)
        out_tif = (
            p.obstruction_mask_tif
            if p.processing_mode == 0
            else os.path.join(p.output_folder, f"{name}_obstruction_mask.tif")
        )

        with rasterio.open(
            out_tif, "w", driver="GTiff",
            height=mask_out.shape[0], width=mask_out.shape[1],
            count=1, dtype=mask_out.dtype, crs=crs, transform=R
        ) as dst:
            dst.write(mask_out, 1)

        print(f"Obstruction mask saved: {out_tif}")

    else:
        masking = np.zeros(img.shape[:2], dtype=bool)
        print("No obstruction mask applied.")

    # 4) apply obstruction mask
    img_float = img.copy()
    img_float[masking] = np.nan
    img_float[img_float == 0] = np.finfo(float).eps
    Rch, Gch, Bch = img_float[..., 0], img_float[..., 1], img_float[..., 2]

    # 5) apply index
    if p.calibration == 1:

        idx_name = best_model["Index"]
        red_threshold = best_model["red_threshold"]
        func_handle = eval(best_model["Func"], {"np": np})

        print(f"Using calibrated index: {idx_name}")

    else:

        idx_name = p.index_single
        red_threshold = p.index_thresh

        if idx_name.lower() == "r/g":
            func_handle = lambda R, G, B: R / (G + np.finfo(float).eps)
        elif idx_name.lower() == "r/b":
            func_handle = lambda R, G, B: R / (B + np.finfo(float).eps)
        elif idx_name.lower() == "r/(b+g)":
            func_handle = lambda R, G, B: R / (B + G + np.finfo(float).eps)
        else:
            raise ValueError(f"Unknown index specified in params: {idx_name}")

        print(f"Using manual index: {idx_name}")

    idxR = func_handle(Rch, Gch, Bch)

    # 6) rhodamine mask
    if p.calibration == 1:
        a, b = best_model["Poly"]
        if a >= 0:
            mask_red = (idxR > red_threshold)
        else:
            mask_red = (idxR < red_threshold)
    else:
        mask_red = (idxR > red_threshold)

    mask_red = mask_red & (~masking) & (~nodata_mask)
    mask_red = binary_fill_holes(mask_red)
    mask_red = opening(mask_red, disk(p.rho_disk))
    mask_red = remove_small_objects(mask_red, min_size=p.rho_min)

    # 7) apply calibration model
    conc_map = np.full_like(idxR, np.nan)

    if p.calibration == 1:
        conc_map[mask_red] = np.polyval(best_model["Poly"], idxR[mask_red])
        print("Applied calibrated polynomial model.")
    else:
        idx_valid = idxR[mask_red]
        conc_map[mask_red] = np.interp(
            idx_valid,
            (np.nanmin(idx_valid), np.nanmax(idx_valid)),
            (p.conc_min, p.conc_max)
        )
        print("Applied normalized concentration scaling (no calibration).")


    plt.figure()
    plt.imshow(conc_map, extent=[xWorld[0], xWorld[-1], yWorld[-1], yWorld[0]],
               vmin=p.conc_min, vmax=p.conc_max)
    plt.colorbar(label=p.colorbar_label)
    plt.title(p.title_conc_map)
    plt.xlabel(p.xlabel)
    plt.ylabel(p.ylabel)
    plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    # SAVE HIGH-RES FIGURE (for paper)
    plt.savefig("concentration_map_highres.svg", dpi=600, bbox_inches="tight")    
    plt.show()

    # save concentration map (RGBA)
    norm = colors.Normalize(vmin=p.conc_min, vmax=p.conc_max)
    cmap = plt.get_cmap("viridis")
    rgba_img = (cmap(norm(conc_map)) * 255).astype(np.uint8)
    alpha = (~np.isnan(conc_map)).astype(np.uint8) * 255

    if p.processing_mode == 0:
        conc_tif = p.concentration_map_tif
    else:
        conc_tif = os.path.join(p.output_folder, f"{name}_concentration_map.tif")

    with rasterio.open(
        conc_tif, "w", driver="GTiff",
        height=rgba_img.shape[0], width=rgba_img.shape[1],
        count=4, dtype="uint8", crs=crs, transform=R
    ) as dst:
        for i in range(3):
            dst.write(rgba_img[:, :, i], i+1)
        dst.write(alpha, 4)
    print(f"Concentration map saved: {conc_tif}")

# 8) segmentation map

    seg_map = np.zeros(masking.shape, dtype=np.uint8)
    seg_map[(~masking) & (~mask_red) & (~nodata_mask)] = 1   # water
    seg_map[masking] = 2                                     # obstruction
    seg_map[mask_red] = 3                                    # dye
    seg_map[nodata_mask] = 255
    seg_rgb = np.zeros((*seg_map.shape, 4), dtype=float)
    seg_rgb[seg_map == 1] = [0, 0, 1, 1]     # blue
    seg_rgb[seg_map == 2] = [0.5, 0.5, 0.5, 1]  # grey
    seg_rgb[seg_map == 3] = [1, 0, 0, 1]     # red
    seg_rgb[seg_map == 255] = [1, 1, 1, 0]     # transparent

    # plt.figure()
    # plt.imshow(seg_rgb, extent=[xWorld[0], xWorld[-1], yWorld[-1], yWorld[0]])
    # plt.title(p.segmentation_map_tif)
    # plt.xlabel(p.xlabel)
    # plt.ylabel(p.ylabel)
    # class_labels = {1: "Water", 2: "Obstruction", 3: "Dye"}
    # colors_seg = {1: "blue", 2: "gray", 3: "red"}

    # plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    # plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    # plt.tight_layout()
    # plt.show()

    plt.figure()
    plt.imshow(seg_rgb, extent=[xWorld[0], xWorld[-1], yWorld[-1], yWorld[0]])
    plt.title(p.segmentation_map_tif)
    plt.xlabel(p.xlabel)
    plt.ylabel(p.ylabel)

# Leyenda (cuadraditos con color)
    legend_patches = [
        mpatches.Patch(color=[0, 0, 1], label="Water"),
        mpatches.Patch(color=[0.5, 0.5, 0.5], label="Obstruction"),
        mpatches.Patch(color=[1, 0, 0], label="Dye"),
]
    plt.legend(handles=legend_patches, loc="upper right")

# Formato ejes
    plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))

    plt.tight_layout()
    plt.show()

    if p.processing_mode == 0:
        seg_tif = p.segmentation_map_tif
    else:
        seg_tif = os.path.join(p.output_folder, f"{name}_segmentation_map.tif")

    with rasterio.open(
            seg_tif, "w", driver="GTiff",
            height=seg_map.shape[0], width=seg_map.shape[1],
            count=1, dtype="uint8", crs=crs, transform=R,
            nodata=255
        ) as dst:
            dst.write(seg_map, 1)

    print(f"Segmentation map saved: {seg_tif}")

print("\n All images processed successfully.")


# ==========================================================
#                   OUTCOME SUMMARY
# ==========================================================

report = os.path.join(p.output_folder if p.processing_mode != 0 else ".", "robia_report.txt")
with open(report, "w") as frep:

    st = "\n====================================================================\n"
    frep.write(st)
    print(st)

    st = "                       ROBIA OUTCOME REPORT\n"
    frep.write(st)
    print(st)

    st = "====================================================================\n\n"
    frep.write(st)
    print(st)

    # Processing mode
    mode_str = "SINGLE IMAGE" if p.processing_mode == 0 else "BATCH MODE"
    st = "Processing mode:                  {:>20s}\n".format(mode_str)
    frep.write(st)
    print(st)

    # Number of images processed
    st = "Images processed:                 {:>20d}\n".format(len(image_paths))
    frep.write(st)
    print(st)

    # Obstruction masking
    mask_status = "YES" if p.processing_obstruction == 1 else "NO"
    st = "Obstruction masking applied:       {:>20s}\n".format(mask_status)
    frep.write(st)
    print(st)

    # Preprocessing (in your case, it's implicit — you can link it to any preprocessing step you have)
    preproc_status = "YES" if p.processing_obstruction == 1 else "NO"
    st = "Preprocessing performed:           {:>20s}\n".format(preproc_status)
    frep.write(st)
    print(st)

    try:
        model_name = best_model.get("Index", "N/A")
        model_poly = best_model.get("Poly", [0, 0])

        st = "Best model selected:              {:>20s}\n".format(model_name)
        frep.write(st)
        print(st)

        st = "Calibration polynomial:            {:>20s}\n".format(str(model_poly))
        frep.write(st)
        print(st)

    except Exception as e:
        st = f"Best model information unavailable ({e})\n"
        frep.write(st)
        print(st)

    # Save location
    st = "\nResults saved in: {}\n".format(p.output_folder if p.processing_mode != 0 else os.getcwd())
    frep.write(st)
    print(st)

    st = "\n====================================================================\n"
    frep.write(st)
    print(st)

    st = "----------------------- ROBIA PROCESSING DONE ----------------------\n"
    frep.write(st)
    print(st)

print("\nReport file created: {}".format(report))


