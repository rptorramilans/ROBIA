"""
Created on Wed Oct 1 2025 ‏‎12:37:26
Code part of RHOD (UPC-BarcelonaTech)
Last modified on Mon Oct 13 2025
@author: raquel peñas torramilans
@contact: raquel.penas@upc.edu
"""

### input files
input_tif = "11_16.tiff"                       # original georreferenced image

### output files
obstruction_mask_tif = "aquaculture_mask.tif"   # obstruction zone mask
rhodamine_mask_tif = "rhoda_mask.tif"           # rhodamine mask
concentration_map_tif = "concentration_map.tif" # concentration map
modelo_mat = "concentration_model.mat"          # concentration model

### obstruction mask parameters
obs_thresh = 200 / 255.0                        # threshold for detecting white structure. scale [0,1]
obs_disk = 5                                    # disk radius for expansion
obs_min = 500                                   # minimum size of objects to be removed in the mask

### rhodamine detection parameters
red_thresh = 0.95                               # Umbral para índice de rojez
rho_disk = 1                                    # disk radius for expansion
rho_min = 2000                                  # minimum size of objects to be removed in the selection

### concentration map parameters
conc_min = 0                                    # minimum concentration value
conc_max = 60                                   # maximum concentration value
colorbar_label = "rhodamine wt concentration (ppb)"          # colorbar label

### labels for graphics
title_original = "original image"
title_mask = "obstruction mask"
title_rhodamine = "rhodamine detection area"
title_conc_map = "concentration map"
xlabel = "latitude"
ylabel = "longitude"
