"""
Created on Wed Oct 1 2025 ‏‎12:37:26
Code part of RHOD (UPC-BarcelonaTech)
Last modified on Fri Nov 28 2025
@author: raquel peñas torramilans
@contact: raquel.penas@upc.edu
"""
### preprocessing

coords_file = "E:/UPC/02_Rhodamine_Methods/Py/coords_best.csv"               # Fitxer amb HORA, PERFIL, X, Y, FLUORIMETRO
input_folder = "E:/UPC/02_Rhodamine_Methods/Py"      
output_file = "E:/UPC/02_Rhodamine_Methods/Py/insitu_rgb.txt"  # Fitxer de sortida (us per best_model.py)

### input files
processing_mode = 1                                        # 1 for multiple, 0 for single
calibration = 1                                        # 1 for YES, 0 for NO
input_tif = "16_37_nou.tif"                                   # original georreferenced image
input_folder = "E:/UPC/02_Rhodamine_Methods/Py"     # folder with georreferenced images

### output files
obstruction_mask_tif = "aquaculture_mask.tif"   # obstruction zone mask
index_map_tif = "index_map.tif"                 # concentration map
rhodamine_mask_tif = "rhoda_mask.tif"           # rhodamine mask
concentration_map_tif = "concentration_map.tif" # concentration map
segmentation_map_tif = "segmentation_map.tif" # concentration map
output_folder = "E:/UPC/02_Rhodamine_Methods/Py/results/last" 

### obstruction mask parameters
processing_obstruction = 1                      # 1 for YES, 0 for NO
#obs_thresh = 200 / 255.0 
obs_thresh = 200 / 255.0    # satellite                    # threshold for detecting white structure. scale [0,1]
#obs_thresh = 150 / 255.0 koot
#obs_disk = 5
obs_disk = 0 # satellite                                 # disk radius for expansion
#obs_min = 100                                   	# minimum size of objects to be removed in the mask
obs_min = 0 # satellite 
### rhodamine detection parameters
index_single = "r/g"                             # R/G
index_thresh = 1   
thr_fraction = 0.02                             # Threshold target fraction of max concentration

#red_thresh = 0.95                              # Umbral para índice de rojez
rho_disk = 0                                    # disk radius for expansion
#rho_min = 500                                  	# minimum size of objects to be removed in the selection
rho_min = 0 

### concentration map parameters
conc_min = 0                                    # minimum concentration value
conc_max = 67                                   # maximum concentration value
colorbar_label = "rhodamine wt concentration (ppb)"          # colorbar label

### labels for graphics
title_original = "original image"
title_mask = "obstruction mask"
title_rhodamine = "rhodamine detection area"
title_conc_map = "concentration map"
segmentation_map_tif = "classification"
xlabel = "latitude"
ylabel = "longitude"

