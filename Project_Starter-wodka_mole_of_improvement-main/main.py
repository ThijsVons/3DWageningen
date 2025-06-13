#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:12:43 2024

Authors: wodka_mole_of_improvement: Engelsma, Naut & Visser, Rianne & Bartek, Michal & Vons, Thijs

Sources are cited locally, alongside their date of acquisition
At some moments ChatGPT was consulted to correct an error in a code, writen by teammembers.
"""

# Import necessary packages
import os
import numpy as np
import Python as funcs

# Make a data and output folder if they don't exist already 
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('output'):
    os.makedirs('output')

# Calculate and show height of buildings in a specified area of Wageningen
# bbox_specified_area = funcs.get_bbox("Wageningen", 5.667, 51.970, 150) # Wageningen bus station
# bbox_specified_area = funcs.get_bbox("Wageningen", 5.662, 51.983, 1000) # Area around WUR campus
bbox_specified_area = funcs.get_bbox("Wageningen", 5.662, 51.965, 80) # Wageningen centre
bbox_Wageningen = funcs.get_bbox("Wageningen", 5.663, 51.974, 2000) # (Nearly) Entirety of Wageningen

# Load DTM and DSM
dtm, dtm_meta, dtm_affine = funcs.crop_tiles("M", "../Raw_data_project/", bbox_specified_area)
dsm, dsm_meta, dsm_affine = funcs.crop_tiles("R", "../Raw_data_project/", bbox_specified_area)

# Preprocess data
# Convert to np array and change no data to actually NaN values
dtm_filtered = dtm.copy()
dsm_filtered = dsm.copy()

dtm_filtered[dtm_filtered < -10E2] = np.nan # No data values are classified -9999.
dsm_filtered[dsm_filtered < -10E2] = np.nan # No data values are classified -9999.
# Do calculations on data
# Interpolate dtm data
dtm_interpolated = funcs.interpolate_raster(dtm_filtered.copy())

# Calculate CHM
CHM = dsm_filtered - dtm_interpolated
#%%
# Write away CHM as a tif file
funcs.write_CHM_as_tif(dtm_meta, dtm_affine, CHM, 'data/CHM.tif')

# Plotting the data
# Call function to plot the data
funcs.plot_data(CHM, dsm_filtered, dtm_filtered, dtm_interpolated, True)
#%%
# Calculate building heights of polygons
funcs.get_heights("../Raw_data_project/OSM_buildings_Wageningen.geojson", bbox_specified_area, CHM, dtm, dtm_meta)

# Compare the heights of the OSM data and our calculations
funcs.compare_heights("../Raw_data_project/OSM_buildings_Wageningen.geojson")

# Visualize calculated data mean and max height
funcs.create_3D_html_comparing_to_OSM('data/comparing_heights.geojson','output/Wageningen_buildings_comparison.html')

# Crop to smaller area to be able to do 3D map visualisation
funcs.crop_building_vector(bbox_specified_area)

# Visualize calculated data mean and max height
funcs.create_3D_html_Wageningen('data/specified_buildings.geojson','output/Wageningen_buildings_visualisation.html')
