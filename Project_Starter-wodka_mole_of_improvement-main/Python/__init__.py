#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:27:51 2024

Authors: wodka_mole_of_improvement: Engelsma, Naut & Visser, Rianne & Bartek, Michal & Vons, Thijs

Sources are cited localy, alongside their date of acquisition
"""

def print_nan_percentage(data):
    """
    Functions that prints the percentage of NaN elements in a numpy array of size (1 x n x m)
    Parameters
    ----------
    data : 3D numpy array (1 x n x m)
        Array containing either floats or NaNs.

    Prints
    -------
    Percentage of NaN elements in the input array
    """
    # Import packages   
    import numpy as np
    
    # Check whether the input has the correct type and dimensions
    if np.shape(data)[0] != 1 or len(np.shape(data)) != 3 or type(data) != np.ndarray:
        raise ValueError('Input data of numpy array type and should only contain one explicit 2D layer')
    
    # Calculate number of pixels which are either NaN or not NaN
    number_of_nan = np.sum(np.isnan(data))
    number_of_nonnan = np.sum(~np.isnan(data))
    data_name = f'{data=}'.split('=')[0]
    
    # Calculate and print the NaN percentage
    print(f'The number of no data pixels in "{data_name}" is: {number_of_nan/(number_of_nan+number_of_nonnan)*100:.2f}%')   
    
def get_bbox (name, lon, lat, distance):
    """
    Function that calculates a boundary box based on input coordinates (WGS84) and a specified distance around the POI.
    'name' is already asked as input for later functionality

    Parameters
    ----------
    name : string
        Name of the location cited, this can be used later on for saving the file with an appropriate name.
    lon : float
        longitude of the POI.
    lat : float
        latitude of the POI.
    distance : float
        distance by which the POI is buffered.

    Returns
    -------
    bbox : GeoPandas.DataFrame
        Define bounding box for a small part of Wageningen (Later we can ask for a central coordinate to create a box around - predefined range with possibility to change).

    """
    
    # Import necessary functions
    import pandas as pd
    import geopandas as gpd
    
    # Check whether the inputs are correct
    if type(name) != str:
        raise ValueError('Please input a string as the name of the location')
    
    if type(lon) != float and type(lon)!= int:
        raise ValueError('Please input a float or int for "lon"')
    
    if type(lat) != float and type(lat)!= int:
        raise ValueError('Please input a float or int for "lat"')
    
    if type(distance) != float and type(distance) != int:
        raise ValueError('Please input a float or int for "distance"')
    
    if distance > 2500 or distance < 0:
        raise ValueError('Buffer distance needs to be kept between 0 and 2500 meters due to computation limits.')
     
    else:
    # Make a Pandas Data Frame containing a city name and its coordinates
        coords = pd.DataFrame(
            {
                "City": [name],
                "x": [lon],
                "y": [lat],
            }
        )
    
        # Return a geopandas dataframe with a point geometry
        POI = gpd.GeoDataFrame(
            coords, geometry=gpd.points_from_xy(coords.x, coords.y), crs="EPSG:4326"
        )
        
        # Set the crs to RD_NEW
        POI_RD = POI.to_crs("EPSG:28992")
    
        # buffer the point by distance to get a polygon.
        POI_RD["buffered"] = POI_RD.buffer(distance)
    
    return POI_RD
    
def plot_data(CHM, DSM, DTM, DTM_interpolated, save_plot=False): 
    """
    Function that plots input data (CHM, DSM and DTM) of a certain extent. Also saves the fiugre if requested.

    Parameters
    ----------
    CHM : 3D numpy array (1 x n x m)
        Canopy Height Model data.
    DSM : 3D numpy array (1 x n x m)
        DSM data.
    DTM : 3D numpy array (1 x n x m)
        DTM data.
    DTM_interpolated : 3D numpy array (1 x n x m)
        interpolated DTM data
    save_plot: Boolean
        Save plot if True

    Returns
    -------
    None.

    """
       
    # Import necessary packages
    import matplotlib.pyplot as plt
    import numpy as np
    
    if np.shape(CHM)[0] != 1 or len(np.shape(CHM)) != 3 or type(CHM) != np.ndarray:
        raise ValueError('CHM input data of numpy array type and should only contain one explicit 2D layer')
        
    if np.shape(DSM)[0] != 1 or len(np.shape(DSM)) != 3 or type(DSM) != np.ndarray:
        raise ValueError('DSM input data of numpy array type and should only contain one explicit 2D layer')
    
    if np.shape(DTM)[0] != 1 or len(np.shape(DTM)) != 3 or type(DTM) != np.ndarray:
        raise ValueError('DTM input data of numpy array type and should only contain one explicit 2D layer')
    
    if np.shape(DTM_interpolated)[0] != 1 or len(np.shape(DTM_interpolated)) != 3 or type(DTM_interpolated) != np.ndarray:
        raise ValueError('Interpolated DTM input data of numpy array type and should only contain one explicit 2D layer')
    
    if type(save_plot) != bool:
        raise ValueError('Please input a boolean for "save_fig"')
        
    # Extract the seperate extent values
    total_extent = len(CHM[0])
    min_plot = int(total_extent/2-500)
    max_plot = int(total_extent/2+500)
    
    # Calculate dimensions of CHM
    depth, width, height = np.shape(CHM)
    
    # If CHM dimensions are too large only plot a small part
    if depth * width * height > 1000 * 1000:
        print("""-----------------------------
Dataset too large for the function plot_data, only plotting the center of 500 m x 500 m""")
        fig, ax = plt.subplots(2, 2, figsize=(9, 8))
        CHM_plot = ax[0,0].imshow(CHM[0,min_plot:max_plot,min_plot:max_plot])                # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html, date of access: 08-10-2024
        ax[0,1].imshow(DSM[0,min_plot:max_plot,min_plot:max_plot])
        ax[1,0].imshow(DTM[0,min_plot:max_plot,min_plot:max_plot])
        ax[1,1].imshow(DTM_interpolated[0,min_plot:max_plot,min_plot:max_plot])
    
    else:
        # Create a figure and axes and plot data
        fig, ax = plt.subplots(2, 2, figsize=(9, 8))
        CHM_plot = ax[0,0].imshow(CHM[0])                # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html, date of access: 08-10-2024
        ax[0,1].imshow(DSM[0])
        ax[1,0].imshow(DTM[0])
        ax[1,1].imshow(DTM_interpolated[0])

    # Set titles
    title_font_size = 12
    ax[0,0].set_title('Canopy Height Model of WUR campus', fontsize=title_font_size)
    ax[0,1].set_title('DSM of WUR campus', fontsize=title_font_size)
    ax[1,0].set_title('DTM of WUR campus', fontsize=title_font_size)
    ax[1,1].set_title('Interpolated DTM of WUR campus', fontsize=title_font_size)
    
    # Create scalebars
    fig.colorbar(CHM_plot, ax=ax.ravel().tolist())
    
    # Set overall figure settings
    fig.suptitle('Raster data Wageningen visualisation', fontsize=16)
    fig.supxlabel('Distance x-direction (RD New) (m)', fontsize=14)
    fig.supylabel('Distance y-direction (RD New) (m)', fontsize=14)
    
    if save_plot:
        plt.savefig('output/WUR_campus_raster_data.png', dpi=300, bbox_inches='tight')
    
    plt.close()

def interpolate_raster(data):
    """
    Function that interpolates raster data based on IDW interpolation.

    Parameters
    ----------
    data : 3D numpy array (1 x n x m)
        Input data to interpolate.

    Returns
    -------
    interpolated_data : 3D numpy array (1 x n x m)
        Interpolated data

    """
    # Import necessary packages
    import numpy as np
    from rasterio.fill import fillnodata                                       # https://rasterio.readthedocs.io/en/latest/api/rasterio.fill.html
    # Fill no data applies a four direction conic search to every pixel to find values to interpolate using Inverse Distance Weighing
    # After the interpolation zero or more smoothing iterations (3x3 average filters on interpolated pixels) are applied to smooth artifacts
       
    if np.shape(data)[0] != 1 or len(np.shape(data)) != 3 or type(data) != np.ndarray:
        raise ValueError('Input data of numpy array type and should only contain one explicit 2D layer')
    
    # Create a mask to specify which pixels to fill (0=fill, 1=do not fill)
    mask = data.copy()
    mask[~np.isnan(data)] = 1
    mask[np.isnan(data)] = 0
    
    # Fill missing values
    interpolated_data = fillnodata(data, mask=mask)
    
    return interpolated_data

def write_CHM_as_tif(source, source_affine, CHM, output_path):
    """
    Functions that writes the input CHM away as a tif file

    Parameters
    ----------
    source : dictionary
        Dictionary with metadata on the input data
    source_affine : affine.Affine
        Affine data of the input data
    CHM : 3D numpy array (1 x n x m)
        CHM data.
    output_path : string
        output file path

    Returns
    -------
    None.

    """

    import rasterio
    import numpy as np
    import affine
      
    if type(source) != dict:
        raise ValueError('Please input the metadat as a dictionary')
        
    if type(source_affine) != affine.Affine:
        raise ValueError('Please input affine data')
           
    if np.shape(CHM)[0] != 1 or len(np.shape(CHM)) != 3 or type(CHM) != np.ndarray:
        raise ValueError('CHM input data of numpy array type and should only contain one explicit 2D layer')
        
    if type(output_path) != str:
        raise ValueError('Please input a string for "output_path"')

    # Taking dtm metadata for CHM
    kwargs = source
    
    # Extract information to change metadata for CHM
    depth, width, height = CHM.shape
    
    # Change kwargs to have correct metadata for CHM
    kwargs['transform'] = source_affine
    kwargs['width'] = width
    kwargs['height'] = height
    
    # Write CHM away as tif file
    with rasterio.open(output_path, 'w', **kwargs) as file:
        file.write(CHM.astype(rasterio.float32))
        
def vector_building(filename_path,bbox, underlying_layer,dtm):
    """
    function will load the geojson then plots it above the raster
    on the clipped bbox
    visualisation is created for inspection and overlays it on top of the CHM raster
    and also returns the polygons for next steps in the project
    sources: 
        https://www.earthdatascience.org/courses/use-data-open-source-python/intro-vector-data-python/vector-data-processing/clip-vector-data-in-python-geopandas-shapely/
        https://stackoverflow.com/questions/7569553/working-with-tiffs-import-export-in-python-using-numpy
    Parameters
    ----------
    filename : is the path to the geojson file where buildings are stored
    bbox : is taken as the extent argument
    underlying_layer: takes a numpy array that is plotted underneath the vector buildings
    dtm is the base dtm for the statistics for the array
    
    Returns
    polygons for the buildings
    gj : buildings geojson
    underlying_layer: is used in the follow up fucntion to calculate the heights
    plot of the raster and the polygons

    """
    #import eeded libraries
    import geojson
    import matplotlib.pyplot as plt
    import geopandas as gpd
    
    #load the geojson with the building polygons
    with open(filename_path) as f:
        gj = geojson.load(f)
    
    #set the coordinate system
    gj['crs'] = {'type': 'name', 'properties': {'name': 'EPSG:28992'}}
    
    #convert to the geodataframe
    gdf = gpd.GeoDataFrame.from_features(gj['features'])
    
    #set crs to wgs
    gdf.set_crs(epsg=4326, inplace=True)
    
    #buffered property we need as a polygon from the bbox layer
    bbox_polygon = bbox.buffered[0]
    bbox_polygon = gpd.GeoSeries([bbox_polygon], crs='EPSG:28992')

    #transforming the buffered polygon to the same CRS as gdf
    bbox_polygon = bbox_polygon.to_crs(gdf.crs)
    #clipping to the extent of bbox
    gj_clip = gpd.clip(gdf, bbox_polygon)
     
    # Set plot limits
    minx, miny, maxx, maxy = gj_clip.total_bounds
    
    # create a plot the figure then close it so it does not show
    fig, ax = plt.subplots(figsize=(9, 6))
    # Plot the CHM raster data
    CHM_plot = ax.imshow(underlying_layer[0], extent=[minx, maxx, miny, maxy])#cmap='viridis') 
    # adding the colorbar to the figure
    fig.colorbar(CHM_plot, ax=ax)
    
    gj_clip.plot(ax=ax, color='red', edgecolor='black', alpha=0.7)
    plt.title("Buildings in EPSG4326")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    
    plt.savefig('data/BuildingsVector.png')
    plt.close(fig)
    print("""=============================
Check loading vector data - Vector buildings layer created clipped to the defined bbox""")

    return gj_clip, underlying_layer

def get_heights(filename_path, bbox, CHM, dtm, dtm_meta):
    """
    Parameters
    ----------
    filename : is the path to the geojson vector file where buildings are stored
    bbox : is taken as the extent argument.
    CHM : takes a numpy array and calculates zonal statistics based on it
    dtm : dtm is the base dtm for the statistics for the array
     

    Returns
    -------
    None.
    Saves the geojson with attributes of the buildings including the mean height
    """
    import rasterio
    import rasterstats as rs
    import geopandas as gpd
    import numpy as np
    
    #initial check if the arguments to the function are of correct format
    if np.shape(CHM)[0] != 1 or len(np.shape(CHM)) != 3 or type(CHM) != np.ndarray:
        raise ValueError('CHM input data of numpy array type and should only contain one explicit 2D layer')
    
    if np.shape(dtm)[0] != 1 or len(np.shape(dtm)) != 3 or type(dtm) != np.ndarray:
         raise ValueError('dtm input data of numpy array type and should only contain one explicit 2D layer')
       
    if type(filename_path) != str:
        raise ValueError('Please input a string for "filename_path"')

    gj_clip, underlying_layer = vector_building(filename_path, bbox, CHM, dtm)
     #get attributes to the array from the dtm
    kwargs = dtm_meta

     #open the raster from a file and write to raster format
    with rasterio.open('data/CHM.tif', 'w', **kwargs) as file:
        file.write(underlying_layer.astype(rasterio.float32))
    
    #before conducting the zonal stats setting the crs to the same CRS of the CHM
    gj_clip = gj_clip.to_crs("28992")
    
    print(f"""=============================
Check calculating heights - Now the model will calculate the statistics for the buildings. Depending on the inputted bbox this might take some time""")
   
    chm_buildings = rs.zonal_stats(gj_clip, 'data/CHM.tif',prefix='CHM_',geojson_out=True)
    
    buildings_gdf = gpd.GeoDataFrame.from_features(chm_buildings)
    buildings_gdf = buildings_gdf.set_crs(epsg=28992)
    buildings_gdf.to_file('data/buildings_with_heights.geojson', driver="GeoJSON") 
    print(f"""-----------------------------
Check calculating heights - The model has successfully saved the calculated statistic into a separate geojson file""")
   
def crop_tiles(map_type, dirpath, bbox):
    
    # Inform user what the computer is doing during longer time of running
    print("Initiating function crop_tiles")
    
    # Load in necessary functions
    import rasterio
    from rasterio import mask
    from rasterio.merge import merge
    from rasterio.plot import show
    import glob
    import os
    import numpy as np
    from rasterio.io import MemoryFile
    
    #Directory path and search criteria for raster files
    search_criteria = f"{map_type}*.tif"
    
    #Inform user what file names the program will search for computation
    print(f"""=============================
Check 1 - File search criteria is {search_criteria}""")
    
    q = os.path.join(dirpath, search_criteria)
    
    #Get a list of raster file paths
    dem_fps = glob.glob(q)
    
    #This will hold the in-memory dataset objects
    cropped_datasets = []
    
    # Inform the user the directory for grabbing data has succesfully been located.
    print("""-----------------------------
Check 2 - Raw data directory locating successful""")
    
    print("""-----------------------------
Initiating numpy array cropping, depending on file size this may take a few minutes.""")
    
    for raster_file in dem_fps:
        try:
            # Open each raster file
            with rasterio.open(raster_file) as src:
                # Crop the raster using the polygon - 'bbox.buffered'
                out_image, out_transform = rasterio.mask.mask(src, bbox.buffered, crop=True, nodata=-9999)
        
                # Copy the metadata from the source raster
                out_meta = src.meta.copy()
        
                # Update the metadata to reflect the new dimensions, transform, and nodata value
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": -9999
                })
        
                # Create a MemoryFile to hold the cropped image
                memfile = MemoryFile()  # Keep the MemoryFile reference
                with memfile.open(**out_meta) as temp_dst:
                    temp_dst.write(out_image)
                    # Append the in-memory dataset after writing
                    cropped_datasets.append(memfile.open())
        except Exception as e:
            print(f" - Data from {raster_file} was not used: {e}")
    
    #Inform the user that the numpy array data was succesfully appended post crop
    print("""-----------------------------
Check 3 - appending of the cropped numpy arrays successful""")
    
    # Inform user that the data is about to be merged.
    
    print("""-----------------------------
Initiating data merging, depending on file size this may take a few minutes.""")
    
    # merge the datasets
    merged_image, merged_transform = merge(cropped_datasets)
    
    # Inform the user the raster merge was succesful
    print("""-----------------------------
Check 4 - Merge successful.""")
    
    # Update metadata for the merged raster
    out_meta = cropped_datasets[0].meta.copy()  # Copy metadata from one of the cropped rasters
    out_meta.update({
        "height": merged_image.shape[1],
        "width": merged_image.shape[2],
        "transform": merged_transform,
        "crs": cropped_datasets[0].crs  # Keep the CRS from the first dataset
    })
    
    # Inform the user that the final product has been given the appropriate meta data.
    print("""-----------------------------
Check 5 - Applying meta data was successful.""")
    
    # Save the merged raster to disk
    output_path = os.path.join("data/", f"merged_raster_{map_type}.tif")
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(merged_image)
    
    # Cleanup: Close the in-memory files explicitly if needed
    for dataset in cropped_datasets:
        dataset.close()
    return merged_image, out_meta, out_transform

def compare_heights(path_to_compare):
    """
    Parameters
    ----------
    path_to_compare : enter the path to the file (in the format of geojson)
        that you want compare agaist our calculations. 
        (this should include a column height to successfully compare against our model)
        

    Returns
    -------
    gdf_finalstats : geodataframe of the stats comparing calculations
        of out model compared to the inputed geojson

    """
    #import important libraries for this task
    import geopandas as gpd
    
    #load in the data for OSM buildings
    #initial check the input if is a geojson
    try:
        gdf_OfficialHeights = gpd.read_file(path_to_compare)
        
    except Exception as error:
        raise TypeError('Invalid GeoJSON file detected. Please input a valid GeoJSON file.') from error
    
    #load in the dataset created and calculatec for the building height from get_heights()
    gdf_OurHeights = gpd.read_file('data/buildings_with_heights.geojson')
    
    #creating the gdf where only osm buildings with height value are
    gdf_osm_non_nanheights = gdf_OfficialHeights.dropna(subset=['height'])
    
    #renaming the height from OSM and calculated CHM mean height into common column
    gdf_osm_non_nanheights = gdf_osm_non_nanheights.rename(columns={'height': 'compared_height'})
    gdf_OurHeights = gdf_OurHeights.rename(columns={'CHM_mean': 'compared_height'})
    
    #aligning two datasets basede on the common 'ID' value
    gdf_osm_non_nanheights['CHM_mean'] = gdf_osm_non_nanheights['@id'].map(gdf_OurHeights.set_index('@id')['compared_height'])
    
    #making sure that the  inputs are float before calculating the height difference
    gdf_osm_non_nanheights['compared_height'] = gdf_osm_non_nanheights['compared_height'].astype(float)
    gdf_osm_non_nanheights['CHM_mean'] = gdf_osm_non_nanheights['CHM_mean'].astype(float)

    #compare the heights between gdfs by the simple calculation
    gdf_osm_non_nanheights['height_difference'] = gdf_osm_non_nanheights['compared_height'] - gdf_osm_non_nanheights['CHM_mean']
    
    gdf_finalstats = gdf_osm_non_nanheights[['name','@id','height_difference', 'compared_height','CHM_mean','geometry']]
    gdf_finalstats = gdf_finalstats.rename(columns={'CHM_mean': 'Calculated_mean_height_from_CHM'})
    
    #take max chm from original dtset
    gdf_finalstats = gdf_finalstats.merge(gdf_OurHeights[['@id', 'CHM_max']],  # Select only the necessary columns
                                          on='@id',  
                                          how='left'
                                          )
    gdf_finalstats = gdf_finalstats.rename(columns={'CHM_max': 'Calculated_max_height_from_CHM'})
    gdf_finalstats = gdf_finalstats.rename(columns={'compared_height': 'OSM_height'})
    gdf_finalstats.to_file('data/comparing_heights.geojson', driver="GeoJSON") 
    gdf_finalstats.to_csv('data/comparing_heights.csv', index=False)
    print("Successfully saved a comparing_heights geojson and csv files with the height comparison in attribute table")
    return gdf_finalstats

def create_3D_html_comparing_to_OSM(input_path, output_path):
    """
    Function that takes the OSM buildings data and our calculated buildings heights.
    Creates a HTML which shows a 3D visualisation of the data.
    Source: https://towardsdatascience.com/visualizing-3d-spatial-data-with-pydeck-b7f58a936c67
    
    Parameters
    ----------
    output_path : string
        Path to where the output HTML should be placed.

    Returns
    -------
    Creates an HTML file with a 3D map of building information requested.

    """
    
    # Import necessary libraries
    import geopandas as gpd
    import pydeck as pdk   # https://deckgl.readthedocs.io/en/latest/

    if type(output_path) != str:
        raise TypeError('Please input a string as "output_path"')

    try:
        # Load the GeoJSON file into a GeoDataFrame
        buildings_data = gpd.read_file(input_path)
        
    except Exception as error: 
        raise TypeError('Invalid file type detected. Please input a valid GeoJSON file.') from error

    # Ensure the GeoDataFrame is in WGS84 (EPSG:4326) coordinate reference system
    buildings_data = buildings_data.to_crs(epsg=4326)
    
    # Create a Pydeck layer for the polygon visualization of the OSM height
    OSM_layer = pdk.Layer(
        'PolygonLayer',  # Layer type
        data=buildings_data,  # Input GeoDataFrame
        get_polygon='geometry.coordinates',  # Extract polygon coordinates
        extruded=True,  # Enable 3D extrusion
        get_elevation='OSM_height',  # Set elevation for extrusion
        get_fill_color=[0, 0, 255, 40],  # Color of the polygons (red)
        elevation_scale=1,  # Elevation scale (adjust for visualization)
    )

    # Create a Pydeck layer for the polygon visualization of the max height
    max_height_layer = pdk.Layer(
        'PolygonLayer',  # Layer type
        data=buildings_data,  # Input GeoDataFrame
        get_polygon='geometry.coordinates',  # Extract polygon coordinates
        extruded=True,  # Enable 3D extrusion
        get_elevation='Calculated_mean_height_from_CHM',  # Set elevation for extrusion
        get_fill_color=[255, 0, 0, 200],  # Color of the polygons (red)
        elevation_scale=1,  # Elevation scale (adjust for visualization)
    )
    
        # Define the initial view state (center of the map and zoom level)
    view_state = pdk.ViewState(
        latitude=51.9691868,  # Latitude for center of the view
        longitude=5.6653948,  # Longitude for center of the view
        zoom=13,  # Zoom level
        pitch=45,  # Tilt to give a better 3D perspective
    )

    # Create the deck.gl map with the defined layer and view state
    deck_map = pdk.Deck(layers=[max_height_layer, OSM_layer], initial_view_state=view_state)
    

    # Export the map to an HTML file
    html_content = deck_map.to_html(as_string=True)

    # Define the custom HTML legend to overlay
    legend_html = """
    <div style="position: absolute; top: 10px; left: 10px; z-index: 999; background-color: gray; padding: 10px; border-radius: 5px; font-family: Arial, sans-serif;">
        <h4>Legend</h4>
        <div><span style="background-color: blue; width: 20px; height: 20px; display: inline-block;"></span> OpenStreetMap</div>
        <div><span style="background-color: red; width: 20px; height: 20px; display: inline-block;"></span> Calculated max height</div>
    </div>
    """
        
    # Define the custom HTML title
    title_html = """
    <div style="position: absolute; top: 10px; right: 500px; width: 333px; text-align: center; z-index: 1000; font-family: Arial, sans-serif; font-size: 24px; font-weight: bold; color: black; background-color: gray; padding: 10px; border-radius: 8px;">
        3D height comparison between Open Street Map data and calculated heights for buildings in Wageningen 
    </div>
    """
    
    # Inject the legend HTML into the exported PyDeck HTML
    full_html = html_content.replace("<body>", f"<body>{legend_html}{title_html}")

    # Save the final HTML with the legend to visualise in browser
    with open(output_path, "w") as f:
        f.write(full_html)

def create_3D_html_Wageningen(input_path, output_path):
    """
    Function that takes our calculated buildings heights.
    Creates a HTML which shows a 3D visualisation of the mean and max heights of the data.
    Source: https://towardsdatascience.com/visualizing-3d-spatial-data-with-pydeck-b7f58a936c67
    
    Parameters
    ----------
    output_path : string
        Path to where the output HTML should be placed.

    Returns
    -------
    Creates an HTML file with a 3D map of building information requested.

    """
    
    # Import necessary libraries
    import geopandas as gpd
    import pydeck as pdk   # https://deckgl.readthedocs.io/en/latest/

    if type(output_path) != str:
        raise TypeError('Please input a string as "output_path"')

    try:
        # Load the GeoJSON file into a GeoDataFrame
        buildings_data = gpd.read_file(input_path)
        
    except Exception as error: 
        raise TypeError('Invalid file type detected. Please input a valid GeoJSON file.') from error

    # Ensure the GeoDataFrame is in WGS84 (EPSG:4326) coordinate reference system
    buildings_data = buildings_data.to_crs(epsg=4326)

    # Create a Pydeck layer for the polygon visualization
    mean_height_layer = pdk.Layer(
        'PolygonLayer',  # Layer type
        data=buildings_data,  # Input GeoDataFrame
        get_polygon='geometry.coordinates',  # Extract polygon coordinates
        extruded=True,  # Enable 3D extrusion
        get_elevation='CHM_mean',  # Set elevation for extrusion
        get_fill_color=[0, 255, 0, 120],  # Color of the polygons (red)
        elevation_scale=1,  # Elevation scale (adjust for visualization)
    )
    # Create a Pydeck layer for the polygon visualization
    max_height_layer = pdk.Layer(
        'PolygonLayer',  # Layer type
        data=buildings_data,  # Input GeoDataFrame
        get_polygon='geometry.coordinates',  # Extract polygon coordinates
        extruded=True,  # Enable 3D extrusion
        get_elevation='CHM_max',  # Set elevation for extrusion
        get_fill_color=[255, 0, 0, 200],  # Color of the polygons (red)
        elevation_scale=1,  # Elevation scale (adjust for visualization)
    )
    
        # Define the initial view state (center of the map and zoom level)
    view_state = pdk.ViewState(
        latitude=51.9691868,  # Latitude for center of the view
        longitude=5.6653948,  # Longitude for center of the view
        zoom=13,  # Zoom level
        pitch=45,  # Tilt to give a better 3D perspective
    )

    # Create the deck.gl map with the defined layer and view state
    deck_map = pdk.Deck(layers=[max_height_layer, mean_height_layer], initial_view_state=view_state)
    

    # Export the map to an HTML file
    html_content = deck_map.to_html(as_string=True)

    # Define the custom HTML legend to overlay
    legend_html = """
    <div style="position: absolute; top: 10px; left: 10px; z-index: 999; background-color: gray; padding: 10px; border-radius: 5px; font-family: Arial, sans-serif;">
        <h4>Legend</h4>
        <div><span style="background-color: green; width: 20px; height: 20px; display: inline-block;"></span> Calculated mean height</div>
        <div><span style="background-color: red; width: 20px; height: 20px; display: inline-block;"></span> Calculated max height</div>
    </div>
    """
        
    # Define the custom HTML title
    title_html = """
    <div style="position: absolute; top: 10px; right: 333px; width: 600px; text-align: center; z-index: 1000; font-family: Arial, sans-serif; font-size: 24px; font-weight: bold; color: black; background-color: gray; padding: 10px; border-radius: 8px;">
        Height 3D buildings Wageningen compared between max and mean height
    </div>
    """
    
    # Inject the legend HTML into the exported PyDeck HTML
    full_html = html_content.replace("<body>", f"<body>{legend_html}{title_html}")

    # Save the final HTML with the legend to visualise in browser
    with open(output_path, "w") as f:
        f.write(full_html)
        
def crop_building_vector (bbox_3D):
    """
    This function is meant to produce a small specified vector data set to which a 3d height comparison amongst OSM and our data can be made later on.
    Parameters
    ----------
    bbox_3D : geopandas geodataframe
        geopandas geodataframe containing a polygon geometry
        
    Writes a new vector data set to the disk
    -------

    """
    #import necessary packages
    import geopandas as gpd
    
    # Load in the dataset created and calculatec for the building height from get_heights()
    vector_data = gpd.read_file('data/buildings_with_heights.geojson')
    
    # Clip the data vector with the bbox 
    clipped_building_vector = vector_data.clip(bbox_3D.buffered)
    
    # Save the clipped building vector to the disk
    clipped_building_vector.to_file('data/specified_buildings.geojson', driver="GeoJSON")