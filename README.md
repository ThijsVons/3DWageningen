# Geoscripting project repository of Wodka Mole of Improvement

- Title: Assessing building heights in Wageningen
- Team name and members: wodka_mole_of_improvement. Members: Engelsma, Naut & Visser, Rianne & Bartek, Michal & Vons, Thijs
- Challenge number: 7
- Sources are cited locally

## Description how to run/reproduce: 
If you come from codegrade, please go to the Wodka Mole of Improvement Geoscripting teams to download all_files_Wodka_Mole_of_Improvement_project.zip followed by extracting it. 

Once you are here you have downloaded all_files_Wodka_Mole_of_Improvement_project.zip, and extracted the file. Well done!

### Directory structure:
Below the directory structure is shown.

- all_files_Wodka_Mole_of_Improvement
    - Project_Starter-wodka_mole_of_improvement
        - main.py
        - Python
            - __init__.py
        - data (added when running main.py)
        - output (added when running main.py)
        - Building_Env.yaml
        - LICENSE
        - README.md
    - Raw_data_project
        - tiff files (raw DTM and DSM data)
        - geojson file (OSM data)

### Instructions before running main.py:
In the terminal make sure your active directory is 'Project_Starter-wodka_mole_of_improvement'.

Install 'Building_Env.yaml' to create an environment. Do this by running: 'mamba env create -f Building_Env.yaml.

Run 'source activate Wodka_Mole_Project' to activate the environment.

Run 'spyder' to open spyder.

Now within spyder, open and run main.py and enjoy the results! Ensure that the specified area is within the greater area of Wageningen. The script might take a couple of minutes to run.

## Outputs:
The first output (Wageningen_buildings_comparison.geojson) is a HTML file which shows a comparison between, buildings with height attributes from OpenStreetMap, and the same buildings with the height calculated in our algorithm.

The second output (Wageningen_buildings_visualisation.geojson) is an area in Wageningen with the mean and max height of multiple buildings calculated in our algorithm.

The third output (WUR_campus_raster_data.png) is a png with information on the raster data (not the full area due to file size issues).
