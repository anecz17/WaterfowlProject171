<div align="center">

![image](https://i.ibb.co/PFLxfVj/image-1.png)

# ECS 171 Waterfowl Project
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/Naereen/badges)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16n72hFmsJis-llT24E0w4_E6ShFTNLNy?usp=sharing)

Created for *ECS 171 - Machine Learning* <br />
Professor Edwin Solares <br />
University of California, Davis

Objective: create a machine learning model that takes a day of NEXRAD readings and correctly classifies the day as contaminated or not.

Keywords: *waterfowl*, *machine learning*, *nexrad*, *convolution*

[Introduction](#introduction) •
[Figures](#figures) •
[Methods](#methods) •
[Results](#results) •
[Discussion](#discussion) •
[Conclusion](#conclusion) •
[Collaboration](#collaboration)

</div>

## Introduction
Two major outbreaks of Highly Pathogenic Avian Influenza (HPAI) in North America (2014-15 and 2022-current) have led to the depopulation of over 90 million commercial and backyard poultry at over 650 premises in 42 states with a total economic impact of over $2 billion dollars and counting. These outbreaks and other similar ones around the world demonstrate the effects that Avian Influenza viruses (AIv) can have on domestic commercial poultry. Since waterfowl are the primary reservoir of AIv’s, understanding waterfowl distribution and movements relative to the location of poultry is an essential component of poultry biosecurity. The ability to identify waterfowl presence/absence and density in close proximity to the over 44,000 commercial poultry operations in the U.S. would offer farmers and state and federal stakeholders the ability to triage biosecurity and surveillance efforts. We use various remote sensing datasets by the government (USGS and CDFA), industry (California Poultry Federation and the Pacific Egg and Poultry Association) and from the national weather surveillance radar network (NEXRAD) to quantify and model waterfowl roosting density and distribution. The current approach- detection of waterfowl manually - has the potential to create a new layer/method of surveillance for the U.S. poultry industry. One significant challenge is the manual screening of historic NEXRAD radar imagery that is used to develop regional machine learning predictive models of waterfowl distributions. This approach is time consuming and poorly scalable. The ability to automate the radar screening would results in more robust and continuously improving models, thus our reasoning behind the creation of this model.

## Figures

## Methods

### Data Exploration

In our model, our data comes from two sources:
1. NEXRAD Level II Files
2. Containment Spreadsheets

#### NEXRAD Level II Files
The NEXRAD Level II Files (National Weather Surveillance Radar Network) are stored on an Amazon Web Server found [here](https://s3.amazonaws.com/noaa-nexrad-level2/index.html). These data files are essentially data from radars that send out EMR waves at various angles. With this in mind, these radars measure how much of the signal is refelected off of objects in the sky (such as biological life and precipitation) and then sends all of the raw data stream processed into various data types. 

There are 20 radar stations provided by the NEXRAD platform. Each of these stations has at least 150 screened days, where each day can contain up to 250 data files. Furthermore, these radar stations take a snapshot of the local environment every 10 minutes. In this project, we are generally working with around 20 files a day as we are only interested in the 3.5 hour (~200 minute) window around sunset. This has a distinct purpose due to the biological habit of waterfowl going out for their nightly feeds after sunset.

Using a random selection of stations, we selected the following stations for our model: ```ABR```, ```IND```, ```JKL```, ```DHL```, and ```LVX```. In addition, the NEXRAD files date back to every year since 1970. However in our model, we will only be taking into account the years of ```2019```, ```2020```, ```2021``` and ```2022```.

Every NEXRAD file being used has a total of ```72``` column, however we are only interested in a small subset of columns for our model. Every column has ```720x1192``` data points where the sizes are standardized. We are interpertating columns as images and data points as pixels. Therefore, the total amount of data we have the ability to work with is ``` 20 x 150 x 20 x 3 x 720 x 1192``` = ~150 billion primitve values. 

We are planning to work with the following data columns: ```correlation coefficient```, ```reflectivity```, and ```velocity```.

• ```correlation coefficient``` - Roundness of the object, values closest to 1 are spheres which most likely is percipation (or bugs depending on the time of day, season and radar location). A value around 0.8 is usually biological, or waterfowl in our usecase.

• ```reflectivity``` - Intensity of the EMR wave that is refelected back to the radar. The stronger the reflected signal- the more total objects that are in the scanned region.

• ```velocity``` - Speed of the total objects in the scanned region.

#### Containment Spreadsheets
Another source of data used in this model is containment information stored in excel spreadsheets. These spreadsheets include the evaluation from scientists at [Agrinerds](https://www.agrinerds.com/) on wheter or not the data is contaminated or not. Moreover, these spreadsheets contain various columns: ```RADAR```, ```DATE```, ```SEASON```, ```DOWNLOAD```, ```STATUS```, ```CONTAMINATION_TYPE```, ```TARGET_ID```, ```SCREENER```, ```SURFACE_WIND```, ```WIND_DIRECTION```, ```APPROXIMATE_SAMPLING_TIME```, ```TARGET_SPEED```, ```GROUND_HEADING```, ```COMMENTS```.

We are interested in the ```STATUS``` column which specifies if the specific day is containtment or not. This is implied by either a ```B``` or ```C``` value found in this column. ```B``` stands for **birds** (not containment) and ```C``` stands for containment (self-explanatory).

### Preprocessing
Due to the large amount of data in the NEXRAD files, we took a computer/cloud approach to store these files for preprocessing. In order to download the massively RAW archived NEXRAD files from our selected station, we used an R script provided by [Agrinerds](https://www.agrinerds.com/) to batch download data. The provided R scripts can be found within the repo [here](downloadnexrad_folders_original.R).

The above R script has a few customizable settings in which we edited to download the correct data we wanted.
```R
radar<-"station_name"
screening<-"station_screening.xls"
nexrad<-read.csv("station_path") 

outpath<-"saving_path" 
```

• ```radar``` - Station name. (eg: ```LVX```) <br />
• ```screening``` - Name of the contatinment spreadsheet for the selected station. (eg: ```KLVX_allscreening.xls```) <br />
• ```nexrad``` - Path of the CSV file that includes data about all NEXRAD stations. (found [here](nexrad_site_list_with_utm.csv) in repo) <br />
• ```outpath``` - Path to save all of the RAW archived data. 

(finish talking about folder and file structure, prprocessing and how we parsed data)

### Modeling

## Results

## Discussion

## Conclusion

## Collaboration

<div align="center">

| Andras Necz | Daria Buka | Mitchell Davis | Colton Perazzo | Zachary Oren | Jonathan Wesely |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Leader | Coder & Writer | Coder | Coder & Writer | Coder | Coder |

</div>
