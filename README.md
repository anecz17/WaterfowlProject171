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
We can take a look at some figures to understand our problem a bit more. Firstly, let's take a look at a ```contaminated``` day. Below we are looking at 3 random images taking during the day and we will be exploring two variables. The first variable is ```correlation coefficient``` which represents how round the objects are- preciptation tends to be a lot more round than birds. Lastly, our second variable is ```reflectivity```, which helps us identify where objects are spatially.

##### Contaminated:
<div align="center">

![1](https://user-images.githubusercontent.com/84054117/202995723-4ba992d7-ce57-4dab-835d-03bcccff4ee3.png)
![2](https://user-images.githubusercontent.com/84054117/202995724-fde5109c-6773-4c06-b29f-69efd51e952c.png)
![3](https://user-images.githubusercontent.com/84054117/202995727-647ade41-302e-4afb-8687-f8b5a218b1d7.png)
![4](https://user-images.githubusercontent.com/84054117/202995728-5e00da31-0144-4ba5-b921-ec894e355d75.png)
![5](https://user-images.githubusercontent.com/84054117/202995882-71bd620b-3f4b-4ac8-a7f1-e2eb961adc46.png)
![6](https://user-images.githubusercontent.com/84054117/202995721-f5e9cf8c-d522-433e-8723-249361e81019.png)

</div>

##### Uncontaminated:
<div align="center">

![7](https://user-images.githubusercontent.com/84054117/203115483-f6d8ec46-403f-4c55-aae1-e3ca20d6c711.png)
![8](https://user-images.githubusercontent.com/84054117/203115486-7cb3e909-0873-48cc-b72e-1abc59067609.png)
![10](https://user-images.githubusercontent.com/84054117/203115488-7d7343de-7c4c-46e0-8b3c-105293f47b12.png)
![9](https://user-images.githubusercontent.com/84054117/203115487-d862714e-26b9-4649-88da-cea4d8b279f6.png)
![12](https://user-images.githubusercontent.com/84054117/203115481-d2affd1b-d1ba-42c7-a536-237032c5f7bf.png)
![11](https://user-images.githubusercontent.com/84054117/203115490-11087c64-c042-4cec-8668-38ccf6b0202c.png)

</div>

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

After downloading these RAW archived data files they are initally in a ```java``` data format in which some of the data values within these files could not
be used in a python enviroment. In order to get around this, we used the package [Metpy](https://unidata.github.io/MetPy/latest/index.html) to read these
NEXRAD weather data files and parse their contents into a readable and useable text file for each day.

```python
def extract_nexrad_level2_data(nexrad_level2_file, sweep=0):
    f = Level2File(nexrad_level2_file)

    az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])

    ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
    ref_range = np.arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate
    ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])

    rho_hdr = f.sweeps[sweep][0][4][b'RHO'][0]
    rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
    rho = np.array([ray[4][b'RHO'][1] for ray in f.sweeps[sweep]])

    phi_hdr = f.sweeps[sweep][0][4][b'PHI'][0]
    phi_range = (np.arange(phi_hdr.num_gates + 1) - 0.5) * phi_hdr.gate_width + phi_hdr.first_gate
    phi = np.array([ray[4][b'PHI'][1] for ray in f.sweeps[sweep]])

    zdr_hdr = f.sweeps[sweep][0][4][b'ZDR'][0]
    zdr_range = (np.arange(zdr_hdr.num_gates + 1) - 0.5) * zdr_hdr.gate_width + zdr_hdr.first_gate
    zdr = np.array([ray[4][b'ZDR'][1] for ray in f.sweeps[sweep]])
    nexrad_level2_data = {"timestamp": f.dt}

    for var_data, var_range, lbl in zip((ref, rho, zdr, phi),
                                        (ref_range, rho_range, zdr_range, phi_range),
                                        ('REF (dBZ)', 'RHO', 'ZDR (dBZ)', 'PHI')):
        # Turn into an array, then mask
        data = np.ma.array(var_data)
        data[np.isnan(data)] = np.ma.masked

        # Convert az,range to x,y
        xlocs = var_range * np.sin(np.deg2rad(az[:, np.newaxis]))
        ylocs = var_range * np.cos(np.deg2rad(az[:, np.newaxis]))
        nexrad_level2_data[lbl] = {"x_data": xlocs, "y_data": ylocs, "z_data": var_data}

    return nexrad_level2_data
```

Our function ```extract_nexrad_level2_data()``` takes an NEXRAD archive file from the amazon web server, parses through the data, and takes
out the neccessary dataset for trainging our model. As stated above, these data values we used are ```correlation coefficient```, ```reflectivity```, and ```velocity```.

After this function is ran, it outputs a ```.txt``` file which can then be used as data in our model.
An example output (very shortened down, average size of data is ```~15,000kb``` per day) is as follows ```R = reflectivity, C = correlation coefficient, Z = velocity```:

```
R
-13.75,-8.270833333333334,-16.104166666666668,-10.333333333333334,-4.520833333333333,4.583333333333333,4.5,0.22916666666666666,-1.8125,-2.1458333333333335,-1.0208333333333333,-0.4791666666666667,0.0,0.0,-1.3333333333333333,0.0,-0.6041666666666666,0.0,0.0,0.0,-0.5208333333333334,-2.2083333333333335,-0.9791666666666666,-2.9375,-1.8125,0.0,-0.75,-0.625,-2.625,-1.0208333333333333,2.3333333333333335,-1.625,-0.6041666666666666,-0.22916666666666666,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, -9.875,-6.395833333333333,-18.395833333333332,-13.270833333333334,-8.479166666666666,-4.291666666666667,-4.791666666666667,-0.3541666666666667,1.2916666666666667,0.25,-2.3958333333333335,-3.4583333333333335,-1.2708333333333333,0.0,-0.2708333333333333,-0.2916666666666667,0.0,0.0,0.0,0.0,0.0,-0.6458333333333334,0.0,0.0,-0.4583333333333333
C
0.38927083333333334,0.6102083333333334,0.3027083333333333,0.47802083333333334,0.7769791666666666,0.2710416666666666,0.5951041666666668,0.845625,0.7548958333333333,0.6781250000000002,0.5610416666666668,0.1415625,0.10708333333333334,0.33416666666666667,0.08020833333333333,0.28385416666666663,0.1823958333333333,0.1515625,0.0,0.0,0.0,0.16260416666666666,0.0,0.0,0.0196875,0.0,0.0,0.0,0.0,0.0,0.0,0.23083333333333336,0.11354166666666668,0.06572916666666667,0.16864583333333333,0.35500000000000004,0.1890625,0.10854166666666668,0.0,0.0,0.15031250000000002,0.17635416666666665,0.4995833333333333,0.27822916666666664,0.4273958333333333,0.6521874999999998,0.5916666666666666,0.3665625,0.0,0.15135416666666665,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
0.4242708333333333,0.50625,0.42104166666666665,0.7726041666666668,0.7657291666666666,0.49125,0.6349999999999999,0.20322916666666668,0.49166666666666664,0.3463541666666667,0.33458333333333334,0.4170833333333333,0.4741666666666667,0.4675,0.0,0.1575,0.2778125,0.0903125,0.10604166666666667,0.0,0.0,0.0,0.12333333333333332,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.12375,0.0,0.0,0.0,0.0,0.0,0.12666666666666668,
Z
-0.53125,0.9921875,-0.22265625,-1.65625,1.2265625,1.58984375,-1.11328125,0.171875,-1.44921875,0.921875,-3.33984375,-0.3359375,0.02734375,-0.14453125,-1.0078125,-2.03125,-1.9921875,1.3671875,0.0,0.0,0.0,-0.8515625,0.0,0.0,0.01171875,0.0,0.0,0.0,0.0,0.0,0.0,0.46875,0.56640625,-0.42578125,0.92578125,2.20703125,-0.15234375,0.0859375,0.0,0.0,0.625,0.4296875,0.72265625,0.265625,0.63671875,2.00390625,-2.453125,-1.640625,0.0,1.234375,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
```

In total after being parsed into a useable format we have around ```~30GB``` of data we can use to train our model. Atlast the final step in our preprocessing is getting the data values from these text files into a data structure that can be used within our model. 

```python

```

### Modeling

For our model, we use two different methods:
1. Threshold
2. Convolution

## Results

## Discussion

## Conclusion

## Collaboration

<div align="center">

| Andras Necz | Daria Buka | Mitchell Davis | Colton Perazzo | Zachary Oren | Jonathan Wesely |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Leader | Coder & Writer | Coder | Coder & Writer | Coder | Coder |

</div>

- ```Andras Necz``` <br />
    - Team leader and contributed to the organization of the project idea and overall project.

- ```Daria Buka``` <br />
    - test

- ```Mitchell Davis``` <br />
    - test

- ```Colton Perazzo``` <br />
    - test

- ```Zachary Oren``` <br />
    - test

- ```Johnathan Wesely``` <br />
    - test
