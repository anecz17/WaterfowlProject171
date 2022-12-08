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
Two major outbreaks of Highly Pathogenic Avian Influenza (HPAI) in North America (2014-15 and 2022-current) have led to the depopulation of over 90 million commercial and backyard poultry with a total economic impact of over $2 billion dollars. These outbreaks and other similar ones around the world demonstrate the effects that Avian Influenza viruses (AIv) can have on domestic commercial poultry. Since waterfowl are the primary reservoir of AIv’s, understanding waterfowl distribution and movements relative to the location of poultry is an essential component of poultry biosecurity. The ability to identify waterfowl presence/absence and density in close proximity to the over 44,000 commercial poultry operations in the U.S. would offer farmers and state and federal stakeholders the ability to triage biosecurity and surveillance efforts. Organizations like Agrinerds use various sensing datasets by the government (USGS and CDFA), industry and from the national weather surveillance radar network (NEXRAD) to quantify and model waterfowl roosting density and distribution. The current approach- detection of waterfowl manually - has the potential to create a new layer/method of surveillance for the U.S. poultry industry. One significant challenge is the manual screening of historic NEXRAD radar imagery that is used to develop regional machine learning predictive models of waterfowl distributions. This approach is time consuming and poorly scalable. The ability to automate the radar screening would result in more robust and continuously improving models and vastly help address the issue of poulty biosecurity. **Through various models, we have trained models to automatate classifying waterfowl presence in NEXRAD radar imagary.**

## Figures
We can take a look at some figures to understand our problem a bit more. Firstly, let's take a look at a ```contaminated``` day. Below we are looking at 3 random images taking during the day and we will be exploring two variables. The first variable is ```correlation coefficient``` which represents how round the objects are- preciptation tends to be a lot more round than birds. Lastly, our second variable is ```reflectivity```, which helps us identify where objects are spatially.

<div align="center">

##### Contaminated:

![1](https://user-images.githubusercontent.com/84054117/202995723-4ba992d7-ce57-4dab-835d-03bcccff4ee3.png)
![2](https://user-images.githubusercontent.com/84054117/202995724-fde5109c-6773-4c06-b29f-69efd51e952c.png)
![3](https://user-images.githubusercontent.com/84054117/202995727-647ade41-302e-4afb-8687-f8b5a218b1d7.png)
![4](https://user-images.githubusercontent.com/84054117/202995728-5e00da31-0144-4ba5-b921-ec894e355d75.png)
![5](https://user-images.githubusercontent.com/84054117/202995882-71bd620b-3f4b-4ac8-a7f1-e2eb961adc46.png)
![6](https://user-images.githubusercontent.com/84054117/202995721-f5e9cf8c-d522-433e-8723-249361e81019.png)

</div>

<div align="center">

##### Uncontaminated:

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
For our models, we use two methods with a 20% test and 80% training split.
1. Thresholded Neural Network
The thresholded model is a Keras Sequential model with the following activation layers:
    ```
    thres_model.add(Dense(units = 4, activation = 'tanh', input_dim = X_train.shape[1]))
    thres_model.add(Dense(units = 9, activation = 'linear'))
    thres_model.add(Dense(units = 7, activation = 'relu'))
    thres_model.add(Dense(units = 1, activation = 'sigmoid'))
    ```
  
2. Convolution Neural Network

The CNN was also a Keras Sequential, but includes 2D Convolutional layers, Max Pooling layers, and Dropout layers. It takes a 180x180x2 array as an input for each observation, which represents a 2D image with two float elements for each location. The first represents Reflectivity and the second represents Correlation Coefficient.
```
conv_model = Sequential()
conv_model.add(Conv2D(8,(4,4),activation='relu',input_shape=(180,180,2)))
conv_model.add(Conv2D(16,(4,4),activation='relu', padding='same'))
conv_model.add(MaxPooling2D((4, 4)))
conv_model.add(Dropout(.3))
conv_model.add(Conv2D(32, (4,4), activation='relu', padding='same'))
conv_model.add(MaxPooling2D((4, 4)))
conv_model.add(Dropout(.3))
conv_model.add(Conv2D(16, (4,4), activation='relu', padding='same'))
conv_model.add(MaxPooling2D((4, 4)))
conv_model.add(Flatten())
conv_model.add(Dense(16, activation='linear'))
conv_model.add(Dense(1, activation='sigmoid'))

```


## Results
1. Thresholded Neural Network

This model used a 0.5 threshold and had the following classification report:

```
3/3 [==============================] - 0s 4ms/step
              precision    recall  f1-score   support

         0.0       0.79      0.95      0.87        44
         1.0       0.94      0.76      0.84        45

    accuracy                           0.85        89
   macro avg       0.87      0.86      0.85        89
weighted avg       0.87      0.85      0.85        89
```


2. Convolution Neural Network


We used a 0.5 threshold for the CNN as well:
```
              precision    recall  f1-score   support

         0.0       0.67      0.94      0.78        17
         1.0       0.97      0.81      0.89        43

    accuracy                           0.85        60
   macro avg       0.82      0.88      0.83        60
weighted avg       0.89      0.85      0.86        60

```
Both created promising results. Currently, the thresholded neutral network had a higher precision, 87%, while the CNN had a precision of 82%. Full details of these executions are present in the _Preprocessing & Model Building_ Jupyter Notebook.


## Discussion
Our model hit many roadblocks during the process we termed "parsing", where we took NEXRAD data and created text files listing all correlation coefficient, reflectivety,and velocity values. Midway through working on our machine learning models, we noticed strange behavior and formatting of certain files, that resulted in the model being more inaccurate than it was. Finetuning our parser was a relavent issue even up until our last final models. This is unfortunate, as it could have allowed more time to create a accurate model.

One of the main limitations of this current model is that it was composed of around 450 datapoints and not a large set. The size limitations were immense, as this was the culmination of working with over 50 GBs of NEXRAD files. It could be revealed our model isn't reliable with a higher dataset. However, given more time and more CPU power, the tools and tweaks we applied to creating our convolution network would likely still work with this a higher dataset. We feel we created a strong starting point for creating a stable tool to identify waterfowl in NEXRAD imagery. However, this also suggests it would be a haste decision to deduce that the Thresholded Neural Network is more viable than the Convolution Neural Network because more data could change this.

The idea behind the convolutional neural net was first to process the image data directly, rather than manually determining heuristics which we believed would have a high correlation with the classification. This allows the model to learn patterns which we potentially would not recognize on our own. The convolutional neural net also is optimized for finding characteristics of images, making it a logical choice for our project. Our first issue was that rather than having a single image for each day, we had a range from 19 to over 30 images per day, for both of Reflectivity and Correlation Coefficient. While we could pass multiple images into a convolutional neural net, the size of the data is a significant issue, so we decided to take the average pixel values over the sets of images to produce a single image for Reflectivity and a single image for Correlation Coefficient. We used Conv2D layers with relu activation as the basis of the model, because this prevents the amount of computation required from growing exponentially with the model's size. We added Dropout layers to reduce overfitting, and Max Pooling layers to select the most important features from previous layers, reducing computational complexity. We tried various amounts of nodes at each layer, and selected values which reduced overfitting. 

The result of 85% accuracy for the CNN was less than expected considering the images are usually very different for contaminated and non-contaminated cases. This is likely due to both the reduction in image size eliminating some features of the images, and the averaging of many images reducing the contrast of the images. To achieve a higher accuracy for this model, we could attempt to use multiple averaged images for each day, so that each represents less total time during the day, which should preserve features better. The drawback of this approach is heavily increased computational complexity, meaning we likely would need to run the model on fewer total days.

It is possible different methods of preprocessing the NEXRAD imagery into 2D arrays could be used, which would reduce the memory constraits of the dataset and subsequently make it easier to use larger sets data when training the models. Each NEXRAD image also has many fields, most of which we ommitted due to our memory-constraits. It would be interesting to see in further models whether these could contributes to higher classification accuracy if a way to efficiently convey this data is developed. 

Overall, it was satisfying to create two models that show some ability to accurately predict whether a reading has waterfowl or not. This project was somewhat out of the scope of the processing power and memory we had availible without divesting monetary resources into this assignment. With using external computing power and memory to preprocess the NEXRAD imagery, it would have been easier to create models. Without it, our group had to delegate small sets of data to each member to process and then upload to a Drive. This was highly inefficient and gave way to a higher likelihood of human errer, such as forgetting files and uploading them to the wrong station folders. 


## Conclusion

Given imagery that is soley composed of waterfowl movement, further models can be developed to analyze their behavior, such as migratory patterns. A further discovery could be attempting to create models that would help answer questions about poultry biosecurity. This would turn our model into a form of preprocessy for further machine learning algorithms.


## Collaboration

<div align="center">

| Andras Necz | Daria Buka | Mitchell Davis | Colton Perazzo | Zachary Oren | Jonathan Wesely |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Leader | Coder & Writer | Coder | Coder & Writer | Coder | Coder |

</div>

- ```Andras Necz``` <br />
    - Team leader
    - Organized meeting with Agrinerds
    - Worked on most scripts

- ```Daria Buka``` <br />
    - Preprocessed contamination .xlsx sheets
    - Trained first model
    - Main contributor of writeup

- ```Mitchell Davis``` <br />
    - Main contributor to parser
    - Parsed majority of data

- ```Colton Perazzo``` <br />
    - Main contributor of writeup
    - Parsed data

- ```Zachary Oren``` <br />
    - Main contributor to parser
    - Main contributor of convolution neural network

- ```Jonathan Wesely``` <br />
    - Main contributor of convolution neural network
