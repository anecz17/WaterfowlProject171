# Waterfowl Project - ECS 171
Group Project for ECS 171

**OBJECTIVE**
Create a machine learning model that takes a day of NEXRAD readings and correctly classifies the day as contaminated or not.

**DATA EXPLORATION**

Number of images:
There are 20 radar stations, each has at least 150 screened days and each day contains 20 files. These files could be seen as images, but they only contain float values connected to the pixels of the image in a 2D array.

Size of images:
The size of an image is 720x1192.

Putting size and the number of images together, we get that the total dataset we work with contains 51 Billion data points (20 * 150 * 720 * 1192).

Are sizes standardized: Yes. We are pulling image information from the NEXRAD Level II files, so all image sizes are standarized.

**PLOTTING DATA**
Plotting images is something we're still working through because it is radar. However, examples below illustrate the important differences between the radar:


The field associated with the images is reflectivity, which refers to the information given by the reflection of waves back to the radar.

All other fields associated with NEXRAD: cross_correlation_ratio, spectrum_width, differential_phase, velocity, differential_reflectivity

cross_correlation_ratio: This describes the roundness of an objects. Great tool to determine whether a data point is participation or not, since the greater this value, the more probable it is participation. (participation is usually small and droplets are round-like) 

spectrum_width: distribution of velocities within a single radar pixel



**ADDRESSING PREPROCESSING**

The largest limitation currently is the massive amount of datapoints we are dealing with per day of data. Each day has 20 images associated with it and only one 'status' classification. We will have to not only blur and crop the images (as most points outside a certain radius are just 0), but likely find an additional method to reduce how many points are being processed, such as selecting a only subset of days out of the entire dataset to train our model with. To avoid bias such as differences in weather due to seasons, we would have to randomize this.

*Jupyter Notebook data download and environment setup requirements: !wget !unzip like functions as well as !pip install functions for non standard libraries not available in colab are required to be in the top section of your jupyter lab notebook. Please see HW2 & HW3 for examples.*

**GOOGLE COLAB**
https://colab.research.google.com/drive/16n72hFmsJis-llT24E0w4_E6ShFTNLNy?usp=sharing

