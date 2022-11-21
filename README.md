# Waterfowl Project - ECS 171
Group Project for ECS 171

**OBJECTIVE**
Create a machine learning model that takes a day of NEXRAD readings and correctly classifies the day as contaminated or not.


*Perform the data exploration step (i.e. evaluate your data, # of observations, details about your data distributions, scales, missing data, column descriptions) Note: For image data you can still describe your data by the number of classes, # of images, size of images, are sizes standardized? do they need to be cropped? normalized? etc.*

Number of images:

Size of images:

Are sizes standardized: Yes! We are working with NEXRAD Level II data.

*Plot your data. For tabular data, you will need to run scatters, for image data, you will need to plot your example classes.*

Classes plotted on COLAB: cross_correlation_ratio, spectrum_width, differential_phase, velocity, differential_reflectivity, reflectivity

cross_correlation_ratio: 

spectrum_width:

differential_phase:

velocity:

differential_reflectivity:

reflectivity:

**ADDRESSING PREPROCESSING**

The largest limitation currently is the massive amount of datapoints we are dealing with per day of data. Each day has 20 images associated with it and only one 'status' classification. We will have to not only blur and crop the images (as most points outside a certain radius are just 0), but likely find an additional method to reduce how many points are being processed, such as selecting a only subset of days out of the entire dataset to train our model with. To avoid bias such as differences in weather due to seasons, we would have to randomize this.

*Jupyter Notebook data download and environment setup requirements: !wget !unzip like functions as well as !pip install functions for non standard libraries not available in colab are required to be in the top section of your jupyter lab notebook. Please see HW2 & HW3 for examples.*

https://colab.research.google.com/drive/16n72hFmsJis-llT24E0w4_E6ShFTNLNy?usp=sharing

