# Waterfowl Project - ECS 171
Group Project for ECS 171

*Perform the data exploration step (i.e. evaluate your data, # of observations, details about your data distributions, scales, missing data, column descriptions) Note: For image data you can still describe your data by the number of classes, # of images, size of images, are sizes standardized? do they need to be cropped? normalized? etc.*

Number of images:

Size of images:

Are sizes standardized: Yes! We are working with NEXRAD Level II data.
*Plot your data. For tabular data, you will need to run scatters, for image data, you will need to plot your example classes.*

*How will you preprocess your data? You should explain this in your Readme.MD file and link your jupyter notebook to it. Your jupyter notebook should be uploaded to your repo.*

The largest limitation currently is the massive amount of datapoints we are dealing with per day of data. Each day has 20 images associated with it and only 1 'status' classification per day. We will have to not only blur and crop the images (as most points outside a certain radius are just 0.

*Jupyter Notebook data download and environment setup requirements: !wget !unzip like functions as well as !pip install functions for non standard libraries not available in colab are required to be in the top section of your jupyter lab notebook. Please see HW2 & HW3 for examples.*

https://colab.research.google.com/drive/16n72hFmsJis-llT24E0w4_E6ShFTNLNy?usp=sharing

