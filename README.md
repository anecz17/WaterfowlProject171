# Waterfowl Project - ECS 171
Group Project for ECS 171

**OBJECTIVE**

Create a machine learning model that takes a day of NEXRAD readings and correctly classifies the day as contaminated or not.

**DATA EXPLORATION**

****The whole data of the project has 2 origins.****

The main origin of data is the National Weather Surveillance Radar Networkis = "NEXRAD" Level II files. These files stored on AWS and all of them has to be pulled thru an API.

The other origin is the evaluatiion of scientist whether the data is contaminated or not. This information is stored in excel spreadsheets.

****Size of raw data:****

Images are pulled from the NEXRAD Level II files, so all image sizes are standarized.
    
There are 20 radar stations, each has at least 150 screened days and each day contains 250 files. Approximately, the radar stations take a snapshot of the environment every 10 minutes. We generally work with 20 files a day, since we are only interested in the ~200 minute (~3.5 hour) window around sunset. This has biological background - the birds go out for their night feeds after sunset.
    
Every file has a total of 72 columns. Every column has 720x1192 data points - sizes are standardized. Columns could be interpeted as images and data points as pixels. But we are only interested in a couple in the order of 3-7. (Discussion is still going on with the biologists to understand which variables indicate bird movement and which variables indicate other origins.)
    
Therefore the total amount of data we work with: 20 x 150 x 20 x 3 x 720 x 1192 = ~150 Billion

  As we can see the data size is huge. These values are not normalized, and their distribution is unknown. We will sort out the missing values. We have other excel files that contain y values, basically whether the day was contaminated by other movement or not. The data is not normalized, it has whatever value it captured, but if we would look at all the data in the columns we could generally set a range.

The columns we planning to work with are correlation coeffitient, reflectivity and velocity. (More discussion is needed and we may add a couple more to better sort out special cases/anomalies.)

Correlation Coeffitient - cross_correlation_ratio - describes the roundness of an objects. Great tool to determine whether a data point is participation or not, since the greater this value, the more probable it is participation. (participation is small and droplets are round-shaped)

Reflectivity - differential_reflectivity - describes what waves come back in the visible light range. Great for detecting objects.

Velocity - velocity - describes the horizontal speed of the object. Also: - spectrum_width - distribution of velocities within a single radar pixel.

****Plotting:****

The files contain 72 column, each of them is a 2D array of floats. We can convert the floats to colors and display them in Unidata IVD.


**ADDRESSING PREPROCESSING**

The largest limitation currently is the massive amount of datapoints we are dealing with per day. Each day has 20 images associated with (51K values even if we only use 3 columns) and only a single 'status' classification. We will have to not only crop the images, but likely find an additional method to reduce how many points are being processed. Data outside of 150 km radius of the radar cannot show birds, and if it is contaminated outside of 250km, the scientist didn't mark it as contaminated. Those parts of the images are cropped.

We are still considering multiple ways to reduce the data size. One of the most promising ideas we have is we would do a sampling of the images and set an integer based on the sample. For example: we would look for high values of roundness and if that reaches a certain treshold in the data, we would assume it contains participation and set a certain value for the image. Add the images values together from the different timestamped images from the same day and feed that single integer to the Neural Network. That single integer may be standardized. This model could nicely work out if the anomalies are distinct and we find the right thresolds.

*Jupyter Notebook data download and environment setup requirements: !wget !unzip like functions as well as !pip install functions for non standard libraries not available in colab are required to be in the top section of your jupyter lab notebook. Please see HW2 & HW3 for examples.*

**GOOGLE COLAB**

https://colab.research.google.com/drive/16n72hFmsJis-llT24E0w4_E6ShFTNLNy?usp=sharing

