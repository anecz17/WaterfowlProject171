# Waterfowl Project - ECS 171
Group Project for ECS 171

**OBJECTIVE**

Create a machine learning model that takes a day of NEXRAD readings and correctly classifies the day as contaminated or not.

**DATA EXPLORATION**

****The whole data has 2 origins:****

The main origin of data is the National Weather Surveillance Radar Networkis = "NEXRAD" Level II files. These files stored on AWS and all of them has to be pulled thru an API. 
_The way that the technology works is that these radars are sending out EMR waves at various angles and are measuring how much of the signal is reflected off of objects in the sky such as biological life and precipitation. There is essentially one raw data stream that gets processed into the various data types._

The other origin is the evaluation of scientist whether the data is contaminated or not. This information is stored in excel spreadsheets.

****Spreadsheets****

The spreadsheets contain information that are not in the NEXRAD files. For example when sunset happened, and based on that we will pull data from the AWS site.

Moreover, the spreadsheet contains the following columns:
RADAR,	DATE,	SEASON,	DOWNLOAD,	STATUS,	CONTAMINATION_TYPE,	TARGET_ID,	SCREENER,	SURFACE_WIND,	WIND_DIRECTION,	APPROXIMATE_SAMPLING_TIME,	TARGET_SPEED,	GROUND_HEADING,	COMMENTS -- We are interested in status, the date to evaluate the day based on status.
 

****NEXRAD -- Size of raw data:****

Images are pulled from the NEXRAD Level II files, so all image sizes are standarized.
    
There are 20 radar stations, each has at least 150 screened days and each day contains 250 files. Approximately, the radar stations take a snapshot of the environment every 10 minutes. We generally work with 20 files a day, since we are only interested in the ~200 minute (~3.5 hour) window around sunset. This has biological background - the birds go out for their night feeds after sunset.
    
Every file has a total of 72 columns. Every column has 720x1192 data points - sizes are standardized. Columns could be interpeted as images and data points as pixels. But we are only interested in a couple in the order of 3-7. (Discussion is still going on with the biologists to understand which variables indicate bird movement and which variables indicate other origins.)
    
Therefore the total amount of data we work with: 20 x 150 x 20 x 3 x 720 x 1192 = ~150 Billion primitive values.

  As we can see the data size is huge. These values are not normalized, and their distribution is unknown. We will sort out the missing values. We have other excel files that contain y values, basically whether the day was contaminated by other movement or not. The data is not normalized, it has whatever value it captured, but if we would look at all the data in the columns we could generally set a range.

The columns we planning to work with are correlation coeffitient, reflectivity and velocity. (More discussion is needed and we may add a couple more to better sort out special cases/anomalies.)

Correlation Coeffitient - Roundness of the object. Values close to 1 are spheres, which are likely precipitation, though it could be bugs depending on the time of day, season, and location. Around 0.8 is typically biological.

Reflectivity - The intensity of the EMR wave that is reflected. The idea is that the stronger the reflected signal, the more objects there are in that region being scanned

Differential_reflectivity - Change in the intensity of the reflectivity signal.

Velocity - Speed of the objects in the area being scanned.

Radial Velocity - 

Height/angle - These scans are binned in various angles above the horizon. We primarily use the lowest angle bin available in the scan (which is about 0.5°). This angle can have signals from birds taking off into the sky and weather signals. I assume that higher angles could exhibit both birds and weather signals, but I think it is more likely weather. At any rate, we tend to not look at these data streams in our process. - may be ignored in our project

Spectrum_width - Depicts a measure of velocity dispersion. In a radar bin, it provides a measure of the variability of the mean radial velocity estimates (movement) due to wind shear, turbulence, and/or the quality of the velocity samples. The proper use of Spectrum Width can help the severe thunderstorm and tornado warning decision process. --not useful for us.

We primarily use the Reflectivity and Correlation Coefficient to determine the status of the data for that day.

****Plotting:****
The first case we are looking at is a "Contaminated" day. We are looking at 3 random images during the day, and we are going to look at two variables. The first is "correlation coefficient" which tells us how round the objects are, precipitation tends to be a lot more round than birds. The second variable is "reflectivity", which helps us identify where objects are spatially.

![1](https://user-images.githubusercontent.com/84054117/202995723-4ba992d7-ce57-4dab-835d-03bcccff4ee3.png)
![2](https://user-images.githubusercontent.com/84054117/202995724-fde5109c-6773-4c06-b29f-69efd51e952c.png)
![3](https://user-images.githubusercontent.com/84054117/202995727-647ade41-302e-4afb-8687-f8b5a218b1d7.png)
![4](https://user-images.githubusercontent.com/84054117/202995728-5e00da31-0144-4ba5-b921-ec894e355d75.png)
![5](https://user-images.githubusercontent.com/84054117/202995882-71bd620b-3f4b-4ac8-a7f1-e2eb961adc46.png)
![6](https://user-images.githubusercontent.com/84054117/202995721-f5e9cf8c-d522-433e-8723-249361e81019.png)


The second case we are looking at is a "Non Contaminated" day.




![7](https://user-images.githubusercontent.com/84054117/203115483-f6d8ec46-403f-4c55-aae1-e3ca20d6c711.png)
![8](https://user-images.githubusercontent.com/84054117/203115486-7cb3e909-0873-48cc-b72e-1abc59067609.png)
![9](https://user-images.githubusercontent.com/84054117/203115487-d862714e-26b9-4649-88da-cea4d8b279f6.png)
![10](https://user-images.githubusercontent.com/84054117/203115488-7d7343de-7c4c-46e0-8b3c-105293f47b12.png)
![11](https://user-images.githubusercontent.com/84054117/203115490-11087c64-c042-4cec-8668-38ccf6b0202c.png)
![12](https://user-images.githubusercontent.com/84054117/203115481-d2affd1b-d1ba-42c7-a536-237032c5f7bf.png)


The files contain 72 column, each of them is a 2D array of floats. We can convert the floats to colors and display them in Unidata IVD.


**ADDRESSING PREPROCESSING**

1. Linear --> sucks --> neuro net? may not work
2. polinomial and logistic? May take forever..
3. Best bet: neuronet to figure it out by itself. 

The largest limitation currently is the massive amount of datapoints we are dealing with per day. Each day has 20 images associated with (51K values even if we only use 3 columns) and only a single 'status' classification. We will have to not only crop the images, but likely find an additional method to reduce how many points are being processed. Data outside of 150 km radius of the radar cannot show birds, and if it is contaminated outside of 250km, the scientist didn't mark it as contaminated. Those parts of the images are cropped.

We are still considering multiple ways to reduce the data size. One of the most promising ideas we have is we would do a sampling of the images and set an integer based on the sample. For example: we would look for high values of roundness and if that reaches a certain treshold in the data, we would assume it contains participation and set a certain value for the image. Add the images values together from the different timestamped images from the same day and feed that single integer to the Neural Network. That single integer may be standardized. This model could nicely work out if the anomalies are distinct and we find the right thresolds.

__Using pycaret to decide which model works best__


**RETRIEVING DATA**

Spreadsheets uploaded to drive:
https://drive.google.com/drive/folders/1-SCc3EW-wtdELNm4zVe5lT-u6V2DQJFJ?usp=share_link

How to retrive the NEXRAD files are described in this R-script:
https://drive.google.com/file/d/12LogSdZTVbxkuO8XF6-a_3MeGVH-_wmg/view?usp=share_link _(Colab doesn't support file storing for multiple runs, therefore it is best to have them donwloaded and stored on computer/cloud.)_

**GOOGLE COLAB**

https://colab.research.google.com/drive/16n72hFmsJis-llT24E0w4_E6ShFTNLNy?usp=sharing

