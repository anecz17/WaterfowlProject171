########################################################################
#  Script to download NEXRAD radar data via R using the aws.s3 package.
#  User can set hours before and after sunset to bracket time range 
#  (in UTC time) to download and store data on OneDrive or locally
########################################################################  

#################################################
# STEP 1. Install and read in initial libraries #
#################################################
{
  ## initial packages to install
  packages <- c("rio","aws.s3","maptools","lubridate","stringr","plyr","openxlsx")
  inst <- packages %in% installed.packages()
  if (length(packages[!inst])>0) install.packages(packages[!inst],dependencies = T)
  lapply(packages,require,character.only=TRUE)
}

#####################################
# STEP 2. Setup paths and variables #
#####################################
{
radar<-"station_name"
STATION<-paste0("K",radar)
screening<-"station_screening.xls"  ## name of screening spreadsheet
nexrad<-read.csv("station_path")  ## location of nexrad site file
dtime<- 1 ## number of hours after sunset you wish to download
btime<- 1 ## number of hours before sunset you wish to download (if 0, the program will download files rounded down to nearest hour to sunset)

path<-"spreadsheet_path"            # location of screening spreadsheet
outpath<-"saving_path"              # base location to save data
saveLoc<-paste0(outpath,radar)    
}

####################################################
# STEP 3. Function to download.  No changes needed #
####################################################
{
# Function to download NEXRAD2 files from AWS -----------------------------
##version: 0.1.1
##Author: Michael Whitby
##EMAIL: michael.whitby@gmail.com
##July 28, 2017 

get_NEXRAD2 <- function(STATION,##SET STATION ID - four letter character string
                        DATES,##vector of dates as a string in the YYMMDD format
                        saveLoc=getwd(),
                        startHOUR,##Start and end hours of interest for each day (must be same for all days) as integer 0-23
                        endHOUR,
                        AWSbucket="https://noaa-nexrad-level2.s3.amazonaws.com/"
) 
  
{
  dates.year <- substring(DATES, 1, 4)
  dates.month <- substring(DATES, 5, 6)
  dates.day <- substring(DATES, 7,9)
  
  times <- seq(startHOUR, endHOUR)
  
  times <- stringr::str_pad(times, 2, pad="0")
  
  #files <- paste0(STATION,DATES,"_", times)
  files <- c(t(outer(DATES, times, paste)))
  files <- sub(" ", "_", files)
  files <- paste0(STATION, files)
  location <- paste0(dates.year, "/", dates.month, "/", dates.day, "/", STATION, "/")
  obj <- c(t(outer(location, files, paste0)))
  
  all <- list()
  all <- lapply( obj, function(x){get_bucket_df(bucket = "noaa-nexrad-level2", prefix=x)}
  )
  
  toDWNLD <- plyr::ldply(all)$Key
  
  DWNLDURLS <- c(t(outer(AWSbucket, toDWNLD, paste0)))
  
  saveLoc=paste0(saveLoc, "/")
  dir.create(saveLoc,showWarnings = F)
  saveFile=substr(toDWNLD, 17, 42)
  dub <- (saveFile  %in% list.files(saveLoc))
  saveFile = saveFile[!dub]
  
  saveFile=paste0(saveLoc, saveFile)
  toDWNLD =toDWNLD[!dub]
  
  lapply(1:length(DWNLDURLS), function (i){ 
    download.file(DWNLDURLS[i], destfile = saveFile[i], mode="wb",quiet=T)  
  })
  
}
}

###############################
# STEP 4. Download radar data #
###############################
{
if (exists("endHOUR")){rm(endHOUR)}  ## making sure this value is removed, otherwise times will get messed up  

##folder to save radar.gz files to
dir.create(saveLoc,showWarnings = T)
Sys.chmod(saveLoc,mode="0777",use_umask = T)
setwd(saveLoc)

data <- import_list(paste0(path,"/",screening), setclass = "tbl", rbind = TRUE)
colnames(data)<-toupper(colnames(data))
## subset for columns DATE and STATUS
if (class(data)[1]=="list") {
      data<-as.data.frame(do.call(rbind, lapply(data, subset, select=c("DATE", "STATUS"))))
      } else {
      data<-subset(data, select=c("DATE","STATUS"))  
      }
## subset for dates to download (not contaminated)
if (length(which(data$STATUS=="B"))==0) {
      data <- data[is.na(data$STATUS),]
      } else {
      data <- data[grep("^B$", data$STATUS),]
      }

data$DATE[nchar(data$DATE)<8]<-as.character(as.Date(as.numeric(data$DATE[nchar(data$DATE)<8]), origin="1899-12-30"))
data$DATE<-as.POSIXct(format(as.Date(data$DATE),'%Y-%m-%d UTC'),tz="UTC")
## Loop through dates using nearest hour before sunset + 2 hours to download
for (j in 1:length(data$DATE)){
  saveLoc<-paste0(outpath,radar,"/",as.character(data$DATE[j]))
  dir.create(saveLoc,showWarnings = T)
  Sys.chmod(saveLoc,mode="0777",use_umask = T)
  setwd(saveLoc)
  
crds<-as.matrix(cbind(nexrad[nexrad$SITE %in% radar,]$LONGITUDE_W,nexrad[nexrad$SITE %in% radar,]$LATITUDE_N))
sunset<-(sunriset(crds, data$DATE[j],proj4string=CRS("+proj=longlat +datum=WGS84"),direction="sunset", POSIXct.out=TRUE)[,2])

startHOUR<- as.numeric(substr(format(floor_date(sunset, unit="hour"), format="%H:%M"),1,2)) - btime ## rounded down to nearest hour to sunset
if (startHOUR<24 & startHOUR>15){
  endHOUR<-as.numeric(substr(format(floor_date(sunset, unit="hour"), format="%H:%M"),1,2))+dtime
  if (endHOUR>23) {endHOUR<-23}
  DATES<-gsub("-","",substr(sunset,1,10))  ## UTC date (not local day)
  tryCatch(get_NEXRAD2(STATION,DATES,saveLoc,startHOUR,endHOUR),
           error=function(e) {print(paste0("no files for ",data$DATE[j]," in AWS bucket -- it's okay to move on"))})
  
}

if (exists("endHOUR")){
  endHOUR<-as.numeric(substr(format(floor_date(sunset, unit="hour"), format="%H:%M"),1,2))+dtime-23
} else {
  endHOUR <- as.numeric(substr(format(floor_date(sunset, unit="hour"), format="%H:%M"),1,2))+dtime   ### increase number if want to download more data
}

## remove MD files
junk <- dir(saveLoc,  pattern="*_MD") 
file.remove(junk)

if (endHOUR<1){
  next
} else {
DATES<-gsub("-","",substr(sunset,1,10))  ## UTC date (not local day)
## but change DATES if startHOUR is from previous day
if (startHOUR<24 & startHOUR>15){DATES<-gsub("-","",substr(sunset+86400,1,10))
startHOUR<-0}  ## UTC date (not local day)

if (as.numeric(substr(format(floor_date(sunset, unit="hour"), format="%H:%M"),1,2))+dtime==24) {endHOUR<-0}
## run function to download data
tryCatch(get_NEXRAD2(STATION,DATES,saveLoc,startHOUR,endHOUR),
         error=function(e) {print(paste0("no files for ",data$DATE[j]," in AWS bucket -- it's okay to move on"))})
rm(endHOUR)
}

## remove MD files
junk <- dir(saveLoc,  pattern="*_MD") 
file.remove(junk)
}
}
