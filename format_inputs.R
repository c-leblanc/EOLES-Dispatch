#### INPUT FORMATTING SCRIPT - README
# This script selects and format the data to be used by the EOLES-Dispatch Model
# The entire script must be run before each run of the model after setting the parameters below
# 
# It takes data from the complete sets stored in <time-varying_inputs> and 
# <renewable_ninja> directories and from the scenario excel file, and stores 
# the formatted data ready to be used by EOLES-Dispatch in the <inputs> directory
########

rm(list = ls())
library(readxl)
library(dplyr)
library(reshape2)
library(lubridate)

library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))

#### BASIC PARAMETERS OF THE MODEL RUN ####
# Replace dates below to use another period (period selected is including start day and excluding end day)
start <- as.POSIXct("2019/01/01",tz="UTC")
end <- as.POSIXct("2020/01/01",tz="UTC")
scenario <- "Scenario_FORMAT" # Set the name of the excel file containing the scenario to run
actCF <- FALSE                    # Set to false to download Renewable Ninja projections instead of historic CFs
rn_horizon <- "CU"                # Set the time horizon to consider for solar and wind capacity factors from renewable.ninja
#     options : "CU"->Current ; "NT"->Near-Term ; "LT"->Long-Term

#### Additional Parameters ####
# Select countries to include among the list below:
# France (FR), Belgium (BE), Germany + Luxembourg (DE), Switzerland (CH), Italy (IT), Spain (ES) and United-Kingdom (UK)
areas <- c("FR","BE","DE","CH","IT","ES","UK")
# Select countries/zones to include as exogenous traders (hourly prices taken as inputs of the model)
exo_areas <- c("NL","DK1","DK2","SE4","PL","CZ","AT","GR","SI","PT","IE")

################################################################################
#######################   NOT FOR MODIFICATION  ################################
################################################################################
##### REFORMAT TIME-VARYING INPUTS #####
  #HOURLY
  tvHourLoad <- function(variable, areas, start, end){
    return(read.csv2(paste0("time-varying_inputs/",variable,".csv"), sep=",", dec=".", stringsAsFactors = F) %>%
             mutate(hour = as.POSIXct(hour, tz="UTC")) %>% 
             filter(hour>=start) %>% filter(hour<end) %>%
             mutate(hour = as.numeric(hour)/3600) %>% mutate(hour = as.integer(hour)) %>%
             select("hour", all_of(areas)) %>% 
             melt(id="hour", variable.name = "area") %>%
             select(area,hour,value))
  }
  ninjaHourLoad <- function(variable, areas, start, end){
    return(read.csv2(paste0("renewable_ninja/",variable,".csv"), sep=",", dec=".", stringsAsFactors = F) %>%
             mutate(hour = as.POSIXct(hour, tz="UTC")) %>%
             filter(hour>=start & hour<end) %>%
             mutate(hour = as.numeric(hour)/3600) %>% mutate(hour = as.integer(hour)) %>%
             select("hour", all_of(areas)) %>% 
             melt(id="hour", variable.name = "area") %>%
             select(area,hour,value))
  }
  
  demand <- tvHourLoad("demand",areas,start,end)
  nmd <- tvHourLoad("nmd",areas,start,end)
  exoPrices <- tvHourLoad("exoPrices",exo_areas,start,end)
  vre_profiles <- data.frame()
  if(actCF){
    for (tec in c("offshore","onshore","pv")){
      temp <- tvHourLoad(tec,areas,start,end)
      temp <- cbind(tec = rep(tec,nrow(temp)),temp)[,c("area","tec","hour","value")] %>% arrange(area)
      vre_profiles <- rbind(vre_profiles,temp)
      rm(temp)
    }
    rm(tec)
  }else{
    offshore <- ninjaHourLoad(paste0("offshore_",rn_horizon),areas,start,end)
    offshore <- cbind(tec = rep("offshore",nrow(offshore)),offshore)[,c("area","tec","hour","value")] %>% arrange(area)
    
    if(rn_horizon %in% c("NT","LT")){
      onshore <- ninjaHourLoad("onshore_NT",areas,start,end)}else{
        onshore <- ninjaHourLoad("onshore_CU",areas,start,end)}
    onshore <- cbind(tec = rep("onshore",nrow(onshore)),onshore)[,c("area","tec","hour","value")] %>% arrange(area)
        
    pv <- ninjaHourLoad("pv",areas,start,end)
    pv <- cbind(tec = rep("pv",nrow(pv)),pv)[,c("area","tec","hour","value")] %>% arrange(area)
    
    vre_profiles <- rbind(offshore, onshore, pv)
    rm(offshore,onshore,pv)
  }
  temp <- tvHourLoad("river",areas,start,end)
  temp <- cbind(tec = rep("river",nrow(temp)),temp)[,c("area","tec","hour","value")] %>% arrange(area)
  vre_profiles <- rbind(vre_profiles,temp)
  rm(temp)
  rm(tvHourLoad,ninjaHourLoad)
  
  #MONTHLY/WEEKLY
  hour_month <- data.frame(hour=unique(demand$hour)) %>% 
    mutate(hour_POSIX = as.POSIXct(hour*3600, tz="UTC", origin='1970-01-01')) %>%
    mutate(month=paste0(year(hour_POSIX),format(hour_POSIX,"%m"))) %>%
    select("hour", "month")
  hour_week <- data.frame(hour=unique(demand$hour)) %>% 
    mutate(hour_POSIX = as.POSIXct(hour*3600, tz="UTC", origin='1970-01-01')) %>%
    mutate(week=paste0(year(hour_POSIX),format(hour_POSIX,"%W"))) %>%
    select("hour", "week")
  
  lake_inflows <- read.csv2("time-varying_inputs/lake_inflows.csv", sep=",", dec=".", stringsAsFactors = F) %>% 
    select("month", all_of(areas)) %>% 
    filter(month %in% hour_month$month) %>%
    melt(id="month", variable.name = "area") %>%
    select(area, month, value)
  hMaxIn <- read.csv2("time-varying_inputs/hMaxIn.csv", sep=",", dec=".", stringsAsFactors = F) %>% 
    select("month", all_of(areas)) %>% 
    filter(month %in% hour_month$month) %>%
    melt(id="month", variable.name = "area") %>%
    select(area, month, value)
  hMaxOut <- read.csv2("time-varying_inputs/hMaxOut.csv", sep=",", dec=".", stringsAsFactors = F) %>% 
    select("month", all_of(areas)) %>% 
    filter(month %in% hour_month$month) %>%
    melt(id="month", variable.name = "area") %>%
    select(area, month, value)
  nucMaxAF <- read.csv2("time-varying_inputs/nucMaxAF.csv", sep=",", dec=".", stringsAsFactors = F) %>% 
    select("week", all_of(areas)) %>% 
    filter(week %in% hour_week$week) %>%
    melt(id="week", variable.name = "area") %>%
    select(area, week, value)
  
  #indexes
  hours <- unique(hour_month$hour)
  weeks <- unique(hour_week$week)
  months <- unique(hour_month$month)
  
####################################
  
  
##### REFORMAT SCENARIO INPUTS #####
  thr_specs <- read_excel(paste0(scenario,".xlsx"), sheet = "thr_specs")
  thr <- thr_specs$tec
  for (v in names(thr_specs)[2:length(thr_specs)]){
    assign(v,thr_specs[,c("tec",v)])
  }
  rm(thr_specs,v)
  
  rsv_req <- read_excel(paste0(scenario,".xlsx"), sheet = "rsv_req")
  vre <- rsv_req$tec
  str_vOM <- read_excel(paste0(scenario,".xlsx"), sheet = "str_vOM")
  str <- str_vOM$tec
  
  tec <- c("nmd",vre,thr,str)
  frr <- c(frr[which(frr$frr=="TRUE"),]$tec,str)
  no_frr <- setdiff(tec,frr)
  
  capa <- melt(read_excel(paste0(scenario,".xlsx"), sheet = "capa"), id="tec", variable.name="area")
  capa <- capa[which(capa$area %in% areas),c("area","tec","value")]
  maxAF <- melt(read_excel(paste0(scenario,".xlsx"), sheet = "maxAF"), id="tec", variable.name="area")
  maxAF <- maxAF[which(maxAF$area %in% areas),c("area","tec","value")]
  yEAF <- melt(read_excel(paste0(scenario,".xlsx"), sheet = "yEAF"), id="tec", variable.name="area")
  yEAF <- yEAF[which(yEAF$area %in% areas),c("area","tec","value")]
  capa_in <- melt(read_excel(paste0(scenario,".xlsx"), sheet = "capa_in"), id="tec", variable.name="area")
  capa_in <- capa_in[which(capa_in$area %in% areas),c("area","tec","value")]
  stockMax <- melt(read_excel(paste0(scenario,".xlsx"), sheet = "stockMax"), id="tec", variable.name="area")
  stockMax <- stockMax[which(stockMax$area %in% areas),c("area","tec","value")]
  links <- melt(read_excel(paste0(scenario,".xlsx"), sheet = "links"), 
                id = c("exporter"), variable.name = "importer")
  links <- links[(links$importer %in% areas)&(links$exporter %in% areas),c("importer","exporter","value")]
  exo_EX <- melt(read_excel(paste0(scenario,".xlsx"), sheet = "exo_EX"), 
                 id = c("exporter"), variable.name = "importer")
  exo_IM <- melt(read_excel(paste0(scenario,".xlsx"), sheet = "exo_IM"), 
                 id = c("importer"), variable.name = "exporter")
  fuel_timeFactor <- melt(read_excel(paste0(scenario,".xlsx"), sheet = "fuel_timeFactor"), id="month", variable.name = "fuel")
  fuel_timeFactor <- fuel_timeFactor[which(fuel_timeFactor$month %in% hour_month$month),c("fuel","month","value")]
  fuel_areaFactor <- melt(read_excel(paste0(scenario,".xlsx"), sheet = "fuel_areaFactor"), id="area", variable.name = "fuel")
  fuel_areaFactor <- fuel_areaFactor[which(fuel_areaFactor$area %in% areas),c("fuel","area","value")]
  
  rm(actCF,start,end)
########################################
  
for (t in ls()[ls()!="t"]){
  write.table(get(t), file = paste0("inputs/",t,".csv"), append = FALSE, quote = FALSE, sep = ",",row.names = FALSE, col.names = FALSE)
}
  
rm(list = ls())
  

