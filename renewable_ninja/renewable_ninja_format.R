rm(list = ls())
library(dplyr)
library(lubridate)

library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))

areas <- c("FR","BE","DE","CH","IT","ES","UK")

pv <- data.frame(hour=as.POSIXct(character()))
for(area in areas){
  pv_temp <- read.csv2(file=paste0("Raw_Data/PV_",area,".csv"), skip=2, sep=",", dec=".", stringsAsFactors = F) %>%
    mutate(time = as.POSIXct(time, tz="UTC")) %>% rename(!! area := national, hour = time)
  pv <- pv %>% full_join(pv_temp, id="hour")
  rm(pv_temp)
}

#### Integrating future fleet : 
# additional files represent the capacity factors of plants to be built
# here they are integrated to the current fleet using weighted average,
# taking renewable.ninja's hypothsesis regarding near term and long term capacity as weights
on_weights <- list()
on_weights[["FR"]] <- c(CU=10831,NT=684)
on_weights[["BE"]] <- c(CU=1488,NT=55)
on_weights[["DE"]] <- c(CU=27871,NT=525)
on_weights[["CH"]] <- c(CU=75,NT=0)
on_weights[["IT"]] <- c(CU=8440,NT=28)
on_weights[["ES"]] <- c(CU=22033,NT=6)
on_weights[["UK"]] <- c(CU=8052,NT=1549)

offweights <- list()
offweights[["FR"]] <- c(CU=6,NT=1951,LT=2039)
offweights[["BE"]] <- c(CU=712,NT=1359,LT=224)
offweights[["DE"]] <- c(CU=3268,NT=8008,LT=10493)
offweights[["CH"]] <- c(CU=0,NT=0,LT=0)
offweights[["IT"]] <- c(CU=0,NT=167,LT=40)
offweights[["ES"]] <- c(CU=0,NT=0,LT=43)
offweights[["UK"]] <- c(CU=5100,NT=16027,LT=13523)

onshore_CU <- data.frame(hour=as.POSIXct(character()))
offshore_CU <- data.frame(hour=as.POSIXct(character()))
onshore_NT <- data.frame(hour=as.POSIXct(character()))
offshore_NT <- data.frame(hour=as.POSIXct(character()))
offshore_LT <- data.frame(hour=as.POSIXct(character()))
for(area in areas){
  #Collect the separate capacity factors for the current fleet
  try(CU <- read.csv2(file=paste0("Raw_Data/Wind_CU_",area,".csv"), skip=2, sep=",", dec=".", stringsAsFactors = F) %>%
    mutate(hour = as.POSIXct(time, tz="UTC")) %>% select(-time))
  if(ncol(CU)==2){CU <- CU %>% mutate(onshore=national,offshore=national)} #Non-existing technology will be ignored through the weight equal to zero
  
  #Collect the separate capacity factors for the near-term fleet
  try(NT <- read.csv2(file=paste0("Raw_Data/Wind_NT_",area,".csv"), skip=2, sep=",", dec=".", stringsAsFactors = F) %>%
      mutate(hour = as.POSIXct(time, tz="UTC")) %>% select(-time))
  try(if(ncol(NT)==2){NT <- NT %>% mutate(onshore=national,offshore=national)}) #Non-existing technology will be ignored through the weight equal to zero
  if(!exists("NT")){NT <- data.frame(onshore=0,offshore=0)}
  
  #Collect the separate capacity factors for the long-term fleet
  try(LT <- read.csv2(file=paste0("Raw_Data/Wind_LT_",area,".csv"), skip=2, sep=",", dec=".", stringsAsFactors = F) %>%
      mutate(hour = as.POSIXct(time, tz="UTC")) %>% select(-time))
  try(if(ncol(LT)==2){LT <- LT %>% mutate(onshore=national,offshore=national)}) #Non-existing technology will be ignored through the weight equal to zero
  if(!exists("LT")){LT <- data.frame(onshore=0,offshore=0)}
  
  #Builds the time-series considering only the current fleet
  onshore_temp <- CU %>% select(hour) %>% 
    mutate(!! area := (CU$onshore*on_weights[[area]]["CU"])/sum(on_weights[[area]][1]))
  offshore_temp <- CU %>% select(hour) %>% 
    mutate(!! area := (CU$offshore*offweights[[area]]["CU"])/sum(offweights[[area]][1]))
  onshore_CU <- onshore_CU %>% full_join(onshore_temp,by="hour")
  offshore_CU <- offshore_CU %>% full_join(offshore_temp,by="hour")
  
  #Builds the time-series considering only current and near-term fleet
  onshore_temp <- CU %>% select(hour) %>% 
    mutate(!! area := (CU$onshore*on_weights[[area]]["CU"]+NT$onshore*on_weights[[area]]["NT"])/sum(on_weights[[area]][1:2]))
  offshore_temp <- CU %>% select(hour) %>% 
    mutate(!! area := (CU$offshore*offweights[[area]]["CU"]+NT$offshore*offweights[[area]]["NT"])/sum(offweights[[area]][1:2]))
  onshore_NT <- onshore_NT %>% full_join(onshore_temp,by="hour")
  offshore_NT <- offshore_NT %>% full_join(offshore_temp,by="hour")
  
  #Builds the time-series considering current, near-term and long-term fleet (available only for offshore)
  offshore_temp <- CU %>% select(hour) %>% 
    mutate(!! area := (CU$offshore*offweights[[area]]["CU"]+NT$offshore*offweights[[area]]["NT"]+LT$offshore*offweights[[area]]["LT"])/sum(offweights[[area]][1:3]))
  offshore_LT <- offshore_LT %>% full_join(offshore_temp,by="hour")
  rm(CU,NT,LT,onshore_temp,offshore_temp)
}

offshore_CU[is.na(offshore_CU)] <- 0
offshore_NT[is.na(offshore_NT)] <- 0
offshore_LT[is.na(offshore_LT)] <- 0

pv <- pv %>% filter(hour>=as.POSIXct("2015/01/01",tz="UTC")) %>% filter(hour<as.POSIXct("2020/01/01",tz="UTC"))
onshore_CU <- onshore_CU %>% filter(hour>=as.POSIXct("2015/01/01",tz="UTC")) %>% filter(hour<as.POSIXct("2020/01/01",tz="UTC"))
onshore_NT <- onshore_NT %>% filter(hour>=as.POSIXct("2015/01/01",tz="UTC")) %>% filter(hour<as.POSIXct("2020/01/01",tz="UTC"))
offshore_CU <- offshore_CU %>% filter(hour>=as.POSIXct("2015/01/01",tz="UTC")) %>% filter(hour<as.POSIXct("2020/01/01",tz="UTC"))
offshore_NT <- offshore_NT %>% filter(hour>=as.POSIXct("2015/01/01",tz="UTC")) %>% filter(hour<as.POSIXct("2020/01/01",tz="UTC"))
offshore_LT <- offshore_LT %>% filter(hour>=as.POSIXct("2015/01/01",tz="UTC")) %>% filter(hour<as.POSIXct("2020/01/01",tz="UTC"))


write.table(pv,file="pv.csv",sep=",",dec=".",quote=F,row.names = FALSE)
write.table(onshore_CU,file="onshore_CU.csv",sep=",",dec=".",quote=F,row.names = FALSE)
write.table(onshore_NT,file="onshore_NT.csv",sep=",",dec=".",quote=F,row.names = FALSE)
write.table(offshore_CU,file="offshore_CU.csv",sep=",",dec=".",quote=F,row.names = FALSE)
write.table(offshore_NT,file="offshore_NT.csv",sep=",",dec=".",quote=F,row.names = FALSE)
write.table(offshore_LT,file="offshore_LT.csv",sep=",",dec=".",quote=F,row.names = FALSE)

rm(list = ls())


