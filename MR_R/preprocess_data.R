rm(list=ls())
library(reshape2)
library(data.table)
library(dplyr)
library(forcats) ## replace NA into None
library(lubridate)
library(ggplot2)

library("keras")
library("magrittr")
library("progress")
library("hrbrthemes")
library("ggsci")
library(cowplot) 

## combine the data 
records <- read.csv(file="data/MSE_OAE_Cat_ML.csv", header=TRUE, sep=",")
articlesIDs <- read.csv(file = 'data/MSE_WoundArticleID_UniqueId_ML.csv',sep=',')
woundIDs <- read.csv(file='data/MSE_WoundProps_WoundId_ML.csv',sep=',')

records <- left_join(records,woundIDs, by =intersect(colnames(records),colnames(woundIDs)))
records <- left_join(records,articlesIDs, by =c("Artikel" = "ArticleId"))

#crete modelID and artcileID matrix
df_pair <- records[,c('WoundId','UniqueArticleId')]

pair <- expand.grid(
  "WoundId" = unique(df_pair$WoundId),
  "Article" = unique(df_pair$UniqueArticleId),
  stringsAsFactors = FALSE
)
pair$"class" <- 0L

# for outlier: wound_ID 12 does not appeared in the recording data
pair_temp <- expand.grid(
  "WoundId" = 12,
  "Article" = unique(df_pair$UniqueArticleId),
  stringsAsFactors = FALSE
)
pair_temp$"class" <- 0L
pair <- rbind(pair_temp,pair)


pb <- progress_bar$new(
  format = "[:bar] :percent eta: :eta", total = nrow(df_pair)
)
for (i in 1L:nrow(df_pair)) {
  pb$tick()
  pair[intersect(
    which(df_pair$WoundId[i] == pair$WoundId),
    which(df_pair$UniqueArticleId[i] == pair$Article)
  ), "class"] <- 1L
}
summary(pair)

## save the data 
saveRDS(pair,"data/pair.RDa")

