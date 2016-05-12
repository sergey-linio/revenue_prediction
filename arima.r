
library(forecast)
#library(plyr)
#library(dplyr)
#library(fpp)

get_forecast2 <- function(df,category,start,date_end,depth,last_week)
{
  result = list()
  
  df1 <- df[df$Cat == category,]
  ts_paid_price <- ts(df1$PaidPrice[c(start:(length(df1$Cat)-last_week))],end = date_end, frequency = 52)
  ts_items_sold <- ts(df1$items_sold[c(start:(length(df1$Cat)-last_week))],end = date_end, frequency = 52)
  
  fit_paid_price  <- auto.arima(log(ts_paid_price + 1))
  fit_items_sold <- auto.arima(log(ts_items_sold + 1))
  
  fc_paid_price <- forecast(fit_paid_price, h = depth)
  fc_items_sold <- forecast(fit_items_sold, h = depth)
  
  weeks <- as.character(seq(as.Date(head(tail(df1,1+last_week)$Date,1)),by = 7,length.out = depth + 1)[c(1:depth+1)])
  
  result$df_paid_price <- cbind.data.frame(rep(category,depth),weeks,round(as.numeric(exp(fc_paid_price$lower[,'80%']))),
                                          round(.6*as.numeric(exp(fc_paid_price$mean)) + .4*as.numeric(exp(fc_paid_price$lower[,'80%']))),
                                          round(1.05*as.numeric(exp(fc_paid_price$mean))))
  names(result$df_paid_price) <- c('Cat','week','pessimistic','average','optimistic')
    
  result$df_items_sold <- cbind.data.frame(rep(category,depth),weeks,round(as.numeric(exp(fc_items_sold$lower[,'80%']))),
                                           round(.6*as.numeric(exp(fc_items_sold$mean)) + .4*as.numeric(exp(fc_items_sold$lower[,'80%']))),
                                            round(1.05*as.numeric(exp(fc_items_sold$mean))))
  names(result$df_items_sold) <- c('Cat','week','pessimistic','average','optimistic')
    
  result
}

get_forecast_for_cats <- function(df,categories,start,date_end,depth,last_week)
{
    result = list()
    df_price = data.frame()
    df_items = data.frame()
    len = length(categories)
    
    for (i in 1:len)
    {
        try(df_fc <- get_forecast2(df,categories[i],start,date_end,depth,last_week))
        #list_items[i] = df_fc$df_items_sold
        #list_price[i] = df_fc$df_paid_price
        try(df_price <- rbind(df_price,df_fc$df_paid_price))
        try(df_items <- rbind(df_items,df_fc$df_items_sold))
        #print(i)
    }
    
    result$items = df_items
    result$price = df_price
    
    result
}

myArgs <- commandArgs(trailingOnly = TRUE)
df1_path = myArgs[2] #path to file with category sales
df2_path = myArgs[3] #path to file with category names

#sales data
df1 <- read.csv2(df1_path,header = TRUE,sep = ',',dec = '.')
names(df1) <- c('X','Cat','Date','PaidPrice','items_sold')

#top cats data
df2 <- read.csv2(df2_path,header = TRUE,sep = ',',dec = '.')
names(df2) <- c('X','items_sold','PaidPrice','Cat')


# Convert to numerics
topN = as.numeric(myArgs[1]) #how many categories to predict
year = as.numeric(myArgs[4]) #current year
week = as.numeric(myArgs[5]) #current week
training_weeks = as.numeric(myArgs[6]) #how many weeks in training period
weeks_back = as.numeric(myArgs[7]) # shift of current week to [weeks_back] (timemachine, to forecast on historic data)
forecast_depth = as.numeric(myArgs[8]) # how many weeks to forecast.

top_cat_names = c(as.character(df2$Cat[c(1:topN)]))

test1 = get_forecast_for_cats(df1,top_cat_names,training_weeks,c(year,week),forecast_depth,weeks_back)

write.csv2(test1$items,'items2.csv')
write.csv2(test1$price,'price2.csv')