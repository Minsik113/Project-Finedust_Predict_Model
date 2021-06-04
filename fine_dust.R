library(ggplot2)
library(readxl)
library(dplyr)
library(writexl)
library(car)
library(Metrics)
# rm(list=ls())

##########################################
############## import - data #############
##########################################
data_raw <- read_xlsx("데이터 찐 최종_v2.xlsx")
data = data_raw # copy
str(data)
data <- as.data.frame(data)

# 약간의 전처리
data$풍향 <- as.factor(data$풍향)
data <- data %>% select(-mean_PM25,-계절)
data <- data %>% mutate(월 = substr(data$일시,6,7))
data$월 <- as.factor(data$월)
str(data)
data <- data[-1]

for(v in unique(data$월)) data[paste0("월_",v)] <- ifelse(data$월==v,1,0)
data <- data %>% select(-월) 
for(v in unique(data$풍향)) data[paste0("풍향_",v)] <- ifelse(data$풍향==v,1,0)
data <- data %>% select(-풍향) 

data2 <- data # copy

##########################################
########## 이상치 제거 or 보간 ###########
##########################################
boxplot(data2$day_tr_sum)$stats
data2$day_tr_sum <- ifelse(data2$day_tr_sum<3944541 , 3944541 , data2$day_tr_sum)
data2$day_tr_sum <- ifelse(data2$day_tr_sum>10667256 , 10667256 , data2$day_tr_sum)
plot(data2$day_tr_sum)

boxplot(data2$LPG)$stats

boxplot(data2$bei_PM10)$stats
data2$bei_PM10 <- log(data2$bei_PM10)

boxplot(data2$shan_PM10)$stats
data2$shan_PM10 <- log(data2$shan_PM10)

boxplot(data2$mean_SO2)$stats

boxplot(data2$mean_CO)$stats

boxplot(data2$mean_O3)$stats

boxplot(data2$평균기온)$stats

boxplot(data2$어제PM10)$stats
data2$어제PM10 <- log(data2$어제PM10)

boxplot(data2$어제중국PM10)$stats
data2$어제중국PM10 <- log(data2$어제중국PM10)

boxplot(data2$일교차)$stats


boxplot(data2$일강수량)$stats
data2$일강수량 <- (data2$일강수량+1)
data2$일강수량 <- log(data2$일강수량)

boxplot(data2$평균풍속)$stats

plot(data2$평균풍속 , data2$mean_PM10)

boxplot(data2$평균상대습도)$stats

boxplot(data2$평균증기압)$stats

boxplot(data2$평균전운량)$stats

boxplot(data2$휘발유)$stats

boxplot(data2$벙커C유)$stats
data2$벙커C유 <- ifelse(data2$벙커C유>5.12903226 , 5.12903226 , data2$벙커C유)

boxplot(data2$등유)$stats

boxplot(data2$경유)$stats

boxplot(data2$기타)$stats
data2$기타 <- ifelse(data2$기타>52.03226 , 52.03226 , data2$기타)

data2$mean_PM10 <- log(data2$mean_PM10)
####################정규화#####################
normalize<-function(x){
  return((x-min(x))/(max(x)-min(x)))
}

str(data2)
a <- data2 %>% select(-mean_PM10) 
data_norm <- sapply(a,normalize)
data_norm <- as.data.frame(data_norm)
str(data_norm)
summary(data_norm)
str(data_norm)
data_norm <- cbind(data_norm,data2 %>% select(mean_PM10))
str(data_norm)
aaaa = data_norm[nrow(data_norm),]
##########################################
##########################################
##########train , test 데이터 추출########
##########################################
library(readxl) # which = sample(1:1825,1460)

data_train_raw = read_xlsx('트레인 데이터.xlsx')
data_test_raw = read_xlsx('테스트 데이터.xlsx')
data_0528_raw = read_xlsx('결과예측 데이터.xlsx')

train_data = as.data.frame(data_train_raw)
test_data = as.data.frame(data_test_raw)
# test_data = as.data.frame(data_0528_raw)
str(train_data)

##########################################
########### 변수추출 - XGboost ###########
##########################################
library(xgboost)

# 8:2 split training, test
train_label = train_data[,51]
test_label = test_data[,51]
# test_label_0528 = data_0528[,-51]

train_data$mean_PM10 = NULL
test_data$mean_PM10 = NULL
# test_label_0528$mean_PM10 = NULL


# Type change. dataframe -> matrix
train_data = as.matrix(train_data)
test_data = as.matrix(test_data)
# test_label_0528 = as.matrix(test_label_0528)

# create xgb.DMatrix
dtrain = xgb.DMatrix(data=train_data,label=train_label)
dtest = xgb.DMatrix(data=test_data,label=test_label)
# d0528 = xgb.DMatrix(data=data_0528,label=test_label_0528)

# train a model using our training data
model_tuned <- xgboost(data = dtrain, # the data           
                       max.depth = 3, # the maximum depth of each decision tree
                       nround = 10, # number of boosting rounds
                       early_stopping_rounds = 5,
                       gamma = 1) # add a regularization term

importance_matrix = xgb.importance(model=model_tuned)

# visualization importance plot
xgb.plot.importance(importance_matrix=importance_matrix,
                    rel_to_first = FALSE,
                    xlab = "Relative importance")


##########################회귀식 예측########
pred = predict(result5, data_test)

rmse(exp(pred),exp(data_test$mean_PM10))

x <- cbind(pred,data_test$mean_PM10)
View(x)


data_train <- data_train5
str(data_train)
data_train$풍향영향 <- as.numeric(data_train$풍향영향)
data_train$월영향 <- as.numeric(data_train$월영향)
str(data_test)
data_test$풍향영향 <- as.numeric(data_test$풍향영향)
data_test$월영향 <- as.numeric(data_test$월영향)


data_train <- data_train %>% select(-풍향)
data_test <- data_test %>% select(-월,-풍향)

data_train$풍향영향 <- data_train$풍향영향-1
data_train$월영향 <- data_train$월영향-1

data_test$풍향영향 <- data_test$풍향영향-1
data_test$월영향 <- data_test$월영향-1

###########인공신경망########
library(neuralnet)
library(caret)
library(Metrics)
############################################
############################################
############################################
# xgboost에서 뽑은 변수로 인공신경
# data_raw의 8대 2 sample
# data_train_raw, data_test_raw
############################################
############################################
############################################
aa = data_train_raw
importance_matrix$Feature
# aa = max_depth==3으로 뽑힌 변수들.
aa = data_train_raw %>% 
  select(importance_matrix$Feature,mean_PM10) 
aa = as.data.frame(aa)
str(aa)

data_train1 = aa # train
data = as.data.frame(data_raw) # 전체. 정규화할때 전체데이터에서 min max값이 필요하니까.
data_test = as.data.frame(data_0528_raw) # row가1개인 데이터
data_test3 = as.data.frame(data_test_raw) # test


# mean_PM10정규화
data_train1$mean_PM10 <- (data_train1$mean_PM10-min(data$mean_PM10))/(max(data$mean_PM10)-min(data$mean_PM10))
data_test$mean_PM10 <-(data_test$mean_PM10-min(data$mean_PM10))/(max(data$mean_PM10)-min(data$mean_PM10))
data_test3$mean_PM10 <-(data_test3$mean_PM10-min(data$mean_PM10))/(max(data$mean_PM10)-min(data$mean_PM10))
str(data_train1)
str(data_test)
str(data_test3)
# 모델링 -> 예측
pm10_model2 <- neuralnet(formula = mean_PM10~., data=data_train1, hidden=c(3,3),act.fct = "relu")
?neuralnet
model_results2 <- compute(pm10_model2, data_test[-51]) # 5/28
model_results3 = compute(pm10_model2, data_test3[-51])

str(pm10_model2)
pre_str2 <- model_results2$net.result
pre_str3 <- model_results3$net.result
pre_str2
plot()

# saveRDS(pm10_model2, "./final_model1.rds")
# tttttt = readRDS("./train12_988andtest13_431and0528_2_85.rds")
# model_results3 = compute(tttttt, data_test3[-51])
# pre_str3 <- model_results3$net.result
# str(model_results2$net.result)
# 다시 원래값으로 되돌리기 (정규환된 값->원래 값)
  # train, 5/28
pre_str2 <- pre_str2*(max(data$mean_PM10)-min(data$mean_PM10))+min(data$mean_PM10)
test_real <- data_test$mean_PM10*(max(data$mean_PM10)-min(data$mean_PM10))+min(data$mean_PM10)
RMSE(pre_str2,test_real)
  # train, test
pre_str3 <- pre_str3*(max(data$mean_PM10)-min(data$mean_PM10))+min(data$mean_PM10)
test_real3 <- data_test3$mean_PM10*(max(data$mean_PM10)-min(data$mean_PM10))+min(data$mean_PM10)
RMSE(pre_str3,test_real3)


cor(pre_str2,test_real)
str(pre_str2)


str(test_real)
str(data)

RMSE(pre_str2,test_real)

mse(pre_str2, test_real)
plot(pre_str2, test_real)

# 학습이 과적합인지 체크
model_results2_train <- compute(pm10_model2, data_train1[-17]) # 17째
pre_str2_train <- model_results2_train$net.result

pre_str2_train <- pre_str2_train*(max(data$mean_PM10)-min(data$mean_PM10))+min(data$mean_PM10)
train_real <- data_train1$mean_PM10*(max(data$mean_PM10)-min(data$mean_PM10))+min(data$mean_PM10)

cor(pre_str2_train,train_real)
RMSE(pre_str2_train,train_real)
mse(pre_str2_train,train_real)
# 모델시각화
plot(pm10_model2)

# 실제,예측
# pre_str2, test_real
library(tidyverse)
library(modelr)
pred_df <- as.data.frame(pre_str3)
str(data_test_raw)
real_df <- as.data.frame(data_test_raw$mean_PM10)
x1 <- cbind(pred_df,real_df)
str(x1)
x1 <- rename(x1, pred = V1)
x1 <- rename(x1, real = 'data_test_raw$mean_PM10')
no <- seq(1:365)
x2 <- cbind(no,x1)
head(x2)
gr1<- ggplot(x2, aes(x=no, y=pred))
gr1 + geom_line(colour = "red") + geom_line(aes(x=no, y=real),colour = "blue") + ylab("서울 미세먼지 농도(PM10)") + xlab("테스트데이터")

# plot. test(365개)중 200~250부분 출력
gr1<- ggplot(x2[200:250,], aes(x=no, y=pred))
gr1 + geom_line(colour = "red") + geom_line(aes(x=x2[200:250,]$no, y=x2[200:250,]$real),colour = "blue")

# plot. 예측값 - 실제값
x2$difference <- x2$pred - x2$real
ggplot(x2, aes(x=no, y=difference)) + geom_point(color="#7896b4") + geom_hline(yintercept=0, linetype='dashed', color='#d84860', size=1.5)

# hist. 예측값 - 실제
ggplot(x2,aes(x=difference)) + geom_histogram(aes(fill=..count..))

x2$difference <- x2$pred_train - x2$train
ggplot(x2, aes(x=no, y=difference)) + geom_point() + abline() + stat_smooth(method = 'lm')
################################################
################################################
################################################
################################################
str(data_raw)
# x <- cbind(pre_str2,data_test$mean_PM10)
# View(x)
# 
# str(data_test)
# model_results <- compute(pm10_model, data_test[-51])
# pre_str <- model_results$net.result
# cor(pre_str,data_test$mean_PM10)
# 
# 
# RMSE(pre_str,data_test$mean_PM10)
# mse(pre_str,data_test$mean_PM10)
# x <- cbind(pre_str,data_test$mean_PM10)
# x <- as.data.frame(x)
# str(x)
# names(x)
# plot(pre_str ,data_test$mean_PM10)
library(ggplot2)
# 서울미세먼지
# 상하이 , 베이징
# 일강수량
setwd('C:\\Study\\[20210426_20211031]AI개발자양성과정\\TeamProject1')
qwer <- read_xlsx("데이터 찐 최종_v4.xlsx")
qwer= as.data.frame(qwer)
plot(qwer$mean_PM10)

ggplot(data=qwer, aes(x=일시, y=mean_PM10)) +
  geom_point(shape=1, size=2, colour="black") +
  geom_point(data=qwer[qwer$mean_PM10>500,],shape=13,size=2,colour="black") +
  ggtitle("Data - fine dust(PM10) in Seoul")

ggplot(data=qwer, aes(x=일시, y=mean_PM10)) +
  geom_point(shape=1, size=2, colour="black") +
  geom_point(data=qwer[qwer$mean_PM10>500,],shape=13,size=2,colour="black") + labs(title="ww")


# 1. 서울시교통량 
# 2. 서울시석유소비량
# 3. 서울시기상정보
# 4. 중국미세먼지
#qwer$

# 1, 서울시교통량 
str(qwer)
library(dplyr)
plot(mean_PM10 ~ day_tr_sum, data=qwer)
res = lm(mean_PM10 ~ day_tr_sum, data=qwer)
abline(col="red",res)
# 2. 서울시석유소비량
# plot(mean_PM10 ~ 등유, data=qwer)
# res = lm(mean_PM10 ~ 등유, data=qwer)
# abline(col="red",res)
plot(mean_PM10 ~ LPG, data=qwer)
res = lm(mean_PM10 ~ LPG, data=qwer)
abline(col="red",res)
# 3. 서울시기상정보 o
plot(mean_PM10 ~ 평균기온, data=qwer)
res = lm(mean_PM10 ~ 평균기온, data=qwer)
abline(col="red",res)
plot(mean_PM10 ~ 평균상대습도, data=qwer)
res = lm(mean_PM10 ~ 평균상대습도, data=qwer)
abline(col="red",res)
# 4. 중국미세먼지 o
plot(mean_PM10 ~ bei_PM10, data=qwer)
res = lm(mean_PM10 ~ bei_PM10, data=qwer)
abline(col="red",res)
plot(mean_PM10 ~ shan_PM10, data=qwer)
res = lm(mean_PM10 ~ shan_PM10, data=qwer)
abline(col="red",res)
