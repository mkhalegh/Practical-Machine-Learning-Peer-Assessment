Course Project-Practical Machine Learning
==============================
### Mona Khaleghy Rad

## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. 

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants and predict the manner in which they did the exercise. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.  This is the "classe" variable in the training set. You may use any of the other variables to predict with. 

## Getting Data


```r
library(knitr)
if(!file.exists("Training.csv")) {
fileURLtr<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileURLtr, destfile="./data/Training.csv", method="curl")
}

Train<-read.csv("./Training.csv")

if(!file.exists("Testing.csv")) {
fileURLts<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileURLts, destfile="./data/Testing.csv", method="curl")
}
Test<-read.csv("./Testing.csv")
```

## Data Cleaning

I keep the variables that doesn't have NAs, then I remove variables with near zero variance. Finally, I remove the irrelevant variables (`X, user_name`, `raw_timestamp_part_1`,`raw_timestamp_part_2`, `cvtd_timestamp`, `new_window`, and `num_window`)


```r
library(caret)
Train<-Train[, colSums(is.na(Train)) == 0]
Train<- Train[, -nearZeroVar(Train)]
Train<-Train[,-c(1:7)]
```

## Data partitioning

We partition the training set into 80% sub training set and 20% sub testing set for future cross validation. This subsetting should be done based on the "classe" variable, which is our training variable. Then since the data is rather large, we sample 1000 random rows for training model. 


```r
TrainPart <- createDataPartition(y = Train$classe, p = 0.8, list = FALSE)
training<-Train[TrainPart,]
testing<-Train[-TrainPart,]
dim(training);dim(testing)
```

```
## [1] 15699    52
```

```
## [1] 3923   52
```

```r
sample<-sample(TrainPart,size=1000,replace=FALSE)
tr<-Train[sample,]
ts<-Train[-sample,]
```

## Training model
I compare three training models, the Rain Forest (RF), which is a very accurate model, the boosting with trees (gbm) model, and the Naive Bayes(nb) model, which is a model-based prediction model.


```r
RF<-train(classe~., data=tr, method="rf")
GBM<-train(classe~.,data=tr,method="gbm",verbose = FALSE)
NB<-train(classe~.,data=tr,method="nb")
```


```r
RFpred=predict(RF,newdata=ts)
RFaccuracy=sum(RFpred == ts$classe)/length(ts$classe)
cat("RFaccuracy: ", RFaccuracy)
```

```
## RFaccuracy:  0.9067769
```

```r
GBMpred=predict(GBM,newdata=ts)
GBMaccuracy=sum(GBMpred == ts$classe)/length(ts$classe)
cat("GBMaccuracy: ", GBMaccuracy)
```

```
## GBMaccuracy:  0.8822898
```

```r
NBpred=predict(NB,newdata=ts)
NBaccuracy=sum(NBpred == ts$classe)/length(ts$classe)
cat("NBaccuracy: ", NBaccuracy)
```

```
## NBaccuracy:  0.6556224
```

From comparison of accuracies of RF, GBM, and NB methods, we can see that the RF method has the highest accuracy. Therefore, we pick the RF method for our predictions. 

We now want to find the top-20 features of the RF model and train on the whole data set partition.


```r
library(caret)
varImp(RF)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 51)
## 
##                   Overall
## pitch_forearm      100.00
## yaw_belt            80.59
## magnet_dumbbell_z   67.68
## magnet_belt_y       44.05
## pitch_belt          41.05
## magnet_belt_z       40.09
## magnet_dumbbell_y   39.00
## magnet_dumbbell_x   31.19
## roll_forearm        30.86
## accel_belt_z        30.77
## roll_dumbbell       30.33
## accel_forearm_x     24.71
## accel_dumbbell_y    22.26
## magnet_arm_z        15.91
## accel_dumbbell_z    15.86
## yaw_dumbbell        14.69
## magnet_forearm_x    14.06
## magnet_forearm_z    12.65
## gyros_belt_z        12.38
## magnet_belt_x       11.79
```

The new testing and training dataset will be a subset of the original test and train, but only includes the top 20 important variables.


```r
TrainPart <- createDataPartition(y = Train$classe, p = 0.8, list = FALSE)
training<-Train[TrainPart,]
testing<-Train[-TrainPart,]

training<-subset(training, select=c("yaw_belt","pitch_forearm","magnet_dumbbell_z","pitch_belt","accel_belt_z","magnet_dumbbell_y","magnet_dumbbell_x","magnet_belt_z","roll_forearm","roll_dumbbell","accel_dumbbell_z","magnet_belt_y","magnet_belt_x","gyros_belt_z","accel_dumbbell_y","yaw_dumbbell","roll_arm","accel_forearm_x","magnet_forearm_z","total_accel_dumbbell", "classe"))

testing<-subset(testing, select=c("yaw_belt","pitch_forearm","magnet_dumbbell_z","pitch_belt","accel_belt_z","magnet_dumbbell_y","magnet_dumbbell_x","magnet_belt_z","roll_forearm","roll_dumbbell","accel_dumbbell_z","magnet_belt_y","magnet_belt_x","gyros_belt_z","accel_dumbbell_y","yaw_dumbbell","roll_arm","accel_forearm_x","magnet_forearm_z","total_accel_dumbbell", "classe"))
```

Now we can use cross validation for training control and apply the RF model on the whole training set. 


```r
tc<-trainControl(method = "cv", number = 7, verboseIter=FALSE , preProcOptions="pca", allowParallel=TRUE)
RFmodel<-train(classe~.,data=training,method="rf",trControl=tc)
```

The expected out of sample error corresponds to the quantity of (1-accuracy) in the cross validation data. On the other hand, accuracy is the proportion of correct classified observation to the total sample in the sub-testing data set. Expected accuracy is the expected accuracy in the out of sample data set (i.e. original testing data set). Thus, the expected value of the out of sample error corresponds to the expected number of missclassified observations/total observations in the Test data set, which is the quantity of (1-accuracy) from the cross-validation data set. 


```r
RFmodel$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 11
## 
##         OOB estimate of  error rate: 0.83%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4456    3    1    3    1 0.001792115
## B   19 2994   22    1    2 0.014483213
## C    0   20 2703   15    0 0.012783053
## D    0    2   23 2544    4 0.011270890
## E    0    1    7    7 2871 0.005197505
```

The expected out of sample error rate for the Random Forest model is ~1 %.

## Cross Validation with sub test set

After choosing our best model, we can now apply the model for prediction on the test set. I also look at the confusion matrix and the out of sample error rate. We expect a low error rate for the random forest model.


```r
CV <- predict(RFmodel, testing)
confusionMatrix(CV, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1114    3    0    0    0
##          B    2  747    5    0    0
##          C    0    8  674    6    0
##          D    0    0    5  637    5
##          E    0    1    0    0  716
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9911          
##                  95% CI : (0.9876, 0.9938)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9887          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9842   0.9854   0.9907   0.9931
## Specificity            0.9989   0.9978   0.9957   0.9970   0.9997
## Pos Pred Value         0.9973   0.9907   0.9797   0.9845   0.9986
## Neg Pred Value         0.9993   0.9962   0.9969   0.9982   0.9984
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1904   0.1718   0.1624   0.1825
## Detection Prevalence   0.2847   0.1922   0.1754   0.1649   0.1828
## Balanced Accuracy      0.9986   0.9910   0.9905   0.9938   0.9964
```

Now it is the time to apply our RF model on 20 sample tests in the Test data.


```r
test<-subset(Test,select=c("yaw_belt","pitch_forearm","magnet_dumbbell_z","pitch_belt","accel_belt_z","magnet_dumbbell_y","magnet_dumbbell_x","magnet_belt_z","roll_forearm","roll_dumbbell","accel_dumbbell_z","magnet_belt_y","magnet_belt_x","gyros_belt_z","accel_dumbbell_y","yaw_dumbbell","roll_arm","accel_forearm_x","magnet_forearm_z","total_accel_dumbbell", "classe"))

dim(test)
TestPred<-predict(RFmodel,newdata=Test)
length(TestPred)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(TestPred)
```
