# Coursera Project Machine Learning 
## Human Activity Recognition

Participants performed barbell lifts correctly and incorrectly in 5 different ways.

Goal: predict the manner in which they did the exercise.

### Data Sources

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project comes from this original source: http://groupware.les.inf.puc-rio.br/har. 

### Install Packages

```r
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
```

### Loading Data
Data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

```r
train <- read.csv("train.csv")
test <- read.csv("test.csv")
```

### Cleaning Data
Training set contains 19.622 observations of 160 variables.

```r
set.seed(189)

# Remove Data with NAs
train <- train[,colSums(is.na(train)) == 0]

# Remove Zero Covariates
nzv <- nearZeroVar(train, saveMetrics = TRUE)
train <- train[,nzv$nzv==FALSE]

# Remove unnecessary variables
train <- train[,-c(1,3,4,5,6)]

# Apply the cleaning to the Test dataset an create a variable classe with NAs
test <- test[,names(test) %in% names(train)]
test$classe <- NA
```

Reduced to 54 variables.

### Data Slicing

```r
inTrain <- createDataPartition(y = train$classe, p=0.75, list=FALSE)
my_training <- train[inTrain,]
my_testing <- train[-inTrain,]
```

### Data Model : Random Forests

```r
rfModel <- randomForest(classe ~ ., data = my_training)

# estimated out of sample error : 0.0044
print(rfModel)
```
### Plot variable importance

```r
png("Importance_variables.png")
varImpPlot(rfModel, type=2, main = "Variable Importance Plot", color = "orange")
dev.off()
```

### Prediction

```r
# Predict on my testing dataset
predictRf <- predict(rfModel, my_testing, type = "class")
cM <- confusionMatrix(my_testing$classe, predictRf)

# accuracy : 0.9969413
accuracy <- cM$overall[1]

# Predict for test dataset
result <- predict(rfModel, test, type = "class")
print(result)
```

