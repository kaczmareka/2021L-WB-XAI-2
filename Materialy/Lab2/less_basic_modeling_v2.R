#install.packages("OpenML")
#install.packages("mlr")

library(OpenML)
library(mlr)
library(pROC)
#library(mlr3)
#library(mlr3learners)
#library(mlr3measures)

#set.seed(1)

### MONKS

# pobranie danych
monks <- getOMLDataSet(data.id = 334L)
monks <- monks$data
head(monks)

# Podzial testowy/treningowy

m <- sample(1:nrow(monks), 0.7*nrow(monks))
monks_train <- monks[m,]
monks_test <- monks[-m,]

classif_task <- makeClassifTask(id = "lvr", data = monks_train, target = "class")

# listowanie learnerow ze wsparciem dla prawdopodobieñstw
listLearners(properties = "prob")$class
# listowanie zbioru hiperparametrów
getLearnerParamSet("classif.ranger")

classif_lrn <- makeLearner("classif.ranger", par.vals = list(num.trees = 500, mtry = 3), predict.type = "prob")

getParamSet(classif_lrn)
helpLearnerParam(classif_lrn)
getHyperPars(classif_lrn)


model <- train(classif_lrn, classif_task)

pred_train <- predict(model, newdata = monks_train)$data$prob.0
pred_test <- predict(model, newdata = monks_test)$data$prob.0
roc(monks_train$class, pred_train)
roc(monks_test$class, pred_test)


### TITANIC
data(titanic_imputed, package = "DALEX")
titanic_imputed$survived <- as.factor(titanic_imputed$survived)
m <- sample(1:nrow(titanic_imputed), 0.7*nrow(titanic_imputed))
titanic_train <- titanic_imputed[m,]
titanic_test <- titanic_imputed[-m,]

classif_task <- makeClassifTask(id = "lvr", data = titanic_train, target = "survived")
classif_lrn <- makeLearner("classif.ranger", par.vals = list(num.trees = 2000, mtry = 3), predict.type = "prob")
model <- train(classif_lrn, classif_task)

pred_train <- predict(model, newdata = titanic_train)$data$prob.0
pred_test <- predict(model, newdata = titanic_test)$data$prob.0
roc(titanic_train$survived, pred_train)
roc(titanic_test$survived, pred_test)


### Walidacja krzy¿owa

classif_task <- makeClassifTask(id = "lvr", data = titanic_train, target = "survived")
classif_lrn <- makeLearner("classif.ranger", par.vals = list(num.trees = 60, mtry = 3), predict.type = "prob")
cv <- makeResampleDesc("CV", iters = 7)
r <- resample(classif_lrn, classif_task, cv, measures = mlr::auc, models = TRUE)
r$models
AUC <- r$aggr
AUC



### Zadanie 2 Stworz model liniowy korzystajacz funkcj glm dla danych monks. Porównaj AUC obu modeli na zbiorze testowym
model_linear <- glm(class~., monks_train, family = "binomial")
roc_obj_glm <- roc(monks_test$class, predict(model_linear, monks_test, type = "response"))

pred <- predict(model, newdata = monks_test)$data
pred <- pred$prob.0
roc_obj_ranger <- roc(monks_test$class, pred)
roc_obj_ranger$auc
c("glm" = roc_obj_glm$auc, "ranger" = roc_obj_ranger$auc)
