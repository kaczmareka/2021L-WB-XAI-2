### Model ###

data(titanic_imputed, package = "DALEX")

model <- ranger::ranger(survived~., data = titanic_imputed, classification = TRUE, probability = TRUE)


###DALEXtra###

library(DALEX)
library(DALEXtra)

explainer <- explain(model = model,
                     data = titanic_imputed[,-8],
                     y = titanic_imputed$survived)


library(mlr)
titanic_imputed_fct <- titanic_imputed
titanic_imputed_fct$survived <- as.factor(titanic_imputed_fct$survived)

classif_task <- makeClassifTask(data = titanic_imputed_fct, target = "survived")
classif_lrn <- makeLearner("classif.svm", predict.type = "prob")
model_mlr <- train(classif_lrn, classif_task)

explainer_mlr <- explain(model = model_mlr,
                         data = titanic_imputed_fct[,-8],
                         y = as.numeric(as.character(titanic_imputed_fct$survived)))

### LIME ###

library("lime")
model_type.dalex_explainer <- DALEXtra::model_type.dalex_explainer
predict_model.dalex_explainer <- DALEXtra::predict_model.dalex_explainer

lime_johnny_mlr <- predict_surrogate(explainer = explainer_mlr, 
                                 new_observation = titanic_imputed_fct[11, -8], 
                                 n_features = 3, 
                                 n_permutations = 1000,
                                 type = "lime")

lime_johnny_ranger <- predict_surrogate(explainer = explainer, 
                                     new_observation = titanic_imputed_fct[11, -8], 
                                     n_features = 3, 
                                     n_permutations = 1000,
                                     type = "lime")

plot(lime_johnny)
plot(lime_johnny_ranger)

### Break Down ###

pp_ranger_bd_1 <- predict_parts(explainer, new_observation = titanic_imputed[11,])
plot(pp_ranger_bd_1)

pp_ranger_bd_2 <- predict_parts(explainer_mlr, new_observation = titanic_imputed[11,])
plot(pp_ranger_bd_2)



# Zadanko

# Wez dowolny zbior, stworz dowolny model. Wygeneruj oraz porównaj wyjasnienie LIME oraz BreakDown

