### Model ###

data(titanic_imputed, package = "DALEX")

model <- ranger::ranger(survived~., data = titanic_imputed, classification = TRUE, probability = TRUE)

# Podejrzyjmy parametry
model

# Podejrzyjmy predyckje
predict(model, head(titanic_imputed))$predictions


###DALEX###

# Explainer jest to obiekt bedacy portem do wszystkich funckjonalnosci. Opakowuje on model w jednolita strukture ktora potem jest wykorzystywana do tego
# aby wyliczyc wszystkie wyjasnienia. Kluczowe elementy to model, data, y oraz predict_function. Domyslnie DALEX wspiera duzo roznych predict function.

library(DALEX)
library(DALEXtra)

explainer <- explain(model = model,
                     data = titanic_imputed[,-8],
                     y = titanic_imputed$survived) # WAZNE: to musi byc wartosc numerczna dla binarnej kalsyfikacji

# Jezeli verbose = TRUE to otrzymamy podsumowanie naszego modelu

# Preparation of a new explainer is initiated
# -> model label       :  ranger  (  default  )
# -> data              :  2207  rows  8  cols 
# -> target variable   :  2207  values 
# -> predict function  :  yhat.ranger  will be used (  default  )
# -> predicted values  :  No value for predict function target column. (  default  )
# -> model_info        :  package ranger , ver. 0.12.1 , task classification (  default  ) 
# -> predicted values  :  numerical, min =  0.01430847 , mean =  0.3222976 , max =  0.9884335  
# -> residual function :  difference between y and yhat (  default  )
# -> residuals         :  numerical, min =  -0.7825395 , mean =  -0.0001408668 , max =  0.8849883  
# A new explainer has been created!  

explainer$predict_function
?yhat
methods("yhat")


library(mlr)
titanic_imputed_fct <- titanic_imputed
titanic_imputed_fct$survived <- as.factor(titanic_imputed_fct$survived)

classif_task <- makeClassifTask(data = titanic_imputed_fct, target = "survived")
classif_lrn <- makeLearner("classif.svm", predict.type = "prob")
model_mlr <- train(classif_lrn, classif_task)

explainer_mlr <- explain(model = model_mlr,
                         data = titanic_imputed_fct[,-8],
                         y = as.numeric(as.character(titanic_imputed_fct$survived)))

# Widzimy, ¿e mlr tez jest domyslnie wspierany

### Break Down ###

pp_ranger_bd_1 <- predict_parts(explainer, new_observation = titanic_imputed[1,], type = "break_down",
                                order = c("gender", "age", "class", "embarked", "fare", "sibsp", "parch"))
plot(pp_ranger_bd_1)

pp_ranger_bd_2 <- predict_parts(explainer_mlr, new_observation = titanic_imputed[13,])
plot(pp_ranger_bd_2)


### SHAP ###

pp_ranger_shap_1 <- predict_parts(explainer, new_observation = titanic_imputed[1,], type = "shap", B = 10)
plot(pp_ranger_shap_1)

pp_ranger_shap_2 <- predict_parts(explainer, new_observation = titanic_imputed[13,], type = "shap", B = 10)
plot(pp_ranger_shap_2)



# Zadanko

# Wez dowolny zbior, stworz dowolny model oraz wygeneruj dla niego wyjasnienie BreakDown

