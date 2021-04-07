library(DALEX)
library(mlr)

diabetes <- read.csv("diabetes.csv")
head(diabetes)
diabetes$class[diabetes$class == "tested_positive"] <- 1
diabetes$class[diabetes$class == "tested_negative"] <- 0
diabetes$class <- as.numeric(diabetes$class)


model <- ranger::ranger(class~., data = diabetes, classification = TRUE, probability = TRUE)


explainer <- explain(model = model,
                     data = diabetes[,-9],
                     y = diabetes$class)

#Prawdopodobienstwo dla obserwacji 1: ~79% dla wyniku pozytywnego, ~21% dla wyniku negatywnego
predict(model, diabetes[1,])$predictions
pp_ranger_shap_1 <- predict_parts(explainer, new_observation = diabetes[1,], type = "shap", B = 10)
plot(pp_ranger_shap_1)
pp_ranger_bd_1 <- predict_parts(explainer, new_observation = diabetes[1,], type = "break_down")
plot(pp_ranger_bd_1)

#Dla obserwacji 4 najważniejsze zmienne to plas i age, a dla obserwacji 10 jest to preg i mass
pp_ranger_shap_2 <- predict_parts(explainer, new_observation = diabetes[4,], type = "shap", B = 10)
plot(pp_ranger_shap_2)

pp_ranger_shap_3 <- predict_parts(explainer, new_observation = diabetes[10,], type = "shap", B = 10)
plot(pp_ranger_shap_3)
#Wszystkie te 4 zmienne moga miec potencjalnie bardzo duzy wplyw na pozytywny wynik.
#Najczesciej pojawiajacymi sie na pierwszych miejsach zmiennymi sa mass oraz plas oznaczajace
#wspolczynnik BMI oraz stezenie glukozy we krwi. Jednak mocno odbiegajace od normy wartosci takich zmiennych
#jak np preg (liczba ciazy) moga sprawic, ze to wlasnie ta zmienna bedzie miala najwiekszy wplyw na koncowa predykcje.

#Pozytywny i negatywny wpyw zmiennej preg = 4
diabetes_filtered <- diabetes[diabetes$preg == 4,]
pp_ranger_bd_2 <- predict_parts(explainer, new_observation = diabetes_filtered[1,], type = "break_down")
plot(pp_ranger_bd_2)
pp_ranger_bd_3 <- predict_parts(explainer, new_observation = diabetes_filtered[20,], type = "break_down")
plot(pp_ranger_bd_3)
#Zmienna taka jak preg (liczba ciazy) moze miec zupelnie rozne znaczenie w zaleznosci od pozostalych zmiennych
#Przykladowo, 3 ciaze w wieku 40 lat moga nie byc niczym niestandardowym, natomiast 3 ciaze w wieku 20 lat juz tak

