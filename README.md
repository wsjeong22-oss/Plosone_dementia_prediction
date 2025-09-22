# Plosone_dementia_prediction_current dementia

#CV
install.packages("caret")



#oversampling
install.packages("ROSE")


install.packages("dplyr")


#SVM
install.packages("kernlab")
install.packages('e1071') 

#Lasso
install.packages("glmnet")

#Random Forest
install.packages("randomForest")


#Decision Tree
install.packages("rpart")
install.packages("rpart.plot")

#ROC
install.packages(pROC)

install.packages("haven")


# Gradient Boosting
install.packages("gbm")

library(gbm)
library(haven)
library(caret)
library(ROSE)
library(dplyr)
library(kernlab)
library(e1071)
library(glmnet)
library(randomForest)
library(rpart)
library(rpart.plot)
library(pROC)
library(keras)
library(tensorflow)
library(mxnet)
library(torch)





table(ws20$CHI)
table(ws20$EDUC)


#CV
set.seed(124)

index <- createDataPartition(ws20$CHI, p=.7, list=FALSE, times=1)

train_ws4 <- ws20[index,]
test_ws4 <- ws20[-index,]


table(train_ws4$CHI)
table(test_ws4$CHI)


#oversampling
set.seed(124)


train_ws4 <- ovun.sample(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC,
                         data=train_ws4,
                         method = "under",
                         N = 118)$data

test_ws4 <- ovun.sample(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC,
                        data=test_ws4,
                        method = "over",
                        N = 4072)$data



table(test_ws4$CHI)
table(train_ws4$CHI)




train_ws4$CHI[train_ws4$CHI==1] <- "yes"
train_ws4$CHI[train_ws4$CHI==0] <- "no"
test_ws4$CHI[test_ws4$CHI==1] <- "yes"
test_ws4$CHI[test_ws4$CHI==0] <- "no"



train_ws4$CHI <- as.factor(train_ws4$CHI)
test_ws4$CHI <- as.factor(test_ws4$CHI)


train_ws4$SEX <- as.factor(train_ws4$SEX)
test_ws4$SEX <- as.factor(test_ws4$SEX)
train_ws4$AGE1 <- as.factor(train_ws4$AGE1)
test_ws4$AGE1 <- as.factor(test_ws4$AGE1)
train_ws4$MARRY <- as.factor(train_ws4$MARRY)
test_ws4$MARRY <- as.factor(test_ws4$MARRY)
train_ws4$DON <- as.factor(train_ws4$DON)
test_ws4$DON <- as.factor(test_ws4$DON)
train_ws4$EDUC <- as.factor(train_ws4$EDUC)
test_ws4$EDUC <- as.factor(test_ws4$EDUC)
train_ws4$ALC <- as.factor(train_ws4$ALC)
test_ws4$ALC <- as.factor(test_ws4$ALC)
train_ws4$EXER <- as.factor(train_ws4$EXER)
test_ws4$EXER <- as.factor(test_ws4$EXER)
train_ws4$SOCIAL <- as.factor(train_ws4$SOCIAL)
test_ws4$SOCIAL <- as.factor(test_ws4$SOCIAL)
train_ws4$DEP <- as.factor(train_ws4$DEP)
test_ws4$DEP <- as.factor(test_ws4$DEP)
train_ws4$EAR <- as.factor(train_ws4$EAR)
test_ws4$EAR <- as.factor(test_ws4$EAR)
train_ws4$SEE <- as.factor(train_ws4$SEE)
test_ws4$SEE <- as.factor(test_ws4$SEE)
train_ws4$EAT <- as.factor(train_ws4$EAT)
test_ws4$EAT <- as.factor(test_ws4$EAT)

#확인
class(train_ws4$AGE1)
class(train_ws4$AGE1)
class(train_ws4$DEP)
class(train_ws4$DEP)
class(train_ws4$EAT)
class(train_ws4$EAT)
class(test_ws4$EXER)
class(test_ws4$ALC)
class(test_ws4$SOCIAL)

#확인
table(test_ws4$EDUC)
table(test_ws4$SEX, test_ws4$CHI)
table(test_ws4$ALC, test_ws4$CHI)
table(test_ws4$EXER, test_ws4$CHI)
table(test_ws4$SOCIAL, test_ws4$CHI)
table(test_ws4$DEP, test_ws4$CHI)
table(test_ws4$EAR, test_ws4$CHI)
table(test_ws4$SEE, test_ws4$CHI)
table(test_ws4$EAT,test_ws4$CHI)

data <- data.frame(col1 = train_ws4$AGE1,
                   col2 = train_ws4$CHI)

table(data)

ctrlspecs <- trainControl(method="cv",
                          number=5,
                          savePredictions="all",
                          classProbs=TRUE)



ctrlspecs2 <- trainControl(method="cv",
                           number=5,
                           search='random',
                           savePredictions= T)




#LASSO LOGISTIC REGRESSION

set.seed(124)

model1_tune <- train( CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC,
                      data = train_ws4,
                      method = "glmnet",
                      trControl = ctrlspecs,
                      family = "binomial",
                      tuneGrid = expand.grid(
                        alpha = 1, # Ridge regularization
                        lambda = seq(0.01, 10, length = 1000)
                      ))



model1_tune

# Extract coefficients from best model
coef_lasso <- coef(model1_tune$finalModel, model1_tune$bestTune$lambda)
coef_lasso

# Train final model with selected lambda
model1 <- train(
  CHI ~ SEX + AGE1 + ALC + EXER + SOCIAL + DEP + EAR + SEE + EAT + MARRY + DON + EDUC,
  data = train_ws4,
  method = "glmnet",
  trControl = ctrlspecs,
  family = "binomial",
  tuneGrid = expand.grid(
    alpha = 1,  # Lasso regularization
    lambda = model2_tune$bestTune$lambda
  )
)



#Use the learned model
predictions1 <- predict(model1, newdata=test_ws4)

#create confusion matrix
confusionMatrix(data = predictions1, test_ws4$CHI)




#Roc Curve
predictions_p1 <- predict(model1, newdata=test_ws4, type= "prob")
numeric_predictions1 <- as.numeric(predictions_p2[[1]])

numeric_predictions1



roc_curve1 <- roc(test_ws4$CHI,numeric_predictions1)

roc_curve1


plot(roc_curve1, main = "ROC Curve", col = "black", lwd = 2, legacy.axes = TRUE, asp = NA, yaxt = "n")






#RIDGE LOGISTIC REGRESSION

set.seed(124)

model2_tune <- train( CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC,
                      data = train_ws4,
                      method = "glmnet",
                      trControl = ctrlspecs,
                      family = "binomial",
                      tuneGrid = expand.grid(
                        alpha = 0, # Ridge regularization
                        lambda = seq(0.01, 10, length = 1000)
                      ))



model2_tune

coef_ridge = coef(model2_tune$finalModel, model2_tune$bestTune$lambda)

coef_ridge




model2 <- train( CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC,
                 data = train_ws4,
                 method = "glmnet",
                 trControl = ctrlspecs,
                 family = "binomial",
                 tuneGrid = expand.grid(
                   alpha = 0, # Lasso regularization
                   lambda = 2.73
                 ))




#Use the learned model
predictions2 <- predict(model2, newdata=test_ws4)

#create confusion matrix
confusionMatrix(data = predictions2, test_ws4$CHI)




#Roc Curve
predictions_p2 <- predict(model2, newdata=test_ws4, type= "prob")
numeric_predictions2 <- as.numeric(predictions_p2[[1]])

numeric_predictions2



roc_curve2 <- roc(test_ws4$CHI,numeric_predictions2)

roc_curve2


plot(roc_curve2, main = "ROC Curve", col = "black", lwd = 2, legacy.axes = TRUE, asp = NA, yaxt = "n")






#SVM Linear
set.seed(124)



model3_tune <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC, 
                     data = train_ws4, 
                     method = 'svmLinear',
                     trControl = ctrlspecs2,
                     tuneLength = 100)
model3_tune
model3_tune$bestTune


model3 <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC, 
                data = train_ws4, 
                method = 'svmLinear',
                trControl = ctrlspecs,
                tuneGrid = expand.grid(
                  C=0.05537025
                ))





#create confusion matrix
predictions3 <- predict(model3, newdata=test_ws4)
confusionMatrix(data = predictions3, test_ws4$CHI)



#Roc Curve
predictions_p3 <- predict(model3, newdata=test_ws4, type= "prob")
numeric_predictions3 <- as.numeric(predictions_p3[[1]])

numeric_predictions3



roc_curve3 <- roc(test_ws4$CHI,numeric_predictions3)

roc_curve3


plot(roc_curve3, main = "ROC Curve", col = "black", lwd = 2, legacy.axes = TRUE, asp = NA, yaxt = "n")







#SVM Radial
set.seed(124)

model4_tune <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC, 
                     data = train_ws4, 
                     method = 'svmRadialSigma',
                     trControl = ctrlspecs2,
                     tuneLength = 100)
model4_tune
model4_tune$bestTune


model4 <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC, 
                data = train_ws4, 
                method = 'svmRadialSigma',
                trControl = ctrlspecs,
                tuneGrid = expand.grid(
                  sigma=0.01490787    ,
                  C=0.2836569
                ))




#create confusion matrix
predictions4 <- predict(model4, newdata=test_ws4)

confusionMatrix(data = predictions4, test_ws4$CHI)




#Roc Curve
predictions_p4 <- predict(model4, newdata=test_ws4, type= "prob")
numeric_predictions4 <- as.numeric(predictions_p4[[1]])

numeric_predictions4



roc_curve4 <- roc(test_ws4$CHI,numeric_predictions4, legacy.axes = TRUE)

roc_curve4


plot(roc_curve4, main = "ROC Curve", col = "black", lwd = 2, legacy.axes = TRUE, asp = NA, yaxt = "n")








#Random Forest
set.seed(124)


grid <- expand.grid(
  mtry = c(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
  splitrule = c("gini"),  
  min.node.size = c(1, 5, 10, 15)
)


model5_tune <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC, 
                     data = train_ws4, 
                     method = "ranger",
                     trControl = ctrlspecs, 
                     tuneGrid = grid,
                     num.trees = 2000)

model5_tune


model5 <- randomForest(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC, 
                       data = train_ws4, 
                       ntree = 2000,  # Number of trees in the forest
                       mtry = 2,  # Number of variables randomly sampled at each split
                       nodesize = 15,
                       importance = TRUE)


print (model5)
importance(model5)


#create confusion matrix
predictions5 <- predict(model5, newdata=test_ws4)
confusionMatrix(data = predictions5, test_ws4$CHI)




#Roc Curve
predictions_p5 <- predict(model5, newdata=test_ws4, type= "vote")
predictions_p5

numeric_predictions5 <- as.numeric(predictions_p5[, 2])
numeric_predictions5

roc_curve5 <- roc(test_ws4$CHI,numeric_predictions5)

roc_curve5
plot(roc_curve5, main = "ROC Curve", col = "black", lwd = 2,  legacy.axes = TRUE, asp = NA, yaxt = "n")









#Decision Tree 
set.seed(124)


cp.grid = expand.grid(.cp= (0:10)*0.001)


model6_tune <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC, 
                     data = train_ws4,
                     method = "rpart",
                     trControl = ctrlspecs,
                     tuneGrid = cp.grid)

model6_tune

best.tree = model6_tune$finalModel

prp(best.tree)



model6 <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC, 
                data = train_ws4,
                method = "rpart",
                trControl = ctrlspecs,
                tuneGrid = expand.grid(cp = 0.01))



#create confusion matrix
predictions6 = predict(model6, newdata = test_ws4, type = "raw")
confusionMatrix(data = predictions6, test_ws4$CHI)








# Train the Gradient Boosting Model with the same control specifications
set.seed(124)
model7_tune <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC,
                     data=train_ws4,
                     method="gbm",
                     trControl=ctrlspecs,  # Reusing the same control specification
                     verbose=FALSE,  # Suppress verbose output
                     tuneGrid=expand.grid(
                       n.trees = c(1000, 1500, 2000, 2500, 3000),  # More values for number of trees
                       interaction.depth = c(1, 3, 5, 7, 9),  # Larger range for interaction depth
                       shrinkage = c(0.001, 0.01, 0.05, 0.1, 0.3),  # Smaller and larger learning rates
                       n.minobsinnode = c(5, 10, 20, 25, 30)  # Varying minimum observations per node
                     ))

# Display the tuned model results
print(model7_tune)



model7_gbm <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC,
                    data=train_ws4,
                    method="gbm",
                    trControl=ctrlspecs,  # Reusing the existing ctrl specification
                    verbose=FALSE,  # Disable printing output during training
                    tuneGrid=expand.grid(
                      n.trees = 1000,  # Number of trees
                      interaction.depth = 7,  # Depth of trees
                      shrinkage = 0.3,  # Learning rate
                      n.minobsinnode = 20  # Minimum number of observations per node
                    ))



# Use the learned GBM model to make predictions on the test set
predictions7 <- predict(model7_gbm, newdata=test_ws4)

# Create confusion matrix
confusionMatrix(data=predictions7, test_ws4$CHI)






# ROC Curve for GBM
predictions_p7 <- predict(model7_gbm, newdata=test_ws4, type= "prob")
numeric_predictions7 <- as.numeric(predictions_p7[[1]])
roc_curve7 <- roc(test_ws4$CHI, numeric_predictions7)


roc_curve7


# Plot ROC Curve and Add to the Combined Plot
plot(roc_curve1, main = "ROC Curve", col = "blue", lwd = 1.5, legacy.axes = TRUE, asp = NA, yaxt = "n")  # Logistic 
lines(roc_curve2, col = "black", lwd = 1.5) # Ridge Logistic
lines(roc_curve3, col = "purple", lwd = 1.5) # SVM Linear
lines(roc_curve4, col = "green", lwd = 1.5) # SVM Radial
lines(roc_curve5, col = "red", lwd = 1.5) # Random Forest
lines(roc_curve7, col = "orange", lwd = 1.5) # Gradient Boosting

# Add legend
legend("bottomright", legend = c("Lasso Logistic Regression", "Ridge Logistic Regression", "SVM Linear", "SVM Radial", 
                                 "Random Forest", "Gradient Boosting Machine"), 
       col = c("blue", "black", "purple", "green", "red", "orange"), 
       lwd = 1.5)





install.packages("fastshap")

# Then load it (and randomForest if needed)
library(fastshap)
library(iml)



probs_test <- predict(model5, newdata = train_ws4, type = "prob")
print(colnames(probs_test))



positive_class_name <- colnames(probs_test)[2]  # automatically picks the second column

predfun <- function(object, newdata) {
  # Predict class probabilities (returns a matrix with columns for each factor level)
  probs <- predict(object, newdata = newdata, type = "prob")
  # Return the probability for the “positive” class
  return(probs[, positive_class_name])
}


test_probs <- head(predfun(model5, train_ws4))
print(test_probs) 





predictor_cols <- c("SEX", "AGE1", "ALC", "EXER", "SOCIAL",
                    "DEP", "EAR", "SEE", "EAT", "MARRY", "DON", "EDUC")

X_train <- train_ws4[, predictor_cols]  # no outcome column here



set.seed(123)  # for reproducibility
shap_values <- explain(
  object       = model5,
  X            = X_train,
  pred_wrapper = predfun,
  nsim         = 50
)



mean_abs_shap <- apply(abs(shap_values), 2, mean)
ranked_features <- sort(mean_abs_shap, decreasing = TRUE)
print(ranked_features)







library(ggplot2)
library(dplyr)
library(tidyr)
library(fastshap)





# ------------------------------------------------------------
# Full R script: Compute SHAP on train + test (all rows) and plot
# ------------------------------------------------------------

# 0) Install / load packages (run once)
# -------------------------------------
# install.packages(c(
#   "randomForest",   # for model fitting
#   "fastshap",       # for SHAP estimation
#   "dplyr", "tidyr", # for data wrangling
#   "ggplot2",        # for plotting
#   "ggbeeswarm"      # for beeswarm layout
# ))
library(randomForest)
library(fastshap)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggbeeswarm)

# 1) Fit random forest on train_ws4
# ----------------------------------
# Assumes train_ws4 is already in your workspace with columns:
#   - CHI  (factor outcome with levels “no”/“yes”)
#   - SEX, AGE1, ALC, EXER, SOCIAL, DEP, EAR, SEE, EAT, MARRY, DON, EDUC

# 1a) Convert those 12 predictors to factors (if not already):
cols_to_factor <- c(
  "SEX","AGE1","ALC","EXER","SOCIAL","DEP","EAR","SEE","EAT","MARRY","DON","EDUC"
)
train_ws4[cols_to_factor] <- lapply(train_ws4[cols_to_factor], factor)

# 1b) Fit the model:
set.seed(123)  # for reproducibility
model5 <- randomForest(
  CHI ~ SEX + AGE1 + ALC + EXER + SOCIAL + DEP + EAR + SEE + EAT + MARRY + DON + EDUC,
  data         = train_ws4,
  ntree        = 2000,
  mtry         = 2,
  nodesize     = 15,
  importance   = TRUE
)

# 2) Build X_train and X_test (just the predictors, as factors)
# --------------------------------------------------------------
predictor_cols <- cols_to_factor

# 2a) X_train (n ≈ 118 rows)
X_train <- train_ws4[, predictor_cols]

# 2b) Make sure test_ws4 has the same factor levels, then build X_test (n ≈ 4,072 rows)
#    (Assumes test_ws4 is already loaded in your workspace.)
test_ws4[cols_to_factor] <- lapply(test_ws4[cols_to_factor], factor)
X_test <- test_ws4[, predictor_cols]

# 3) Define a prediction‐wrapper that returns P(CHI = “yes”)
# ------------------------------------------------------------
# (We checked earlier that colnames(predict(model5, type="prob")) = c("no","yes").)
predfun <- function(object, newdata) {
  probs <- predict(object, newdata = newdata, type = "prob")
  return(probs[, "yes"])  # return the “yes” probability
}

# Quick sanity check (optional):
# head(predfun(model5, X_train))  # should be a numeric vector of length 118
# head(predfun(model5, X_test))   # numeric vector length ≈ 4,072

# 4) Compute SHAP on X_train and X_test separately
# ------------------------------------------------
set.seed(123)
shap_train_values <- explain(
  object       = model5,
  X            = X_train,
  pred_wrapper = predfun,
  nsim         = 50
)  # output: 118 × 12 matrix

set.seed(123)
shap_test_values <- explain(
  object       = model5,
  X            = X_test,
  pred_wrapper = predfun,
  nsim         = 50
)  # output: 4,072 × 12 matrix

# 5) Combine SHAP matrices (so you have one big matrix for train+test)
# ---------------------------------------------------------------------
shap_values <- rbind(shap_train_values, shap_test_values)

# 5a) Also combine the raw-feature data frames
X_all <- rbind(X_train, X_test)

# Sanity check: the number of rows must match





shap_long_all <- 
  as.data.frame(shap_all_values) %>%   # (n_train+n_test) × 12 → data.frame
  mutate(row = row_number()) %>%       # add "row" = 1:(n_train+n_test)
  pivot_longer(                        # turn into long format
    cols      = -row,
    names_to  = "feature",
    values_to = "shap_value"
  ) %>%
  
  # 2) Join to X_all (also pivoted longer) so that each (row,feature) has a feature_value
  left_join(
    X_all %>%
      mutate(row = row_number()) %>%
      pivot_longer(
        cols      = -row,
        names_to  = "feature",
        values_to = "feature_value"
      ),
    by = c("row", "feature")
  ) %>%
  
  # 3) Force feature_value to numeric (instead of factor/character)
  mutate(
    feature_value = as.numeric(as.character(feature_value))
  )

# 4) Recompute feature‐ordering by median absolute SHAP over all rows
feature_order_all <- shap_long_all %>%
  group_by(feature) %>%
  summarize(median_abs = median(abs(shap_value))) %>%
  arrange(desc(median_abs)) %>%
  pull(feature)

shap_long_all <- shap_long_all %>%
  mutate(feature_value = as.factor(feature_value))

# Now `shap_long_all$feature_value` may contain levels like "0", "1" (for binary features),
# or "1","2","3","4" (for a 4‐level feature), etc.

# ------------------------------------------------------------
# 1) Compute the x‐axis limits you want (here we force –0.3 → +0.3)
# ------------------------------------------------------------
x_limits <- c(-0.3, 0.3)
x_breaks <- seq(-0.3, 0.3, by = 0.1)
x_labels <- sprintf("%.1f", x_breaks)

# ------------------------------------------------------------
# 2) Plot with a discrete color scale
# ------------------------------------------------------------
ggplot(shap_long_all, aes(x = shap_value, y = feature, color = feature_value)) +
  
  # (A) Dashed vertical line at zero
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", size = 0.3) +
  
  # (B) Beeswarm points (each point is one observation’s SHAP for one feature)
  geom_quasirandom(
    alpha    = 0.7,
    size     = 1.2,
    width    = 0.4,
    varwidth = FALSE
  ) +
  
  # (C) Use a discrete color scale (e.g. "Set1" from RColorBrewer), because feature_value is a factor
  scale_color_brewer(
    palette = "Set1",
    na.value = "grey80"  # if there are any NA feature_values
  ) +
  
  # (D) Minimal theme, keep only horizontal grid lines
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.x = element_line(color = "grey90", size = 0.3),
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.background   = element_rect(fill = "white", color = NA),
    plot.background    = element_rect(fill = "white", color = NA)
  ) +
  
  # (E) Titles & axis labels
  labs(
    title = "SHAP Summary (Beeswarm) for Random Forest Model (All Observations)",
    x     = "SHAP value (impact on model output)",
    y     = "Feature (ordered by median |SHAP|)",
    color = "Category"  # this will label your legend
  ) +
  
  # (F) Tweak font sizes, margin, and legend
  theme(
    plot.title      = element_text(size = 14, face = "bold", hjust = 0),
    axis.text.y     = element_text(
      size   = 10,
      margin = ggplot2::margin(t = 0, r = 8, b = 0, l = 0)
    ),
    axis.text.x     = element_text(size = 10),
    axis.title.y    = element_text(size = 12, vjust = 1),
    axis.title.x    = element_text(size = 12, vjust = -0.5),
    legend.title    = element_text(size = 10),
    legend.text     = element_text(size = 8),
    legend.position = "right"
  ) +
  
  # (G) Reverse y‐axis so the feature with highest median(|SHAP|) is at the top
  scale_y_discrete(limits = rev(levels(shap_long_all$feature))) +
  
  # (H) Force x‐axis limits exactly from –0.3 to +0.3, with tick marks at −0.3, −0.2, …, +0.3
  scale_x_continuous(
    limits = x_limits,
    breaks = x_breaks,
    labels = x_labels
  )



  # Plosone_dementia_prediction_future dementia

  #CV
install.packages("caret")



#oversampling
install.packages("ROSE")


install.packages("dplyr")


#SVM
install.packages("kernlab")
install.packages('e1071') 

#Lasso
install.packages("glmnet")

#Random Forest
install.packages("randomForest")


#Decision Tree
install.packages("rpart")
install.packages("rpart.plot")

#ROC
install.packages(pROC)

install.packages("haven")
# Gradient Boosting
install.packages("gbm")

library(gbm)
library(haven)
library(caret)
library(ROSE)
library(dplyr)
library(kernlab)
library(e1071)
library(glmnet)
library(randomForest)
library(rpart)
library(rpart.plot)
library(pROC)





table(ws20_2years$CHI)
table(ws20_2years$EDUC)


#CV
set.seed(129)

index <- createDataPartition(ws20_2years$CHI, p=.7, list=FALSE, times=1)

train_ws4 <- ws20_2years[index,]
test_ws4 <- ws20_2years[-index,]


table(train_ws4$CHI)
table(test_ws4$CHI)


#oversampling
set.seed(129)


train_ws4 <- ovun.sample(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC + REGION,
                         data=train_ws4,
                         method = "under",
                         N = 92)$data

test_ws4 <- ovun.sample(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC + REGION,
                        data=test_ws4,
                        method = "over",
                        N = 3678)$data



table(test_ws4$CHI)




train_ws4$CHI[train_ws4$CHI==1] <- "yes"
train_ws4$CHI[train_ws4$CHI==0] <- "no"
test_ws4$CHI[test_ws4$CHI==1] <- "yes"
test_ws4$CHI[test_ws4$CHI==0] <- "no"



train_ws4$CHI <- as.factor(train_ws4$CHI)
test_ws4$CHI <- as.factor(test_ws4$CHI)


train_ws4$SEX <- as.factor(train_ws4$SEX)
test_ws4$SEX <- as.factor(test_ws4$SEX)
train_ws4$AGE1 <- as.factor(train_ws4$AGE1)
test_ws4$AGE1 <- as.factor(test_ws4$AGE1)
train_ws4$MARRY <- as.factor(train_ws4$MARRY)
test_ws4$MARRY <- as.factor(test_ws4$MARRY)
train_ws4$DON <- as.factor(train_ws4$DON)
test_ws4$DON <- as.factor(test_ws4$DON)
train_ws4$EDUC <- as.factor(train_ws4$EDUC)
test_ws4$EDUC <- as.factor(test_ws4$EDUC)
train_ws4$REGION <- as.factor(train_ws4$REGION)
test_ws4$REGION <- as.factor(test_ws4$REGION)
train_ws4$ALC <- as.factor(train_ws4$ALC)
test_ws4$ALC <- as.factor(test_ws4$ALC)
train_ws4$EXER <- as.factor(train_ws4$EXER)
test_ws4$EXER <- as.factor(test_ws4$EXER)
train_ws4$SOCIAL <- as.factor(train_ws4$SOCIAL)
test_ws4$SOCIAL <- as.factor(test_ws4$SOCIAL)
train_ws4$DEP <- as.factor(train_ws4$DEP)
test_ws4$DEP <- as.factor(test_ws4$DEP)
train_ws4$EAR <- as.factor(train_ws4$EAR)
test_ws4$EAR <- as.factor(test_ws4$EAR)
train_ws4$SEE <- as.factor(train_ws4$SEE)
test_ws4$SEE <- as.factor(test_ws4$SEE)
train_ws4$EAT <- as.factor(train_ws4$EAT)
test_ws4$EAT <- as.factor(test_ws4$EAT)

#확인
class(train_ws4$AGE1)
class(train_ws4$AGE1)
class(train_ws4$DEP)
class(train_ws4$DEP)
class(train_ws4$EAT)
class(train_ws4$EAT)
class(test_ws4$EXER)
class(test_ws4$ALC)
class(test_ws4$SOCIAL)

#확인
table(test_ws4$CHI)
table(test_ws4$SEX, test_ws4$CHI)
table(test_ws4$ALC, test_ws4$CHI)
table(test_ws4$EXER, test_ws4$CHI)
table(test_ws4$SOCIAL, test_ws4$CHI)
table(test_ws4$DEP, test_ws4$CHI)
table(test_ws4$EAR, test_ws4$CHI)
table(test_ws4$SEE, test_ws4$CHI)
table(test_ws4$EAT,test_ws4$CHI)

data <- data.frame(col1 = train_ws4$AGE1,
                   col2 = train_ws4$CHI)

table(data)

ctrlspecs <- trainControl(method="cv",
                          number=5,
                          savePredictions="all",
                          classProbs=TRUE)



ctrlspecs2 <- trainControl(method="cv",
                           number=5,
                           search='random',
                           savePredictions= T)





#LASSO LOGISTIC REGRESSION

set.seed(124)

model1_tune <- train( CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC,
                      data = train_ws4,
                      method = "glmnet",
                      trControl = ctrlspecs,
                      family = "binomial",
                      tuneGrid = expand.grid(
                        alpha = 1, # Ridge regularization
                        lambda = seq(0.01, 10, length = 1000)
                      ))



model1_tune

# Extract coefficients from best model
coef_lasso <- coef(model1_tune$finalModel, model1_tune$bestTune$lambda)
coef_lasso

# Train final model with selected lambda
model1 <- train(
  CHI ~ SEX + AGE1 + ALC + EXER + SOCIAL + DEP + EAR + SEE + EAT + MARRY + DON + EDUC,
  data = train_ws4,
  method = "glmnet",
  trControl = ctrlspecs,
  family = "binomial",
  tuneGrid = expand.grid(
    alpha = 1,  # Lasso regularization
    lambda = 0.03
  )
)



#Use the learned model
predictions1 <- predict(model1, newdata=test_ws4)

#create confusion matrix
confusionMatrix(data = predictions1, test_ws4$CHI)




#Roc Curve
predictions_p1 <- predict(model1, newdata=test_ws4, type= "prob")
numeric_predictions1 <- as.numeric(predictions_p1[[1]])

numeric_predictions1



roc_curve1 <- roc(test_ws4$CHI,numeric_predictions1)

roc_curve1


plot(roc_curve1, main = "ROC Curve", col = "black", lwd = 2, legacy.axes = TRUE, asp = NA, yaxt = "n")







#RIDGE LOGISTIC REGRESSION

set.seed(129)

model2_tune <- train( CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC + REGION,
                      data = train_ws4,
                      method = "glmnet",
                      trControl = ctrlspecs,
                      family = "binomial",
                      tuneGrid = expand.grid(
                        alpha = 0, # Ridge regularization
                        lambda = seq(0.01, 10, length = 1000)
                      ))



model2_tune

coef_ridge = coef(model2_tune$finalModel, model2_tune$bestTune$lambda)

coef_ridge




model2 <- train( CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC + REGION,
                 data = train_ws4,
                 method = "glmnet",
                 trControl = ctrlspecs,
                 family = "binomial",
                 tuneGrid = expand.grid(
                   alpha = 0, # Lasso regularization
                   lambda = 8.9
                 ))




#Use the learned model
predictions2 <- predict(model2, newdata=test_ws4)

#create confusion matrix
confusionMatrix(data = predictions2, test_ws4$CHI)




#Roc Curve
predictions_p2 <- predict(model2, newdata=test_ws4, type= "prob")
numeric_predictions2 <- as.numeric(predictions_p2[[1]])

numeric_predictions2



roc_curve2 <- roc(test_ws4$CHI,numeric_predictions2)

roc_curve2


plot(roc_curve2, main = "ROC Curve", col = "black", lwd = 2, legacy.axes = TRUE, asp = NA, yaxt = "n")






#SVM Linear
set.seed(129)


model3_tune <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC + REGION, 
                data = train_ws4, 
                method = 'svmLinear',
                trControl = ctrlspecs2,
                tuneLength = 100)

model3_tune$bestTune


model3 <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC + REGION, 
                     data = train_ws4, 
                     method = 'svmLinear',
                     trControl = ctrlspecs,
                     tuneGrid = expand.grid(
                       C=0.1666947
                     ))





#create confusion matrix
predictions3 <- predict(model3, newdata=test_ws4)
confusionMatrix(data = predictions3, test_ws4$CHI)



#Roc Curve
predictions_p3 <- predict(model3, newdata=test_ws4, type= "prob")
numeric_predictions3 <- as.numeric(predictions_p3[[1]])

numeric_predictions3



roc_curve3 <- roc(test_ws4$CHI,numeric_predictions3)

roc_curve3


plot(roc_curve3, main = "ROC Curve", col = "black", lwd = 2, legacy.axes = TRUE, asp = NA, yaxt = "n")







#SVM Radial
set.seed(129)

model4_tune <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC + REGION, 
                     data = train_ws4, 
                     method = 'svmRadialSigma',
                     trControl = ctrlspecs2,
                     tuneLength = 100)
model4_tune
model4_tune$bestTune


model4 <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC + REGION, 
                data = train_ws4, 
                method = 'svmRadialSigma',
                trControl = ctrlspecs,
                tuneGrid = expand.grid(
                  sigma=0.01214017    ,
                  C=4.935686
                ))




#create confusion matrix
predictions4 <- predict(model4, newdata=test_ws4)

confusionMatrix(data = predictions4, test_ws4$CHI)




#Roc Curve
predictions_p4 <- predict(model4, newdata=test_ws4, type= "prob")
numeric_predictions4 <- as.numeric(predictions_p4[[1]])

numeric_predictions4



roc_curve4 <- roc(test_ws4$CHI,numeric_predictions4, legacy.axes = TRUE)

roc_curve4


plot(roc_curve4, main = "ROC Curve", col = "black", lwd = 2, legacy.axes = TRUE, asp = NA, yaxt = "n")








#Random Forest
set.seed(129)


grid <- expand.grid(
  mtry = c(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
  splitrule = c("gini"),  
  min.node.size = c(1, 5, 10, 15)
)


model5_tune <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC + REGION, 
                     data = train_ws4, 
                     method = "ranger",
                     trControl = ctrlspecs, 
                     tuneGrid = grid,
                     num.trees = 2000)

model5_tune


model5 <- randomForest(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC + REGION, 
                       data = train_ws4, 
                       ntree = 2000,  # Number of trees in the forest
                       mtry = 2,  # Number of variables randomly sampled at each split
                       nodesize = 15,
                       importance = TRUE)


print (model5)
importance(model5)


#create confusion matrix
predictions5 <- predict(model5, newdata=test_ws4)
confusionMatrix(data = predictions5, test_ws4$CHI)




#Roc Curve
predictions_p5 <- predict(model5, newdata=test_ws4, type= "vote")
predictions_p5

numeric_predictions5 <- as.numeric(predictions_p5[, 2])
numeric_predictions5

roc_curve5 <- roc(test_ws4$CHI,numeric_predictions5)

roc_curve5
plot(roc_curve5, main = "ROC Curve", col = "black", lwd = 2,  legacy.axes = TRUE, asp = NA, yaxt = "n")









#Decision Tree 
set.seed(129)


cp.grid = expand.grid(.cp= (0:10)*0.001)


model6_tune <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC + REGION, 
                     data = train_ws4,
                     method = "rpart",
                     trControl = ctrlspecs,
                     tuneGrid = cp.grid)

model6_tune

best.tree = model6_tune$finalModel


prp(best.tree)



model6 <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC + REGION, 
                data = train_ws4,
                method = "rpart",
                trControl = ctrlspecs,
                tuneGrid = expand.grid(cp = 0))



#create confusion matrix
predictions6 = predict(model6, newdata = test_ws4, type = "raw")
confusionMatrix(data = predictions6, test_ws4$CHI)




# Train the Gradient Boosting Model with the same control specifications
set.seed(124)
model7_tune <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC,
                     data=train_ws4,
                     method="gbm",
                     trControl=ctrlspecs,  # Reusing the same control specification
                     verbose=FALSE,  # Suppress verbose output
                     tuneGrid=expand.grid(
                       n.trees = c(1000, 1500, 2000, 2500, 3000),  # More values for number of trees
                       interaction.depth = c(1, 3, 5, 7, 9),  # Larger range for interaction depth
                       shrinkage = c(0.001, 0.01, 0.05, 0.1, 0.3),  # Smaller and larger learning rates
                       n.minobsinnode = c(5, 10, 20, 25, 30)  # Varying minimum observations per node
                     ))

# Display the tuned model results
print(model7_tune)



model7_gbm <- train(CHI ~ SEX + AGE1 + ALC +EXER + SOCIAL + DEP + EAR + SEE + EAT+ MARRY + DON + EDUC,
                    data=train_ws4,
                    method="gbm",
                    trControl=ctrlspecs,  # Reusing the existing ctrl specification
                    verbose=FALSE,  # Disable printing output during training
                    tuneGrid=expand.grid(
                      n.trees = 1000,  # Number of trees
                      interaction.depth = 1,  # Depth of trees
                      shrinkage = 0.05,  # Learning rate
                      n.minobsinnode = 10  # Minimum number of observations per node
                    ))



# Use the learned GBM model to make predictions on the test set
predictions7 <- predict(model7_gbm, newdata=test_ws4)

# Create confusion matrix
confusionMatrix(data=predictions7, test_ws4$CHI)






# ROC Curve for GBM
predictions_p7 <- predict(model7_gbm, newdata=test_ws4, type= "prob")
numeric_predictions7 <- as.numeric(predictions_p7[[1]])
roc_curve7 <- roc(test_ws4$CHI, numeric_predictions7)


roc_curve7


# Plot ROC Curve and Add to the Combined Plot
plot(roc_curve1, main = "ROC Curve", col = "blue", lwd = 1.5, legacy.axes = TRUE, asp = NA, yaxt = "n")  # Logistic 
lines(roc_curve2, col = "black", lwd = 1.5) # Ridge Logistic
lines(roc_curve3, col = "purple", lwd = 1.5) # SVM Linear
lines(roc_curve4, col = "green", lwd = 1.5) # SVM Radial
lines(roc_curve5, col = "red", lwd = 1.5) # Random Forest
lines(roc_curve7, col = "orange", lwd = 1.5) # Gradient Boosting

# Add legend
legend("bottomright", legend = c("Lasso Logistic Regression", "Ridge Logistic Regression", "SVM Linear", "SVM Radial", 
                                 "Random Forest", "Gradient Boosting Machine"), 
       col = c("blue", "black", "purple", "green", "red", "orange"), 
       lwd = 1.5)

d








install.packages("fastshap")

# Then load it (and randomForest if needed)
library(fastshap)
library(iml)



probs_test <- predict(model5, newdata = train_ws4, type = "prob")
print(colnames(probs_test))



positive_class_name <- colnames(probs_test)[2]  # automatically picks the second column

predfun <- function(object, newdata) {
  # Predict class probabilities (returns a matrix with columns for each factor level)
  probs <- predict(object, newdata = newdata, type = "prob")
  # Return the probability for the “positive” class
  return(probs[, positive_class_name])
}


test_probs <- head(predfun(model5, train_ws4))
print(test_probs) 





predictor_cols <- c("SEX", "AGE1", "ALC", "EXER", "SOCIAL",
                    "DEP", "EAR", "SEE", "EAT", "MARRY", "DON", "EDUC", "REGION")

X_train <- train_ws4[, predictor_cols]  # no outcome column here



set.seed(123)  # for reproducibility
shap_values <- explain(
  object       = model5,
  X            = X_train,
  pred_wrapper = predfun,
  nsim         = 50
)



mean_abs_shap <- apply(abs(shap_values), 2, mean)
ranked_features <- sort(mean_abs_shap, decreasing = TRUE)
print(ranked_features)







library(ggplot2)
library(dplyr)
library(tidyr)
library(fastshap)





# ------------------------------------------------------------
# Full R script: Compute SHAP on train + test (all rows) and plot
# ------------------------------------------------------------

# 0) Install / load packages (run once)
# -------------------------------------
# install.packages(c(
#   "randomForest",   # for model fitting
#   "fastshap",       # for SHAP estimation
#   "dplyr", "tidyr", # for data wrangling
#   "ggplot2",        # for plotting
#   "ggbeeswarm"      # for beeswarm layout
# ))
library(randomForest)
library(fastshap)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggbeeswarm)

# 1) Fit random forest on train_ws4
# ----------------------------------
# Assumes train_ws4 is already in your workspace with columns:
#   - CHI  (factor outcome with levels “no”/“yes”)
#   - SEX, AGE1, ALC, EXER, SOCIAL, DEP, EAR, SEE, EAT, MARRY, DON, EDUC

# 1a) Convert those 12 predictors to factors (if not already):
cols_to_factor <- c(
  "SEX","AGE1","ALC","EXER","SOCIAL","DEP","EAR","SEE","EAT","MARRY","DON","EDUC"
)
train_ws4[cols_to_factor] <- lapply(train_ws4[cols_to_factor], factor)

# 1b) Fit the model:
set.seed(123)  # for reproducibility
model5 <- randomForest(
  CHI ~ SEX + AGE1 + ALC + EXER + SOCIAL + DEP + EAR + SEE + EAT + MARRY + DON + EDUC,
  data         = train_ws4,
  ntree        = 2000,
  mtry         = 2,
  nodesize     = 15,
  importance   = TRUE
)

# 2) Build X_train and X_test (just the predictors, as factors)
# --------------------------------------------------------------
predictor_cols <- cols_to_factor

# 2a) X_train (n ≈ 118 rows)
X_train <- train_ws4[, predictor_cols]

# 2b) Make sure test_ws4 has the same factor levels, then build X_test (n ≈ 4,072 rows)
#    (Assumes test_ws4 is already loaded in your workspace.)
test_ws4[cols_to_factor] <- lapply(test_ws4[cols_to_factor], factor)
X_test <- test_ws4[, predictor_cols]

# 3) Define a prediction‐wrapper that returns P(CHI = “yes”)
# ------------------------------------------------------------
# (We checked earlier that colnames(predict(model5, type="prob")) = c("no","yes").)
predfun <- function(object, newdata) {
  probs <- predict(object, newdata = newdata, type = "prob")
  return(probs[, "yes"])  # return the “yes” probability
}

# Quick sanity check (optional):
# head(predfun(model5, X_train))  # should be a numeric vector of length 118
# head(predfun(model5, X_test))   # numeric vector length ≈ 4,072

# 4) Compute SHAP on X_train and X_test separately
# ------------------------------------------------
set.seed(123)
shap_train_values <- explain(
  object       = model5,
  X            = X_train,
  pred_wrapper = predfun,
  nsim         = 50
)  # output: 118 × 12 matrix

set.seed(123)
shap_test_values <- explain(
  object       = model5,
  X            = X_test,
  pred_wrapper = predfun,
  nsim         = 50
)  # output: 4,072 × 12 matrix



X_test$SEE <- factor(X_test$SEE, levels = c("0", "1"), labels = c("1", "0"))
X_train$SEE <- factor(X_train$SEE, levels = c("0", "1"), labels = c("1", "0"))
X_test$DON <- factor(X_test$DON, levels = c("0", "1","2","3"), labels = c("1", "2","3","4"))
X_train$DON <- factor(X_train$DON, levels = c("0", "1","2","3"), labels = c("1", "2","3","4"))
X_test$MARRY <- factor(X_test$MARRY, levels = c("1", "2"), labels = c("1", "0"))
X_train$MARRY <- factor(X_train$MARRY, levels = c("1", "2"), labels = c("1", "0"))

table (X_test$MARRY)

# 5) Combine SHAP matrices (so you have one big matrix for train+test)
# ---------------------------------------------------------------------
shap_all_values <- rbind(shap_train_values, shap_test_values)

# 5a) Also combine the raw-feature data frames
X_all <- rbind(X_train, X_test)

# Sanity check: the number of rows must match




shap_long_all <- 
  as.data.frame(shap_all_values) %>%   # (n_train+n_test) × 12 → data.frame
  mutate(row = row_number()) %>%       # add "row" = 1:(n_train+n_test)
  pivot_longer(                        # turn into long format
    cols      = -row,
    names_to  = "feature",
    values_to = "shap_value"
  ) %>%
  
  # 2) Join to X_all (also pivoted longer) so that each (row,feature) has a feature_value
  left_join(
    X_all %>%
      mutate(row = row_number()) %>%
      pivot_longer(
        cols      = -row,
        names_to  = "feature",
        values_to = "feature_value"
      ),
    by = c("row", "feature")
  ) %>%
  
  # 3) Force feature_value to numeric (instead of factor/character)
  mutate(
    feature_value = as.numeric(as.character(feature_value))
  )

# 4) Recompute feature‐ordering by median absolute SHAP over all rows
feature_order_all <- shap_long_all %>%
  group_by(feature) %>%
  summarize(mean_abs = mean(abs(shap_value))) %>%
  arrange(desc(mean_abs)) %>%
  pull(feature)

shap_long_all <- shap_long_all %>%
  mutate(feature_value = as.factor(feature_value),
         feature = factor(feature, levels = feature_order_all))

# Now `shap_long_all$feature_value` may contain levels like "0", "1" (for binary features),
# or "1","2","3","4" (for a 4‐level feature), etc.

# ------------------------------------------------------------
# 1) Compute the x‐axis limits you want (here we force –0.3 → +0.3)
# ------------------------------------------------------------
x_limits <- c(-0.3, 0.3)
x_breaks <- seq(-0.3, 0.3, by = 0.1)
x_labels <- sprintf("%.1f", x_breaks)

# ------------------------------------------------------------
# 2) Plot with a discrete color scale
# ------------------------------------------------------------
ggplot(shap_long_all, aes(x = shap_value, y = feature, color = feature_value)) +
  
  # (A) Dashed vertical line at zero
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", size = 0.3) +
  
  # (B) Beeswarm points (each point is one observation’s SHAP for one feature)
  geom_quasirandom(
    alpha    = 0.7,
    size     = 1.2,
    width    = 0.4,
    varwidth = FALSE
  ) +
  
  # (C) Use a discrete color scale (e.g. "Set1" from RColorBrewer), because feature_value is a factor
  scale_color_brewer(
    palette = "Set1",
    na.value = "grey80"  # if there are any NA feature_values
  ) +
  
  # (D) Minimal theme, keep only horizontal grid lines
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.x = element_line(color = "grey90", size = 0.3),
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.background   = element_rect(fill = "white", color = NA),
    plot.background    = element_rect(fill = "white", color = NA)
  ) +
  
  # (E) Titles & axis labels
  labs(
    x     = "SHAP value (impact on model output)",
    color = "Category"  # this will label your legend
  ) +
  
  # (F) Tweak font sizes, margin, and legend
  theme(
    plot.title      = element_text(size = 14, face = "bold", hjust = 0),
    axis.text.y     = element_text(
      size   = 10,
      margin = ggplot2::margin(t = 0, r = 8, b = 0, l = 0)
    ),
    axis.text.x     = element_text(size = 10),
    axis.title.y    = element_text(size = 12, vjust = 1),
    axis.title.x    = element_text(size = 12, vjust = -0.5),
    legend.title    = element_text(size = 10),
    legend.text     = element_text(size = 8),
    legend.position = "right"
  ) +
  
  # (G) Reverse y‐axis so the feature with highest median(|SHAP|) is at the top
  scale_y_discrete(limits = rev(levels(shap_long_all$feature))) +
  
  # (H) Force x‐axis limits exactly from –0.3 to +0.3, with tick marks at −0.3, −0.2, …, +0.3
  scale_x_continuous(
    limits = x_limits,
    breaks = x_breaks,
    labels = x_labels
  )






# Load necessary libraries
library(ggplot2)
library(dplyr)

# 1. Compute mean(|SHAP|) per feature
mean_abs_shap <- apply(abs(shap_all_values), 2, mean)

# 2. Convert to data frame and sort by importance
importance_df <- data.frame(
  feature = names(mean_abs_shap),
  mean_abs_shap = mean_abs_shap
) %>%
  arrange(desc(mean_abs_shap))  # Sort by importance (largest first)

# 3. Plot: horizontal bar chart like Fig. 4B
ggplot(importance_df, aes(x = reorder(feature, mean_abs_shap), y = mean_abs_shap)) +
  geom_bar(stat = "identity", fill = "#2c7fb8") +
  coord_flip() +
  theme_minimal(base_size = 13) +
  labs(
    title = "Feature Importance (Mean Absolute SHAP)",
    x = "Feature",
    y = "Mean(|SHAP value|)"
  ) +
  theme(
    plot.title      = element_text(face = "bold"),
    axis.text.y     = element_text(size = 11),
    axis.text.x     = element_text(size = 11),
    axis.title.x    = element_text(size = 12, margin = ggplot2::margin(t = 10)),
    axis.title.y    = element_text(size = 12),
    plot.background = element_rect(fill = "white", color = NA),
    panel.grid.major.x = element_line(color = "grey90"),
    panel.grid.major.y = element_blank()
  )


