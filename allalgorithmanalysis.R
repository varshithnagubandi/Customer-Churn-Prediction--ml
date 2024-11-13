# Install and load required libraries
install.packages("caTools")
install.packages("class")
install.packages("gmodels")
install.packages("e1071")
install.packages("ggplot2")
install.packages("randomForest")
library(caTools)
library(class)
library(gmodels)
library(e1071)
library(ggplot2)
library(randomForest)

# Load data
data <- read.csv(file.choose())
View(data)

# Standardize dropped columns (CustomerID, Gender, Contract_Type, Internet_Service)
drop_columns <- c("CustomerID", "Gender", "Contract_Type", "Internet_Service")
data <- data[, !(names(data) %in% drop_columns)]

# Convert 'Churn' column to a factor
data$Churn <- as.factor(data$Churn)

# Normalize numeric columns
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
data_norm <- as.data.frame(lapply(data[, sapply(data, is.numeric)], normalize))
data_norm$Churn <- data$Churn  # Add Churn back to normalized dataset

# Split data into training and testing sets with a fixed seed for reproducibility
set.seed(42)
split <- sample.split(data_norm$Churn, SplitRatio = 0.7)
train_data <- subset(data_norm, split == TRUE)
test_data <- subset(data_norm, split == FALSE)

# Separate labels
train_labels <- train_data$Churn
test_labels <- test_data$Churn

# Remove label column for model training/testing datasets
train_data <- train_data[, -which(names(train_data) == "Churn")]
test_data <- test_data[, -which(names(test_data) == "Churn")]

# 1. K-Nearest Neighbors
k_value <- 22
knn_pred <- knn(train = train_data, test = test_data, cl = train_labels, k = k_value)
knn_cm <- table(test_labels, knn_pred)
knn_accuracy <- sum(diag(knn_cm)) / sum(knn_cm)
print(paste("KNN Accuracy:", round(knn_accuracy * 100, 2), "%"))

# 2. Naive Bayes
nb_classifier <- naiveBayes(Churn ~ ., data = subset(data_norm, split == TRUE))
nb_pred <- predict(nb_classifier, newdata = subset(data_norm, split == FALSE))
nb_cm <- table(test_labels, nb_pred)
nb_accuracy <- sum(diag(nb_cm)) / sum(nb_cm)
print(paste("Naive Bayes Accuracy:", round(nb_accuracy * 100, 2), "%"))

# 3. Support Vector Machine
svm_classifier <- svm(Churn ~ ., data = subset(data_norm, split == TRUE), type = 'C-classification', kernel = 'linear')
svm_pred <- predict(svm_classifier, newdata = subset(data_norm, split == FALSE))
svm_cm <- table(test_labels, svm_pred)
svm_accuracy <- sum(diag(svm_cm)) / sum(svm_cm)
print(paste("SVM Accuracy:", round(svm_accuracy * 100, 2), "%"))

# 4. Random Forest
rf_classifier <- randomForest(x = train_data, y = train_labels, ntree = 500)
rf_pred <- predict(rf_classifier, newdata = test_data)
rf_cm <- table(test_labels, rf_pred)
rf_accuracy <- sum(diag(rf_cm)) / sum(rf_cm)
print(paste("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%"))

# Visualization function for Confusion Matrices
plot_confusion_matrix <- function(cm, title) {
  cm_df <- as.data.frame(as.table(cm))
  colnames(cm_df) <- c("Actual", "Predicted", "Count")
  ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
    geom_tile() +
    geom_text(aes(label = Count), color = "white", size = 6) +
    scale_fill_gradient(low = "lightblue", high = "blue") +
    labs(title = title, x = "Predicted Labels", y = "Actual Labels") +
    theme_minimal()
}

# Plot confusion matrices for each model
plot_confusion_matrix(knn_cm, "K-Nearest Neighbors Confusion Matrix")
plot_confusion_matrix(nb_cm, "Naive Bayes Confusion Matrix")
plot_confusion_matrix(svm_cm, "SVM Confusion Matrix")
plot_confusion_matrix(rf_cm, "Random Forest Confusion Matrix")

# Accuracy Comparison
accuracy_df <- data.frame(
  Model = c("K-Nearest Neighbors", "Naive Bayes", "Support Vector Machine", "Random Forest"),
  Accuracy = c(knn_accuracy, nb_accuracy, svm_accuracy, rf_accuracy) * 100
)

# Plot accuracy comparison
ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste(round(Accuracy, 2), "%")), vjust = -0.3, color = "black", size = 5) +
  labs(title = "Accuracy Comparison of Different Classification Models", x = "Model", y = "Accuracy (%)") +
  theme_minimal() +
  theme(legend.position = "none")

