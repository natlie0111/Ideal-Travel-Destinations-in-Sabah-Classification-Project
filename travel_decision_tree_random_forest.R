# Load required libraries
library(caret)
library(C50)
library(randomForest)
library(dplyr)
library(ggplot2)

# Read the CSV file with simplified column names
data <- read.csv("/path/to/your/file.csv")

# Define the possible destinations for the multiple-choice question
destinations <- c("Mountain sites", "Beaches in Sabah", "Islands")

# Create binary columns for each destination
for (dest in destinations) {
    data[[dest]] <- ifelse(grepl(dest, data$VisitedDestinations), 1, 0)
}

# Drop the original multiple-choice column
data <- data[, !colnames(data) %in% "VisitedDestinations"]

# Convert columns to factors and then to numeric
data$AcademicYear <- as.numeric(factor(data$AcademicYear, levels = c("Year 1", "Year 2", "Year 3", "Year 4", "Year 5"), ordered = TRUE))
data$Gender <- as.numeric(factor(data$Gender, levels = c("Female", "Male")))
data$Origin <- as.numeric(factor(data$Origin, levels = c("Non-Sabahan", "Sabahan")))
data$TravelFrequency <- as.numeric(factor(data$TravelFrequency, levels = c("Never", "Annually", "Monthly", "Weekly"), ordered = TRUE))
data$Budget <- as.numeric(factor(data$Budget, levels = c("<RM100", "RM100 - RM200", ">RM200"), ordered = TRUE))
data$PreferredDistance <- as.numeric(factor(data$PreferredDistance, levels = c("Short (<1 hour)", "Long (>1 hour)"), ordered = TRUE))
data$TripDuration <- as.numeric(factor(data$TripDuration, levels = c("1 day", "2 - 3 days", "More than 3 days"), ordered = TRUE))
data$TravelWith <- as.numeric(factor(data$TravelWith, c("Alone", "Family", "Friends")))
data$ScenicBeauty <- as.numeric(factor(data$ScenicBeauty, levels = c("Not important", "Somewhat important", "Neutral", "Important", "Very important"), ordered = TRUE))
data$AdventureOpportunities <- as.numeric(factor(data$AdventureOpportunities, levels = c("Not important", "Somewhat important", "Neutral", "Important", "Very important"), ordered = TRUE))
data$Accommodation <- as.numeric(factor(data$Accommodation, levels = c("Not important", "Somewhat important", "Neutral", "Important", "Very important"), ordered = TRUE))
data$CostBudget <- as.numeric(factor(data$CostBudget, levels = c("Not important", "Somewhat important", "Neutral", "Important", "Very important"), ordered = TRUE))
data$MountainRating <- as.numeric(factor(data$MountainRating, levels = c("I have not visited this destination", "Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"), ordered = TRUE))
data$BeachRating <- as.numeric(factor(data$BeachRating, levels = c("I have not visited this destination", "Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"), ordered = TRUE))
data$IslandRating <- as.numeric(factor(data$IslandRating, levels = c("I have not visited this destination", "Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"), ordered = TRUE))
data$FavouriteDestination <- factor(data$FavouriteDestination, levels = c("Mountains", "Beaches", "Islands"))

# Update final class labels
levels(data$FavouriteDestination) <- c("Fave_Mountain", "Fave_Beach", "Fave_Island")

# Set seed for reproducibility
set.seed(123)

# Split the data into training (85%) and testing (15%) sets
trainIndex <- createDataPartition(data$FavouriteDestination, p = 0.85, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Separate features and target variable
train_features <- train_data[, !names(train_data) %in% "FavouriteDestination"]
train_target <- train_data$FavouriteDestination
test_features <- test_data[, !names(test_data) %in% "FavouriteDestination"]
test_target <- test_data$FavouriteDestination

# Set seed for reproducibility
set.seed(123)

# Train a baseline decision tree model
baseline_model <- C5.0(train_features, train_target)

# Make predictions on the test set
predictions <- predict(baseline_model, test_features)

# Print confusion matrix
confusionMatrix(predictions, test_target)

# TRIALS
# Define the grid of trials (depth) values to test
trials_grid <- c(1, 5, 10, 15, 20, 25, 30, 35, 40, 50)

# Initialize an empty list to store models and results
trial_results <- data.frame()

# Train and evaluate models with different depths
for (trials in trials_grid) {
    cat("### Training Decision Tree Model with trials =", trials, "###\n")
    model <- C5.0(train_features, train_target, trials = trials)
    predictions <- predict(model, test_features)
    
    # Evaluate the model
    cm <- confusionMatrix(predictions, test_target)
    accuracy <- cm$overall['Accuracy']
    cat("Accuracy for trials =", trials, ":", accuracy, "\n")
    
    # Store the results
    trial_results <- rbind(trial_results, data.frame(Trials = trials, Accuracy = accuracy))
}

# Identify the best model based on accuracy
best_trials <- trial_results[which.max(trial_results$Accuracy), "Trials"]
cat("### Best model is with trials =", best_trials, "###\n")

# Train the best model
best_model <- C5.0(train_features, train_target, trials = best_trials)
predictions <- predict(best_model, test_features)

# Evaluate the best model
confusionMatrix(predictions, test_target)

# MINCASE
# Define the grid of minCases (minimum samples split) values to test
minCases_grid <- c(2, 5, 10, 15, 20, 25, 30, 35, 40, 50)

# Initialize an empty list to store models and results
minCases_results <- data.frame()

# Train and evaluate models with different minCases values
for (minCases in minCases_grid) {
    cat("### Training Decision Tree Model with minCases =", minCases, "###\n")
    model <- C5.0(train_features, train_target, control = C5.0Control(minCases = minCases))
    predictions <- predict(model, test_features)
    
    # Evaluate the model
    cm <- confusionMatrix(predictions, test_target)
    accuracy <- cm$overall['Accuracy']
    cat("Accuracy for minCases =", minCases, ":", accuracy, "\n")
    
    # Store the results
    minCases_results <- rbind(minCases_results, data.frame(MinCases = minCases, Accuracy = accuracy))
}

# Identify the best model based on accuracy
best_minCases <- minCases_results[which.max(minCases_results$Accuracy), "MinCases"]
cat("### Best model is with minCases =", best_minCases, "###\n")

# Train the best model
best_model <- C5.0(train_features, train_target, control = C5.0Control(minCases = best_minCases))
predictions <- predict(best_model, test_features)

# Evaluate the best model
confusionMatrix(predictions, test_target)

# MINLEAF
# Define the grid of minLeaf (minimum samples per leaf) values to test
minLeaf_grid <- c(1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50)

# Initialize an empty list to store models and results
minLeaf_results <- data.frame()

# Train and evaluate models with different minLeaf values
for (minLeaf in minLeaf_grid) {
    cat("### Training Decision Tree Model with minLeaf =", minLeaf, "###\n")
    model <- C5.0(train_features, train_target, control = C5.0Control(minCases = minLeaf))
    predictions <- predict(model, test_features)
    
    # Evaluate the model
    cm <- confusionMatrix(predictions, test_target)
    accuracy <- cm$overall['Accuracy']
    cat("Accuracy for minLeaf =", minLeaf, ":", accuracy, "\n")
    
    # Store the results
    minLeaf_results <- rbind(minLeaf_results, data.frame(MinLeaf = minLeaf, Accuracy = accuracy))
}

# Identify the best model based on accuracy
best_minLeaf <- minLeaf_results[which.max(minLeaf_results$Accuracy), "MinLeaf"]
cat("### Best model is with minLeaf =", best_minLeaf, "###\n")

# Train the best model
best_model <- C5.0(train_features, train_target, control = C5.0Control(minCases = best_minLeaf))
predictions <- predict(best_model, test_features)

# Evaluate the best model
confusionMatrix(predictions, test_target)

# Plot accuracy vs trials values
ggplot(trial_results, aes(x = Trials, y = Accuracy)) +
    geom_line() +
    geom_point() +
    labs(title = "Decision Tree Model Accuracy vs. Trials",
         x = "Trials", y = "Accuracy")

# Plot accuracy vs minCases values
ggplot(minCases_results, aes(x = MinCases, y = Accuracy)) +
    geom_line() +
    geom_point() +
    labs(title = "Decision Tree Model Accuracy vs. MinCases",
         x = "MinCases", y = "Accuracy")

# Plot accuracy vs minLeaf values
ggplot(minLeaf_results, aes(x = MinLeaf, y = Accuracy)) +
    geom_line() +
    geom_point() +
    labs(title = "Decision Tree Model Accuracy vs. MinLeaf",
         x = "MinLeaf", y = "Accuracy")

# Set seed for reproducibility
set.seed(123)

# Define the training control
train_control <- trainControl(method = "cv", number = 5)  # Example of 5-fold cross-validation

# Train a Random Forest model using caret
rf_model <- train(FavouriteDestination ~ ., data = train_data, method = "rf",
                  trControl = train_control, tuneLength = 5)

# Make predictions on the test set
rf_predictions <- predict(rf_model, newdata = test_data)

# Print confusion matrix for Random Forest model
confusionMatrix(rf_predictions, test_data$FavouriteDestination)
