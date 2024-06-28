library(testthat)
library(caret)
library(C50)
library(randomForest)
library(dplyr)

# Load the data
data <- read.csv("/path/to/your/file.csv")

# Test data loading
test_that("Data is loaded correctly", {
  expect_true(ncol(data) > 0)
  expect_true(nrow(data) > 0)
})

# Define the possible destinations for the multiple-choice question
destinations <- c("Mountain sites", "Beaches in Sabah", "Islands")

# Test binary column creation
test_that("Binary columns are created correctly", {
  for (dest in destinations) {
    data[[dest]] <- ifelse(grepl(dest, data$VisitedDestinations), 1, 0)
    expect_true(all(data[[dest]] %in% c(0, 1)))
  }
})

# Drop the original multiple-choice column
data <- data[, !colnames(data) %in% "VisitedDestinations"]

# Test column dropping
test_that("Original multiple-choice column is dropped", {
  expect_false("VisitedDestinations" %in% colnames(data))
})

# Convert columns to factors and then to numeric
convert_to_numeric <- function(data, column, levels, ordered = FALSE) {
  return(as.numeric(factor(data[[column]], levels = levels, ordered = ordered)))
}

test_that("Columns are converted to numeric correctly", {
  data$AcademicYear <- convert_to_numeric(data, "AcademicYear", c("Year 1", "Year 2", "Year 3", "Year 4", "Year 5"), TRUE)
  data$Gender <- convert_to_numeric(data, "Gender", c("Female", "Male"))
  data$Origin <- convert_to_numeric(data, "Origin", c("Non-Sabahan", "Sabahan"))
  data$TravelFrequency <- convert_to_numeric(data, "TravelFrequency", c("Never", "Annually", "Monthly", "Weekly"), TRUE)
  data$Budget <- convert_to_numeric(data, "Budget", c("<RM100", "RM100 - RM200", ">RM200"), TRUE)
  data$PreferredDistance <- convert_to_numeric(data, "PreferredDistance", c("Short (<1 hour)", "Long (>1 hour)"), TRUE)
  data$TripDuration <- convert_to_numeric(data, "TripDuration", c("1 day", "2 - 3 days", "More than 3 days"), TRUE)
  data$TravelWith <- convert_to_numeric(data, "TravelWith", c("Alone", "Family", "Friends"))
  data$ScenicBeauty <- convert_to_numeric(data, "ScenicBeauty", c("Not important", "Somewhat important", "Neutral", "Important", "Very important"), TRUE)
  data$AdventureOpportunities <- convert_to_numeric(data, "AdventureOpportunities", c("Not important", "Somewhat important", "Neutral", "Important", "Very important"), TRUE)
  data$Accommodation <- convert_to_numeric(data, "Accommodation", c("Not important", "Somewhat important", "Neutral", "Important", "Very important"), TRUE)
  data$CostBudget <- convert_to_numeric(data, "CostBudget", c("Not important", "Somewhat important", "Neutral", "Important", "Very important"), TRUE)
  data$MountainRating <- convert_to_numeric(data, "MountainRating", c("I have not visited this destination", "Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"), TRUE)
  data$BeachRating <- convert_to_numeric(data, "BeachRating", c("I have not visited this destination", "Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"), TRUE)
  data$IslandRating <- convert_to_numeric(data, "IslandRating", c("I have not visited this destination", "Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"), TRUE)
  data$FavouriteDestination <- factor(data$FavouriteDestination, levels = c("Mountains", "Beaches", "Islands"))
  levels(data$FavouriteDestination) <- c("Fave_Mountain", "Fave_Beach", "Fave_Island")
  
  expect_true(is.numeric(data$AcademicYear))
  expect_true(is.numeric(data$Gender))
  expect_true(is.numeric(data$Origin))
  expect_true(is.numeric(data$TravelFrequency))
  expect_true(is.numeric(data$Budget))
  expect_true(is.numeric(data$PreferredDistance))
  expect_true(is.numeric(data$TripDuration))
  expect_true(is.numeric(data$TravelWith))
  expect_true(is.numeric(data$ScenicBeauty))
  expect_true(is.numeric(data$AdventureOpportunities))
  expect_true(is.numeric(data$Accommodation))
  expect_true(is.numeric(data$CostBudget))
  expect_true(is.numeric(data$MountainRating))
  expect_true(is.numeric(data$BeachRating))
  expect_true(is.numeric(data$IslandRating))
  expect_true(is.factor(data$FavouriteDestination))
})

# Integration test for the entire process
test_that("The entire process works correctly", {
  set.seed(123)
  trainIndex <- createDataPartition(data$FavouriteDestination, p = 0.85, list = FALSE)
  train_data <- data[trainIndex, ]
  test_data <- data[-trainIndex, ]

  # Ensure the target is a factor
  train_target <- factor(train_data$FavouriteDestination)
  test_target <- factor(test_data$FavouriteDestination)
  train_features <- train_data[, !names(train_data) %in% "FavouriteDestination"]
  test_features <- test_data[, !names(test_data) %in% "FavouriteDestination"]

  expect_true(is.factor(train_target))
  expect_true(is.factor(test_target))

  baseline_model <- C5.0(train_features, train_target)
  predictions <- predict(baseline_model, test_features)

  cm <- confusionMatrix(predictions, test_target)
  expect_true(cm$overall['Accuracy'] > 0)

  rf_model_baseline <- randomForest(
    x = train_features,
    y = train_target
  )

  rf_predictions <- predict(rf_model_baseline, newdata = test_features)
  rf_cm <- confusionMatrix(rf_predictions, test_target)
  expect_true(rf_cm$overall['Accuracy'] > 0)
})
