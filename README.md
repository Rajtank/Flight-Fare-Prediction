# Indian Domestic Flight Fare Prediction

This project revolves around predicting domestic flight prices within India using a dataset encompassing 11 columns and approximately 11,000 observations. The dataset attributes are as follows:

1. Airline: Type of airline the traveler used (e.g., Indigo, SpiceJet, Air India).
2. Date of Journey: Date when the travel occurred.
3. Source: Departure location.
4. Destination: Arrival location.
6. Route: Flight route (e.g., Bengaluru to Mumbai).
7. Departure Time: Take-off time.
8. Arrival Time: Landing time.
9. Duration: Total travel time.
10. Total Stops: Number of stops in the flight.
11. Additional Information: Flight details.
12. Price: Flight price (target variable).

# Data Preprocessing

In order to create a robust machine learning model, the dataset underwent several preprocessing steps:

1. Removed Null Values: Deleted rows containing null values.
2. Date Format: Modified date entries into a consistent format.
3. Text Cleaning: Performed text cleaning using the re module.
4. Feature Engineering: Derived relevant features from existing ones.
5. Encoding Categorical Data: Converted categorical variables into numerical format.

# Implementation

1. Principal Component Analysis (PCA) and Feature Selection

PCA was applied to potentially reduce dimensionality, followed by feature selection to identify the most relevant predictors for the flight price.

2. Machine Learning Models
The following machine learning models were implemented:

Random Forest: Employed Random Forest regressor for predictive modeling.
Decision Tree: Implemented Decision Tree regressor as a baseline model.

3. Hyperparameter Tuning
Hyperparameters of the models were optimized to enhance performance. Techniques like grid search or random search were used for this purpose.
