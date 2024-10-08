# OasisInfobyte - Data Science Internship

## Overview
This repository contains the project submissions for the **Oasis Infobyte Data Science Internship**. The internship involved a series of data science tasks aimed at developing skills in data analysis, visualization, and machine learning. The tasks are implemented using Python, along with popular libraries such as Pandas, NumPy, Matplotlib, and Scikit-learn.

Each task addresses a specific problem, and the objective is to apply relevant data science techniques to derive insights and build models.

## Tasks Overview

### Task 1: Iris Flower Classification
- **Objective**: To classify iris flowers into three different species based on the provided dataset using machine learning models.
- **Steps Involved**:
  - Data Preprocessing: Handling missing values, feature scaling, etc.
  - Exploratory Data Analysis (EDA): Visualizing relationships between features using histograms, scatter plots, and pair plots.
  - Model Building: Implementing machine learning algorithms (e.g., Logistic Regression, K-Nearest Neighbors, Decision Trees) to classify the species of iris flowers.
  - Model Evaluation: Evaluating models using accuracy, precision, recall, and F1-score.
  - **Dataset**: The Iris dataset from UCI Machine Learning Repository.
  
  **Results**: The best performing model achieved an accuracy of **98%** on the test data.

### Task 2: Unemployment Rate Analysis
- **Objective**: To analyze unemployment rates in different regions and visualize trends over time.
- **Steps Involved**:
  - Data Cleaning: Formatting the dataset to remove irrelevant columns and handle missing values.
  - Data Visualization: Using Matplotlib and Seaborn to create line plots, bar charts, and heatmaps to understand unemployment trends.
  - Trend Analysis: Identifying regions with the highest and lowest unemployment rates, and observing trends during specific time periods (e.g., during economic downturns).
  - **Dataset**: Unemployment Rate dataset from the Bureau of Labor Statistics.
  
  **Results**: Provided insights into regional unemployment trends, revealing that the **Northeast** region had the highest unemployment rate at **8%** during the peak of the pandemic.

### Task 3: Sales Prediction for Retail Store
- **Objective**: To predict future sales for a retail store based on historical sales data using machine learning.
- **Steps Involved**:
  - Data Preprocessing: Handling missing values, creating new features (e.g., date features like day, month, year).
  - Exploratory Data Analysis: Understanding seasonal trends, monthly and yearly sales patterns using visualizations.
  - Model Building: Using regression models such as Linear Regression, Random Forest, or XGBoost to predict future sales.
  - Model Evaluation: Evaluating the model using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), etc.
  - **Dataset**: Retail sales data from Kaggle.
  
  **Results**: The model achieved an RMSE of **150** on the test data, providing a reliable sales prediction.

### Task 4: Temperature Change Analysis
- **Objective**: To analyze historical temperature data and predict future temperature trends using regression models.
- **Steps Involved**:
  - Data Preprocessing: Cleaning the dataset and handling any outliers or missing values.
  - Time Series Analysis: Using time-series visualization techniques to analyze temperature changes over the years.
  - Model Building: Implementing time series forecasting models such as ARIMA, and Linear Regression for predicting future temperatures.
  - Model Evaluation: Evaluating the model performance using metrics such as R-squared and Mean Squared Error.
  - **Dataset**: Historical temperature data from NOAA.
  
  **Results**: Forecasted temperature trends with an R-squared of **0.85**, highlighting a potential average increase of **2 degrees Celsius** by 2050.

## Prerequisites
To run the code in this repository, you need to install the following dependencies:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

You can install all dependencies by running:
pip install -r requirements.txt

## How to Run
1. Clone this repository:
git clone https://github.com/sangrambhise/OasisInfobyte.git
2. Navigate to the project folder:
cd OasisInfobyte
3. Install the required libraries (if not installed):
pip install -r requirements.txt
4. Run the Jupyter Notebook for the task you want to explore:
jupyter notebook task1.ipynb

## Folder Structure
OasisInfobyte/

├── Task1_Iris_Classification/

│   ├── task1.ipynb        # Jupyter Notebook for Iris Classification
│   ├── dataset.csv        # Iris dataset
│   └── results.png        # Example of model evaluation results


├── Task2_Unemployment_Analysis/

│   ├── task2.ipynb        # Jupyter Notebook for Unemployment Analysis
│   ├── dataset.csv        # Unemployment dataset
│   └── results.png        # Visualizations of trends


├── Task3_Sales_Prediction/

│   ├── task3.ipynb        # Jupyter Notebook for Sales Prediction
│   ├── dataset.csv        # Retail sales dataset
│   └── results.png        # Predictions and visualizations


├── Task4_Temperature_Analysis/

│   ├── task4.ipynb        # Jupyter Notebook for Temperature Analysis
│   ├── dataset.csv        # Historical temperature data
│   └── results.png        # Forecasted temperature trends


└── requirements.txt       # List of dependencies



## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For any questions or feedback, feel free to contact Sangram Bhise at below given links
LinkedIn: linkedin.com/in/sangrambhise
Project Link: [OasisInfobyte Repository](https://github.com/sangrambhise/Weather-Website)
