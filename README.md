Step-by-Step Explanation of the App and Model Process
1. Data Generation
Benefit: Ensures a large, diverse dataset for training and testing models, simulating real-world HR data.

Steps:
Define Constants: Set the range for age, years of experience, and salary.
Generate Data: Use Faker to create realistic names, and numpy and random libraries to generate ages, years of experience, and salaries.
Create DataFrame: Store the generated data in a pandas DataFrame.
Save Data: Export the raw and cleaned datasets to CSV files.
2. Data Cleaning
Benefit: Prepares the data for training by ensuring it is clean and formatted correctly.

Steps:
Drop Irrelevant Columns: Remove the 'NAME' column since it's not needed for salary prediction.
Save Cleaned Data: Export the cleaned dataset to a CSV file.
3. Data Splitting
Benefit: Divides the data into training and testing sets to evaluate the model's performance on unseen data.

Steps:
Feature and Target Separation: Separate the features (AGE, YEARS_OF_EXPERIENCE) and the target variable (CURRENT_SALARY).
Train-Test Split: Use train_test_split to create training and testing datasets.
4. Model Training
Benefit: Develops predictive models to estimate salaries based on age and years of experience.

Linear Regression Model:
Train the Model: Fit a linear regression model on the training data.
Save the Model: Export the trained model using joblib.
Random Forest Model:
Train the Model: Fit a Random Forest model on the training data.
Hyperparameter Tuning: Use GridSearchCV to find the best hyperparameters.
Save the Best Model: Export the best Random Forest model using joblib.
5. Model Evaluation
Benefit: Assesses the accuracy and performance of the models, identifying the best one for salary prediction.

Steps:
Predict on Test Data: Use the trained models to predict salaries for the test data.
Calculate Metrics: Evaluate model performance using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) metrics.
Visualize Predictions: Plot actual vs. predicted salaries to visualize model accuracy.
6. Prediction Function
Benefit: Provides an interface to predict salaries for new entries based on age and years of experience.

Steps:
Define Prediction Function: Create a function that uses the best model to predict salary for given age and years of experience.
Example Prediction: Demonstrate the function with example inputs.
7. Streamlit Dashboard
Benefit: Offers an interactive interface for users to explore the data, visualize model performance, and predict salaries.

Steps:
Set Up Streamlit: Create a Streamlit app with a title and sidebar for user input.
User Input: Add sliders for users to input age and years of experience.
Model Prediction: Display the predicted salary based on user input.
Data Visualization: Include histograms, scatter plots, and line charts to visualize the dataset and model predictions.
Model Performance: Show performance metrics for both the Linear Regression and Random Forest models.
Benefits of the App and Model
User-Friendly Interface: The Streamlit dashboard allows HR professionals and users to easily interact with the model and explore salary predictions without requiring technical expertise.
Predictive Insights: The model provides valuable insights into salary expectations based on age and years of experience, aiding in salary negotiations and HR planning.
Data Visualization: Visualizations help users understand the distribution and trends in the dataset, making the data more accessible and actionable.
Model Comparison: The app compares the performance of different models, demonstrating the effectiveness of the Random Forest model over Linear Regression.
Scalability: The approach can be scaled to incorporate more features and more sophisticated models for even better predictions in the future.
Overall, this step-by-step process ensures a comprehensive approach to building, evaluating, and deploying a salary prediction model, making it a practical tool for real-world HR applications.

![output 1](https://github.com/suman10-05-1990/salary-prediction-dashboard/assets/152400389/643cb1cf-c8e7-4243-b748-226c83e58238)

![output2](https://github.com/suman10-05-1990/salary-prediction-dashboard/assets/152400389/8a5dba81-d30c-4d94-874c-b449a378ee3a)
![local server deployment ](https://github.com/suman10-05-1990/salary-prediction-dashboard/assets/152400389/fcd49083-13ad-4cb1-9ccf-256a9dad0eff)
![prediction dashboard1](https://github.com/suman10-05-1990/salary-prediction-dashboard/assets/152400389/e215a7e3-40ea-43ad-9cd2-098a76b56977)
![prediction dashboard2](https://github.com/suman10-05-1990/salary-prediction-dashboard/assets/152400389/e2fc9013-73f8-4163-baad-59913a7bb18c)
![prediction dashboard3](https://github.com/suman10-05-1990/salary-prediction-dashboard/assets/152400389/7d5d60d8-cfb6-4711-9ae5-e38d3f03f6ca)
![prediction dashboard4](https://github.com/suman10-05-1990/salary-prediction-dashboard/assets/152400389/f0136ee5-c064-49ba-8368-f95d35215694)
![prediction dashboard6](https://github.com/suman10-05-1990/salary-prediction-dashboard/assets/152400389/66d4c568-3d4a-4423-800f-4724eeb31fb7)


https://github.com/suman10-05-1990/salary-prediction-dashboard/assets/152400389/ba9e48f2-14b5-432e-b6aa-77de96db7509


![prediction dashboard5](https://github.com/suman10-05-1990/salary-prediction-dashboard/assets/152400389/cb3766c4-9698-466f-a195-bc06637a85ef)


https://github.com/suman10-05-1990/salary-prediction-dashboard/assets/152400389/1c199301-8b4d-41e0-ab4e-d40521d0f628



https://github.com/suman10-05-1990/salary-prediction-dashboard/assets/152400389/20062acb-05f1-4fe5-916f-9f57f88ba143

